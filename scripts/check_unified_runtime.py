"""Check native backend imports and optional H100 BF16 kernels in one interpreter."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "slime"], default="all")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("This runtime profile requires Python 3.12")
    import torch
    packages = ["torch", "vllm", "sglang", "skyrl", "verl", "slime", "megatron-core",
                "flash-attn", "flash-attn-4", "transformer-engine"]
    modules = [
        "vllm._custom_ops", "sglang.srt.server_args", "sglang.srt.entrypoints.engine",
        "megatron.training.arguments", "slime.backends.megatron_utils.actor",
        "slime.data.batch_builder", "slime.ray.placement_group",
        "loom.integrations.skyrl_train.memory_main", "loom.integrations.verl.memory_agent_loop",
        "skyrl.backends.skyrl_train.inference_servers.vllm_server_actor",
        "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        "verl.trainer.ppo.v1.utils", "transformer_engine.pytorch",
    ]
    if args.backend == "slime":
        packages = [p for p in packages if p not in ("vllm", "skyrl", "verl")]
        packages += ["nvidia-resiliency-ext", "torch-memory-saver"]
        modules = [m for m in modules if m.startswith(("sglang.", "slime.", "megatron.", "transformer_engine."))]
        modules += ["loom.integrations.slime.memory_rollout",
                    "nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader"]
    report = {"python": sys.version.split()[0], "packages": {p: metadata.version(p) for p in packages}}
    for module in modules:
        importlib.import_module(module)
    report["native_imports"] = modules
    if args.gpu:
        from flash_attn import flash_attn_func
        from sglang.srt.layers.layernorm import RMSNorm
        from transformer_engine.pytorch import Linear

        if not torch.cuda.is_available() or not torch.version.cuda.startswith("13."):
            raise RuntimeError("A CUDA 13 Torch build with a visible NVIDIA GPU is required")
        torch.cuda.set_device(args.device)
        if torch.cuda.get_device_capability(args.device)[0] < 9:
            raise RuntimeError("The unified kernel profile targets Hopper or newer GPUs")
        device = torch.device("cuda", args.device)
        q, k, v = [torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
                   for _ in range(3)]
        output = flash_attn_func(q, k, v, causal=True)
        output.float().square().mean().backward()
        assert torch.isfinite(output).all() and all(torch.isfinite(t.grad).all() for t in (q, k, v))
        layer = Linear(64, 64, params_dtype=torch.bfloat16).to(device)
        x = torch.randn(8, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
        y = layer(x)
        y.float().square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()
        norm = RMSNorm(64, eps=1e-6).to(device=device, dtype=torch.bfloat16)
        normalized = norm(x.detach())
        reference = (x.detach().float() * torch.rsqrt(x.detach().float().square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        torch.testing.assert_close(normalized, reference, rtol=.02, atol=.02)
        if args.backend == "slime":
            from torch_memory_saver import torch_memory_saver as saver

            saver.hook_mode = "torch"
            with saver.region(tag="loom_check", enable_cpu_backup=True):
                allocation = torch.arange(65536, dtype=torch.int32, device=device)
            pointer = allocation.data_ptr()
            expected = allocation.cpu()
            torch.cuda.synchronize(device)
            saver.pause("loom_check")
            saver.resume("loom_check")
            assert allocation.data_ptr() == pointer
            torch.testing.assert_close(allocation.cpu(), expected)
            del allocation
            # The pinned allocator requires its test pool to be destroyed inside the region.
            with saver._impl._with_region_config(tag="loom_check", enable_cpu_backup=True):
                saver._impl._mem_pools.clear()
                torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        report["gpu"] = {"name": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
                         "flash_attention_backward": True, "transformer_engine_backward": True,
                         "sglang_rmsnorm": True}
        if args.backend == "slime":
            report["gpu"]["memory_saver_backup_restore"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
