"""Keep reset-context memory steps aligned with VeRL's trajectory-level GRPO."""

import uuid

import numpy as np
import torch


def align_rollout_batch(batch, rollout, num_trajectories):
    source_indices = rollout.non_tensor_batch.get("_loom_source_index")
    if source_indices is None:
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
        return batch.repeat(repeat_times=num_trajectories, interleave=True).union(rollout)
    if any(i < 0 or i >= len(batch) for i in source_indices):
        raise ValueError("Memory rollout source indices are outside the input batch")
    batch = batch.select_idxs(source_indices)
    batch.non_tensor_batch["uid"] = np.array([str(i) for i in source_indices], dtype=object)
    return batch.union(rollout)


def compute_stepwise_advantage(data, compute_advantage, **kwargs):
    if "_loom_is_last_step" not in data.non_tensor_batch:
        return compute_advantage(data, **kwargs)
    if kwargs["adv_estimator"] != "grpo":
        raise ValueError("Memory reset-context trajectories currently support GRPO only")
    keys = list(zip(data.non_tensor_batch["uid"], data.non_tensor_batch["_loom_repetition_id"]))
    last_indices = np.flatnonzero(data.non_tensor_batch["_loom_is_last_step"])
    terminal_keys = [keys[i] for i in last_indices]
    if len(set(terminal_keys)) != len(terminal_keys) or set(keys) != set(terminal_keys):
        raise ValueError("Each memory trajectory must have exactly one terminal step")
    terminal = data.select_idxs(last_indices)
    # Include failed trajectories' zero outcomes in the group; their original loss masks stay zero.
    terminal.batch["response_mask"] = torch.ones_like(terminal.batch["response_mask"])
    terminal = compute_advantage(terminal, **kwargs)
    lookup = {key: i for i, key in enumerate(terminal_keys)}
    indices = torch.tensor([lookup[key] for key in keys], device=data.batch.device)
    for field in ("advantages", "returns"):
        scalar = terminal.batch[field][:, 0]
        data.batch[field] = scalar[indices, None] * data.batch["response_mask"]
    return data


def validate_memory_training(config):
    """Fail before starting Ray when this MemAgent path cannot execute correctly."""
    from omegaconf import OmegaConf

    task_path = OmegaConf.select(config, "skyrl_agent.task_yaml")
    if not task_path:
        return
    task = OmegaConf.load(task_path)
    if task.agent_cls != "loom.agents.memory.MemAgent":
        return
    if task.generator.infer_backend != "verl":
        raise ValueError("MemAgent's VeRL task must set generator.infer_backend=verl")
    if config.algorithm.adv_estimator != "grpo" or config.algorithm.use_kl_in_reward:
        raise ValueError("MemAgent's VeRL adapter requires GRPO and use_kl_in_reward=false; use actor KL loss")
    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm" or rollout.mode not in ("sync", "async"):
        raise ValueError("MemAgent's VeRL adapter requires vLLM rollout")
    if rollout.mode == "sync":
        if not rollout.get("memory_task_yaml"):
            raise ValueError("Use the MemAgent entrypoint to configure the V0 memory worker")
        # shortcut: source indices are worker-local; carry global indices before enabling multiple GPUs.
        if config.trainer.n_gpus_per_node * config.trainer.nnodes != 1 or rollout.tensor_model_parallel_size != 1:
            raise ValueError("The V0 memory adapter currently supports one GPU and TP=1 only")
        if config.actor_rollout_ref.model.get("lora_rank", 0):
            raise ValueError("The V0 memory adapter currently supports full-parameter training, not LoRA")
        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise ValueError("The V0 memory adapter currently supports the FSDP strategy only")
        import os

        if os.environ.get("VLLM_USE_V1") != "0":
            raise ValueError("The synchronous V0 memory adapter requires VLLM_USE_V1=0")
    elif torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError(
            "VeRL 0.5's async rollout uses vLLM V1, which requires CUDA compute capability >= 8.0. "
            "This GPU (e.g. V100, sm_70) needs a separate V0 rollout adapter; setting FP16 or "
            "VLLM_USE_V1=0 alone does not adapt the async worker/server."
        )
