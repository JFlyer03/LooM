"""VeRL adapter checks use native DataProto/GRPO, without Ray actors or a GPU model."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verl")
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage

from loom import AutoAgentRunner
from loom.integrations.verl.memory_adapter import (
    align_rollout_batch, compute_stepwise_advantage, validate_memory_training,
)
from loom.integrations.verl.verl_async_manager import SkyAgentLoopManager
from loom.integrations.verl.verl_main_ppo import SkyAgentRewardManager
from loom.integrations.verl.verl_trainer import SkyAgentPPOTrainer


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("norm_by_std", [False, True])
def test_native_grpo_counts_trajectories_once_despite_unequal_step_counts(norm_by_std):
    # A successful two-step trajectory and a failed three-step trajectory share a question.
    # Counting steps would bias the group mean to 2/5; the correct mean is 1/2.
    mask = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 0], [0, 0]])
    data = DataProto.from_dict(tensors={
        "response_mask": mask,
        "token_level_rewards": torch.tensor([[0., 1.], [1., 0.], [0., 0.], [0., 0.], [0., 0.]]),
    }, non_tensors={
        "uid": np.array(["q"] * 5, dtype=object),
        "_loom_repetition_id": np.array([0, 0, 1, 1, 1]),
        "_loom_is_last_step": np.array([False, True, False, False, True]),
    })
    # Native balancing may reorder the rows before advantage computation.
    data = data.select_idxs([4, 0, 2, 1, 3])
    out = compute_stepwise_advantage(data, compute_advantage, adv_estimator="grpo",
                                     norm_adv_by_std_in_grpo=norm_by_std)
    expected = (0.5 / (2 ** -0.5 + 1e-6)) if norm_by_std else 0.5
    assert torch.allclose(out.batch["advantages"][1], torch.tensor([expected, expected]))
    assert torch.allclose(out.batch["advantages"][3], torch.tensor([expected, 0.]))
    assert not out.batch["advantages"][[0, 2, 4]].any()
    assert torch.equal(out.batch["returns"], out.batch["advantages"])
    bad = data.select_idxs([0, 1, 2, 4])
    with pytest.raises(ValueError, match="exactly one terminal"):
        compute_stepwise_advantage(bad, compute_advantage, adv_estimator="grpo")


class Tokenizer:
    padding_side = "left"
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"])

    def pad(self, rows, max_length, **kwargs):
        ids, masks = [], []
        for row in rows:
            tokens = row["input_ids"]
            padding = [0] * (max_length - len(tokens))
            ids.append(padding + tokens if self.padding_side == "left" else tokens + padding)
            masks.append([int(t != 0) for t in ids[-1]])
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class Server:
    async def generate(self, request_id, prompt_ids, sampling_params):
        prompt = Tokenizer().decode(prompt_ids)
        text = "short memory" if "<section>" in prompt else r"\boxed{France}"
        return text, {"output_tokens": Tokenizer().encode(text), "finish_reason": "stop", "logprobs": None}


def test_verl_memory_runner_preserves_reset_steps_and_collapses_validation(tmp_path):
    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent_verl.yaml")
    task.generator.num_trajectories = task.generator.val_config.num_trajectories = 2
    task.memory.chunk_size = 3
    task.memory.max_chunk_number = 3
    path = tmp_path / "task.yaml"
    OmegaConf.save(task, path)
    runner = AutoAgentRunner.from_task(str(path), Server(), Tokenizer())
    source = DataProto.from_dict(tensors={"placeholder": torch.arange(2)}, non_tensors={
        "instance": np.array([
            {"context": "ABCDEF", "question": "Country?", "answers": ["France"]},
            {"context": "ABC", "question": "Country?", "answers": ["France"]},
        ], dtype=object),
        "original_index": np.array(["a", "b"], dtype=object),
    })
    manager = SkyAgentLoopManager.__new__(SkyAgentLoopManager)
    manager.tokenizer = Tokenizer()
    manager.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "prompt_length": 1, "response_length": 1,
    }}})
    rows = asyncio.run(runner.run(source))
    assert len(rows["response_ids"]) == 10  # (2 updates + answer)*2 + (1 update + answer)*2
    rollout = manager._postprocess(rows)
    aligned = align_rollout_batch(source, rollout, 2)
    assert len(aligned) == 10
    assert aligned.non_tensor_batch["original_index"].tolist() == ["a"] * 6 + ["b"] * 4
    assert aligned.non_tensor_batch["uid"].tolist() == ["0"] * 6 + ["1"] * 4
    assert aligned.batch["response_mask"].sum() > 0
    assert aligned.batch["attention_mask"].shape[1] == (
        aligned.batch["prompts"].shape[1] + aligned.batch["responses"].shape[1]
    )
    validation = manager._postprocess(rows, val_mode=True)
    assert len(validation) == 4
    assert validation.non_tensor_batch["rewards"].tolist() == [1.] * 4
    assert validation.non_tensor_batch["_loom_is_last_step"].all()
    assert all(Tokenizer().decode(ids[ids != 0].tolist()) == r"\boxed{France}"
               for ids in validation.batch["responses"])


def test_existing_single_row_rollout_still_repeats_by_num_trajectories():
    source = DataProto.from_dict(tensors={"source": torch.arange(2)})
    rollout = DataProto.from_dict(tensors={"response": torch.arange(4)})
    out = align_rollout_batch(source, rollout, 2)
    assert out.batch["source"].tolist() == [0, 0, 1, 1]
    assert out.non_tensor_batch["uid"][0] == out.non_tensor_batch["uid"][1]
    assert out.non_tensor_batch["uid"][0] != out.non_tensor_batch["uid"][2]


def test_native_validation_reports_one_score_per_complete_trajectory_after_unpadding(tmp_path):
    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent_verl.yaml")
    task.generator.val_config.num_trajectories = 2
    task.memory.chunk_size = 3
    path = tmp_path / "task.yaml"
    OmegaConf.save(task, path)
    tokenizer = Tokenizer()
    runner = AutoAgentRunner.from_task(str(path), Server(), tokenizer)
    manager = SkyAgentLoopManager.__new__(SkyAgentLoopManager)
    manager.tokenizer = tokenizer
    manager.skyagent_generator = runner
    manager.config = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            "prompt_length": 1, "response_length": 1, "val_kwargs": {"do_sample": True},
            "agent": {"num_workers": 2}, "free_cache_engine": False,
        }},
        "reward_model": {"enable": False}, "trainer": {"validation_data_dir": None},
    })
    trainer = SkyAgentPPOTrainer.__new__(SkyAgentPPOTrainer)
    trainer.config = manager.config
    trainer.tokenizer = tokenizer
    trainer.async_rollout_mode = True
    trainer.async_rollout_manager = manager
    trainer.actor_rollout_wg = SimpleNamespace(world_size=1)
    trainer.val_reward_fn = SkyAgentRewardManager(tokenizer, 0)
    trainer.val_dataloader = [{
        "input_ids": torch.tensor([[ord("?")]] * 3),
        "attention_mask": torch.ones(3, 1, dtype=torch.long),
        "position_ids": torch.zeros(3, 1, dtype=torch.long),
        "raw_prompt_ids": np.array([[ord("?")]] * 3, dtype=object),
        "data_source": np.array(["memory"] * 3, dtype=object),
        "instance": np.array([
            {"context": "ABC" * n, "question": "Country?", "answers": ["France"]}
            for n in (1, 2, 3)
        ], dtype=object),
    }]
    captured = {}
    trainer._maybe_log_val_generations = lambda **kwargs: captured.update(kwargs)
    metrics = trainer._validate()
    assert metrics
    assert len(captured["inputs"]) == len(captured["outputs"]) == len(captured["scores"]) == 6
    assert captured["scores"] == [1.] * 6


def test_memory_launch_rejects_v100_before_starting_ray(monkeypatch):
    config = OmegaConf.create({
        "skyrl_agent": {"task_yaml": str(ROOT / "examples/MemAgent/memagent_verl.yaml")},
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "actor_rollout_ref": {"rollout": {"name": "vllm", "mode": "async"}},
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 0))
    with pytest.raises(RuntimeError, match="separate V0 rollout adapter"):
        validate_memory_training(config)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 0))
    validate_memory_training(config)


def test_v0_worker_wraps_native_rollout_and_retains_weight_sync_manager(tmp_path, monkeypatch):
    from loom.integrations.verl.memory_v0 import MemoryV0Worker, MemoryV0Rollout
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    class Engine:
        def generate(self, prompts, sampling_params, use_tqdm):
            assert sampling_params.n == 1 and use_tqdm is False
            prompt = Tokenizer().decode(prompts[0]["prompt_token_ids"])
            text = "memory" if "<section>" in prompt else r"\boxed{France}"
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text=text, token_ids=Tokenizer().encode(text), finish_reason="stop",
            )])]

    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent_verl.yaml")
    task.generator.num_trajectories = task.generator.val_config.num_trajectories = 2
    task.memory.chunk_size = 3
    path = tmp_path / "task.yaml"
    OmegaConf.save(task, path)
    engine = Engine()
    native = SimpleNamespace(inference_engine=engine)
    native_manager = object()
    monkeypatch.setattr(ActorRolloutRefWorker, "_build_rollout",
                        lambda self, trust_remote_code=False: (native, native_manager))
    worker = MemoryV0Worker.__new__(MemoryV0Worker)
    worker.config = OmegaConf.create({"rollout": {
        "memory_task_yaml": str(path), "prompt_length": 1, "response_length": 1,
    }})
    worker.tokenizer = Tokenizer()
    rollout, manager = worker._build_rollout()
    assert manager is native_manager
    assert isinstance(rollout, MemoryV0Rollout)
    assert rollout.inference_engine is engine
    prompts = DataProto.from_dict(tensors={"placeholder": torch.zeros(1)}, non_tensors={
        "instance": np.array([{"context": "ABCDEF", "question": "Country?", "answers": ["France"]}],
                             dtype=object),
    })
    training = rollout.generate_sequences(prompts)
    assert len(training) == 6
    assert training.non_tensor_batch["_loom_is_last_step"].sum() == 2
    prompts.meta_info["val_mode"] = True
    validation = rollout.generate_sequences(prompts)
    assert len(validation) == 2
    assert validation.non_tensor_batch["rewards"].tolist() == [1., 1.]


def test_v0_preflight_allows_v100_but_requires_single_gpu_and_v0(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V1", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 0))
    config = OmegaConf.create({
        "skyrl_agent": {"task_yaml": str(ROOT / "examples/MemAgent/memagent_verl.yaml")},
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
        "actor_rollout_ref": {"model": {}, "actor": {"strategy": "fsdp"}, "rollout": {
            "name": "vllm", "mode": "sync", "memory_task_yaml": "task.yaml",
            "tensor_model_parallel_size": 1,
        }},
    })
    validate_memory_training(config)
    config.trainer.n_gpus_per_node = 2
    with pytest.raises(ValueError, match="one GPU"):
        validate_memory_training(config)
    config.trainer.n_gpus_per_node = 1
    monkeypatch.setenv("VLLM_USE_V1", "1")
    with pytest.raises(ValueError, match="VLLM_USE_V1=0"):
        validate_memory_training(config)
