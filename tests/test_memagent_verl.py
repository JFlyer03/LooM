"""Native VERL V1 grouping and AgentLoop outputs, without launching Ray actors."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("verl")
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.rollout.replica import TokenOutput
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from loom.integrations.verl.memory_agent_loop import MemoryAgentLoop

ROOT = Path(__file__).resolve().parents[1]


def test_memory_dataset_preserves_reference_answers_for_validation(monkeypatch):
    from loom.integrations.verl.memory_dataset import MemoryDataset, RLHFDataset

    monkeypatch.setattr(RLHFDataset, "__getitem__", lambda self, item: {
        "instance": {"answers": ["Chief of Protocol", "US Chief of Protocol"]},
    })
    example = MemoryDataset.__new__(MemoryDataset)[0]
    assert example["reward_model"] == {
        "style": "rule", "ground_truth": ["Chief of Protocol", "US Chief of Protocol"],
    }


@pytest.mark.parametrize("total,free,raises", [(0, 0, False), (1024, 0, True), (1024, 512, False)])
def test_vllm_shm_check_distinguishes_unlimited_tmpfs_from_a_full_mount(monkeypatch, total, free, raises):
    from vllm.distributed.device_communicators import shm_broadcast

    monkeypatch.setattr(shm_broadcast.shutil, "disk_usage", lambda path: SimpleNamespace(total=total, free=free))
    if raises:
        with pytest.raises(RuntimeError, match="Insufficient space"):
            shm_broadcast.check_shm_free_space(256)
    else:
        shm_broadcast.check_shm_free_space(256)


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
    keys = ["q_0_0", "q_0_1", "q_1_0", "q_1_1", "q_1_2"]
    data = data.select_idxs([4, 0, 2, 1, 3])
    out = compute_advantage_for_multi_trajectories(data, batch_keys=[keys[i] for i in [4, 0, 2, 1, 3]], adv_estimator="grpo",
                                     norm_adv_by_std_in_grpo=norm_by_std)
    expected = (0.5 / (2 ** -0.5 + 1e-6)) if norm_by_std else 0.5
    assert torch.allclose(out.batch["advantages"][1], torch.tensor([expected, expected]))
    assert torch.allclose(out.batch["advantages"][3], torch.tensor([expected, 0.]))
    assert not out.batch["advantages"][[0, 2, 4]].any()
    assert torch.equal(out.batch["returns"], out.batch["advantages"])


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"])


def test_native_agent_loop_emits_each_reset_context_in_one_session(tmp_path):
    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent_verl.yaml")
    task.memory.chunk_size = 3
    task.memory.max_memorization_length = task.memory.max_final_response_length = 128
    path = tmp_path / "memory.yaml"
    OmegaConf.save(task, path)

    class Client:
        calls = 0

        async def generate(self, request_id, prompt_ids, sampling_params):
            self.calls += 1
            prompt = Tokenizer().decode(prompt_ids)
            text = "memory" if "<section>" in prompt else r"\boxed{France}"
            tokens = Tokenizer().encode(text)
            return TokenOutput(
                token_ids=tokens, log_probs=[-.5] * len(tokens), stop_reason="completed",
                extra_fields={"min_global_steps": self.calls - 1, "max_global_steps": self.calls},
            )

    loop = MemoryAgentLoop.__new__(MemoryAgentLoop)
    loop.config = OmegaConf.create({"skyrl_agent": {"task_yaml": str(path)}})
    loop.server_manager = Client()
    loop.tokenizer = Tokenizer()
    outputs = asyncio.run(loop.run({"temperature": .7}, uid="question", instance={
        "context": "ABCDEF", "question": "Country?", "answers": ["France"],
    }))
    assert len(outputs) == 3
    assert "ABC" in Tokenizer().decode(outputs[0].prompt_ids)
    assert "DEF" not in Tokenizer().decode(outputs[0].prompt_ids)
    assert "DEF" in Tokenizer().decode(outputs[1].prompt_ids)
    for i, output in enumerate(outputs):
        packed = output.as_dict()
        assert packed["extra_fields"]["min_global_steps"] == i
        assert packed["extra_fields"]["max_global_steps"] == i + 1
        assert output.reward_score == 1.
        assert packed["rollout_log_probs"].shape == packed["response_mask"].shape
        assert packed["rm_scores"].sum().item() == 1.
    assert OmegaConf.load(path).generator.num_trajectories == 16
