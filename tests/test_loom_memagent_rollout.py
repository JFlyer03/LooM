import asyncio
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")
pytest.importorskip("verl.workers.rollout.async_server")

import numpy as np
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from loom import AutoAgentRunner
from loom.integrations.verl.trajectory_batch import compute_trajectory_grpo
from loom.integrations.verl.verl_async_manager import SkyAgentLoopManager
from loom.integrations.verl.verl_main_ppo import SkyAgentRewardManager
from loom.integrations.verl.verl_trainer import SkyAgentPPOTrainer
from loom.tasks.prepare_memory_data import convert_records


TASK = Path(__file__).resolve().parents[1] / "examples/run_verl/verl_memagent.yaml"


class Tokenizer:
    def __init__(self):
        self.padding_side = "left"

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids if token != 0)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"])

    def pad(self, items, *, max_length, **kwargs):
        ids, masks = [], []
        for item in items:
            tokens = item["input_ids"]
            padding = [0] * (max_length - len(tokens))
            if self.padding_side == "left":
                ids.append(padding + tokens)
                masks.append(padding + [1] * len(tokens))
            else:
                ids.append(tokens + padding)
                masks.append([1] * len(tokens) + padding)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class Engine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        prompt = self.tokenizer.decode(prompt_ids)
        self.calls.append((request_id, prompt, sampling_params))
        await asyncio.sleep(0)
        answer = r"\boxed{Paris}" if "Your answer:" in prompt else "<think>reasoning</think>saved memory"
        return answer, {"output_tokens": tuple(self.tokenizer.encode(answer)), "finish_reason": "stop", "logprobs": None}


def test_memagent_runs_by_task_yaml_and_validation_keeps_the_final_answer(tmp_path):
    cfg = OmegaConf.load(TASK)
    cfg.agent_kwargs.chunk_size = 2
    cfg.generator.num_trajectories = 2
    cfg.data.instance_id_key = "id"
    cfg.generator.val_config.num_trajectories = 2
    yaml_path = tmp_path / "task.yaml"
    OmegaConf.save(cfg, yaml_path)

    records = list(convert_records([
        {"context": "abc", "extra_info": {"question": "Capital?"}, "reward_model": {"ground_truth": ["Paris"]}},
        {"context": "x", "input": "Second?", "answers": "London"},
    ]))
    prompts = DataProto.from_single_dict({
        "instance": np.array([record["instance"] for record in records], dtype=object),
        "id": np.array(["duplicate", "duplicate"], dtype=object),
        "data_source": np.array([record["data_source"] for record in records], dtype=object),
    })
    tokenizer = Tokenizer()
    engine = Engine(tokenizer)
    runner = AutoAgentRunner.from_task(yaml_path, engine, tokenizer)
    manager = SkyAgentLoopManager.__new__(SkyAgentLoopManager)
    manager.config = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"prompt_length": 32, "response_length": 8, "free_cache_engine": False}},
    })
    manager.tokenizer = tokenizer
    manager.skyagent_generator = runner

    train = manager.generate_sequences(prompts)
    assert train.non_tensor_batch["rollout_input"].tolist() == [0] * 6 + [1] * 4
    assert train.non_tensor_batch["rollout_trajectory"].tolist() == [0] * 3 + [1] * 3 + [2] * 2 + [3] * 2
    assert train.non_tensor_batch["rollout_final"].tolist() == [False, False, True] * 2 + [False, True] * 2
    assert train.non_tensor_batch["rewards"].tolist() == [1] * 6 + [0] * 4
    assert train.non_tensor_batch["rollout_sampled_logprobs"].tolist() == [None] * 10
    calls = train.non_tensor_batch["rollout_calls"]
    assert [call[0]["phase"] for call in calls[:3]] == ["memory_update", "memory_update", "answer"]
    assert all(call[0]["policy_id"] == "actor" for call in calls)
    assert len({request_id for request_id, _, _ in engine.calls}) == 10
    assert sum("No previous memory" in prompt for _, prompt, _ in engine.calls) == 4
    answer_prompt = tokenizer.decode(train.batch["prompts"][2])
    assert "<memory>\nsaved memory\n</memory>" in answer_prompt

    prompts.meta_info["val_mode"] = True
    validation = manager.generate_sequences(prompts)
    assert validation.non_tensor_batch["rollout_input"].tolist() == [0, 0, 1, 1]
    assert validation.non_tensor_batch["rewards"].tolist() == [1, 1, 0, 0]
    assert all(call[0]["phase"] == "answer" for call in validation.non_tensor_batch["rollout_calls"])

    trainer = SkyAgentPPOTrainer.__new__(SkyAgentPPOTrainer)
    trainer.config = OmegaConf.create({
        "reward_model": {"enable": False},
        "actor_rollout_ref": {"rollout": {"val_kwargs": {"do_sample": False}, "agent": {"num_workers": 3}}},
        "trainer": {"validation_data_dir": str(tmp_path), "log_val_generations": 0},
    })
    trainer.async_rollout_mode = True
    trainer.async_rollout_manager = manager
    trainer.tokenizer = tokenizer
    trainer.global_steps = 1
    trainer.val_reward_fn = SkyAgentRewardManager(tokenizer, num_examine=0)
    validation_ids = torch.tensor([tokenizer.encode("Capital?"), tokenizer.encode("Second? ")])
    trainer.val_dataloader = [{
        **prompts.non_tensor_batch,
        "raw_prompt_ids": np.array([list(row) for row in validation_ids.tolist()], dtype=object),
        "input_ids": validation_ids,
        "attention_mask": torch.ones_like(validation_ids),
        "position_ids": torch.arange(8).repeat(2, 1),
        "reward_model": np.array([record["reward_model"] for record in records], dtype=object),
    }]
    metrics = trainer._validate()
    assert metrics["val-core/loom_memagent/reward/mean@2"] == 0.5
    dumped = [json.loads(line) for line in (tmp_path / "1.jsonl").read_text().splitlines()]
    assert [row["gts"] for row in dumped] == [["Paris"], ["Paris"], ["London"], ["London"]]
    assert all("Your answer:" in row["input"] for row in dumped)


def test_grpo_counts_each_trajectory_once_with_reordered_fragments_and_masks():
    metadata = {
        "rollout_trajectory": np.array([2, 1, 0, 1, 3, 1, 0]),
        "rollout_final": np.array([True, False, True, False, True, True, False]),
        "uid": np.array(["second", "first", "first", "first", "second", "first", "first"], dtype=object),
    }
    rewards = torch.tensor([[0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0]], dtype=torch.float32)
    masks = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 1], [0, 1]])
    batch = DataProto(
        batch=TensorDict({"token_level_rewards": rewards, "response_mask": masks}, batch_size=[7]),
        non_tensor_batch=metadata,
    )
    compute_trajectory_grpo(batch, norm_adv_by_std=False)
    expected = torch.tensor([0, -0.5, 0.5, -0.5, 0, -0.5, 0.5])[:, None] * masks
    torch.testing.assert_close(batch.batch["advantages"], expected)
    torch.testing.assert_close(batch.batch["returns"], expected)
