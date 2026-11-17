"""Native slime grouping checks, without starting model workers."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("slime")
from slime.data.batch_builder import BatchBuilder
from slime.utils.types import Sample
from omegaconf import OmegaConf


@pytest.mark.parametrize("by_std", [False, True])
def test_native_slime_centers_outcomes_once_for_unequal_memory_step_counts(by_std):
    builder = BatchBuilder.__new__(BatchBuilder)
    builder.custom_reward_post_process_func = None
    builder.args = SimpleNamespace(
        reward_key=None, advantage_estimator="grpo", rewards_normalization=True,
        grpo_std_normalization=by_std, n_samples_per_prompt=2, rollout_batch_size=1,
    )
    samples = [Sample(index=i, group_index=8, rollout_id=rid, reward=reward)
               for i, (rid, reward) in enumerate([(0, 1.), (0, 1.), (1, 0.), (1, 0.), (1, 0.)])]
    # Distribution across workers need not preserve the original step order.
    samples = [samples[i] for i in (4, 0, 2, 1, 3)]
    raw, centered = builder._post_process_rewards(samples)
    expected = .5 / (2 ** -.5 + 1e-6) if by_std else .5
    assert raw == [0., 1., 0., 1., 0.]
    assert centered == pytest.approx([-expected, expected, -expected, expected, -expected])


@pytest.mark.parametrize("evaluation", [False, True])
def test_native_hook_preserves_context_tokens_and_per_step_weight_versions(tmp_path, monkeypatch, evaluation):
    from loom)1.integrations.slime.memory_rollout import generate
    from slime.rollout import sglang_rollout
    from slime.utils import http_utils

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

        def apply_chat_template(self, messages, **kwargs):
            return self.encode(messages[0]["content"])

    tokenizer = Tokenizer()
    task = OmegaConf.load(Path(__file__).resolve().parents[1] / "examples/MemAgent/memagent_slime.yaml")
    task.memory.chunk_size = 3
    task.memory.max_memorization_length = task.memory.max_final_response_length = 128
    path = tmp_path / "task.yaml"
    OmegaConf.save(task, path)
    calls = []

    async def post(url, payload, **kwargs):
        calls.append(payload)
        prompt = tokenizer.decode(payload["input_ids"])
        text = "memory" if "<section>" in prompt else r"\boxed{France}"
        return {"text": text, "meta_info": {
            "output_token_logprobs": [[-.5, token, None] for token in tokenizer.encode(text)],
            "weight_version": str(len(calls)), "finish_reason": {"type": "stop"},
        }}

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer))
    args = SimpleNamespace(loom_task=path, sglang_router_ip="localhost", sglang_router_port=1)
    sample = Sample(index=7, group_index=3, metadata={"instance": {
        "context": "ABCDEF", "question": "Country?", "answers": ["France"],
    }})
    result = asyncio.run(generate(args, sample, {"temperature": 1., "max_new_tokens": 128}, evaluation=evaluation))
    segments = [result] if evaluation else result
    assert len(calls) == 3
    assert len(segments) == (1 if evaluation else 3)
    assert "ABC" in tokenizer.decode(calls[0]["input_ids"])
    assert "DEF" not in tokenizer.decode(calls[0]["input_ids"])
    assert "DEF" in tokenizer.decode(calls[1]["input_ids"])
    for i, segment in enumerate(segments, start=2 if evaluation else 0):
        assert segment.rollout_id == 7 and segment.group_index == 3
        assert segment.weight_versions == [str(i + 1)]
        assert segment.tokens == calls[i]["input_ids"] + tokenizer.encode(segment.response)
        assert segment.reward == 1. and segment.status == Sample.Status.COMPLETED
        assert len(segment.loss_mask) == len(segment.rollout_log_probs) == segment.response_length
    if evaluation:
        assert result.metadata["loom_last_step"]
        assert result.response == r"\boxed{France}"
    assert not sample.tokens and sample.reward is None
