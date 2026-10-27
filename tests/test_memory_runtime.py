import ast
import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from loom import AutoAgentRunner
from loom.agents.memory import MemAgent
from loom.config.configuration_utils import TrajectoryConfig
from loom.integrations.base import BackendSpec, register_backend
from loom.integrations.skyrl_train.memory_generator import MemoryGenerator
from loom.memory.recurrent import ANSWER_PROMPT, UPDATE_PROMPT
from loom.tasks.memory import LongContextMemoryTask, extract_boxed_content
from loom.tasks.prepare_memory_data import convert_records

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    """Character tokens make context-reset checks inspectable without a model."""

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")


class Backend:
    def __init__(self, infer_engine=None, **kwargs):
        self.calls = []
        self.no_logprobs = False
        self.truncated = False

    async def async_generate_ids(self, input_ids, sampling_params, request_id):
        prompt = Tokenizer().decode(input_ids)
        self.calls.append((request_id, prompt))
        if "<section>" in prompt:
            text = f"<think>analysis</think>memory-{request_id}"
        else:
            text = r"The answer is \boxed{France}"
        ids = Tokenizer().encode(text)
        await asyncio.sleep(0)
        return text, {
            "output_tokens": ids,
            "finish_reason": "length" if self.truncated else "stop",
            "logprobs": None if self.no_logprobs else [-0.5] * len(ids),
        }


class Input:
    def __init__(self, input_batch):
        self.input_batch = input_batch["env_extras"]


class Output:
    def __init__(self, result):
        self.result = result


@pytest.fixture
def task_yaml(tmp_path):
    register_backend("test", BackendSpec(Backend, Output, Input))
    cfg = OmegaConf.load(ROOT / "examples/memory/memagent.yaml")
    cfg.generator.infer_backend = "test"
    cfg.generator.num_trajectories = 2
    cfg.memory.chunk_size = 3
    cfg.memory.max_chunk_number = 3
    cfg.memory.reward_mode = "exact_match"
    path = tmp_path / "task.yaml"
    OmegaConf.save(cfg, path)
    return path


def instance(context="ABCDEF"):
    return {"context": context, "input": "Which country?", "answers": ["France"]}


def test_prompts_match_legacy_without_importing_old_trainer():
    tree = ast.parse((ROOT / "loom_agent/common/utils.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    assert UPDATE_PROMPT == values["TEMPLATE"]
    assert ANSWER_PROMPT == values["TEMPLATE_FINAL_BOXED"]


def test_runner_isolates_samples_and_preserves_raw_rewards(task_yaml):
    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance(), "__loom_uid": "q1"}]}))
    assert out["traj_rewards"] == [1.0, 1.0]
    assert out["rewards"] == [1.0] * 6  # raw rewards, not zero normalized advantages
    assert out["episode_nums"] == [3, 3]
    assert out["is_last_episode"] == [False, False, True] * 2
    assert out["trajectory_keys"] == [("q1", 0)] * 3 + [("q1", 1)] * 3
    assert all(all(mask) for mask in out["loss_masks"])
    assert all(len(tokens) == len(lp) for tokens, lp in zip(out["response_ids"], out["rollout_logprobs"]))
    by_request = {}
    for request_id, prompt in runner.infer_engine.calls:
        by_request.setdefault(request_id, []).append(prompt)
    assert len(by_request) == 2
    for request_id, prompts in by_request.items():
        assert "No previous memory" in prompts[0]
        assert "ABC" in prompts[0] and "DEF" not in prompts[0]
        assert "DEF" in prompts[1] and "ABC" not in prompts[1]
        assert f"memory-{request_id}" in prompts[1]
        assert "analysis" not in prompts[1]
        assert "<section>" not in prompts[2]
        assert "ABC" not in prompts[2] and "DEF" not in prompts[2]
        for other_id in by_request:
            if other_id != request_id:
                assert other_id not in "".join(prompts)
    assert runner.trajectories == {}


@dataclass
class ID:
    instance_id: str
    repetition_id: int


def native_input():
    return {
        "trajectory_ids": [ID("q1", 0), ID("q1", 1), ID("q2", 0), ID("q2", 1)],
        "env_extras": [{"instance": instance()} for _ in range(4)],
        "batch_metadata": SimpleNamespace(training_phase="train"),
        "sampling_params": {"temperature": 1, "max_tokens": 20},
    }


def test_native_adapter_does_not_double_expand_grpo_groups(task_yaml):
    cfg = SimpleNamespace(step_wise_trajectories=True, task=str(task_yaml))
    generator = MemoryGenerator(cfg, None, Tokenizer())
    out = asyncio.run(generator.generate(native_input()))
    assert len(out["response_ids"]) == 12  # 2 tasks * 2 attempts * 3 calls
    assert sum(out["is_last_step"]) == 4
    keys = [(t.instance_id, t.repetition_id) for t, last in zip(out["trajectory_ids"], out["is_last_step"]) if last]
    assert keys == [("q1", 0), ("q1", 1), ("q2", 0), ("q2", 1)]
    assert out["rewards"] == [1.0] * 12
    bad = native_input()
    bad["trajectory_ids"] = [ID("q1", 0)] * 4
    with pytest.raises(ValueError, match="samples-per-prompt"):
        asyncio.run(generator.generate(bad))


def test_missing_logprobs_are_not_fabricated(task_yaml):
    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    runner.infer_engine.no_logprobs = True
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["rollout_logprobs"] is None


def test_truncation_masks_entire_trajectory(task_yaml):
    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    runner.infer_engine.truncated = True
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["traj_rewards"] == [0.0, 0.0]
    assert out["episode_nums"] == [1, 1]
    assert all(not any(mask) for mask in out["loss_masks"])


def test_runner_cleans_up_after_a_failed_batch(task_yaml):
    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    with pytest.raises(ValueError, match="no silent truncation"):
        asyncio.run(runner.run({"env_extras": [{"instance": instance("X" * 40)}]}))
    assert runner.trajectories == {} and runner.batch == []
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["traj_rewards"] == [1.0, 1.0]


def test_oversize_context_is_rejected_instead_of_silently_truncated():
    cfg = TrajectoryConfig(instance_id="q", trajectory_id=0, memory={"chunk_size": 1, "max_chunk_number": 2})
    agent = MemAgent(cfg, Backend(), Tokenizer())
    with pytest.raises(ValueError, match="no silent truncation"):
        asyncio.run(agent.run(LongContextMemoryTask.normalize_instance(instance())))
    assert agent.transitions == []


def test_data_conversion_preserves_all_reference_answers():
    records = convert_records([{"context": "Paris is in France", "input": "Country?", "answers": ["France", "French Republic"]}])
    assert records[0]["instance"]["answers"] == ["France", "French Republic"]
    assert records[0]["prompt"][0]["content"] == "Country?"
    with pytest.raises(ValueError):
        convert_records([{"context": [1, 2], "input": "Country?", "answers": ["France"]}])
    assert extract_boxed_content(r"prefix \boxed{a{b}}") == "a{b}"


def test_bad_backend_token_alignment_fails_loudly():
    class BadBackend(Backend):
        async def async_generate_ids(self, **kwargs):
            return "x", {"output_tokens": [1, 2], "logprobs": [0.0], "finish_reason": "stop"}

    cfg = TrajectoryConfig(instance_id="q", trajectory_id=0, max_prompt_length=6000)
    agent = MemAgent(cfg, BadBackend(), Tokenizer())
    with pytest.raises(ValueError, match="identical lengths"):
        asyncio.run(agent.run(LongContextMemoryTask.normalize_instance(instance(""))))


def test_all_runtime_imports_use_the_loom_namespace():
    package = ROOT / "loom"
    assert (package / "agents/memory/agent.py").is_file()
    forbidden = ("skyrl_agent", "loom.skyrl_agent_1", "loom.loom")
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden), str(path)
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(forbidden) for alias in node.names), str(path)


def test_root_package_and_yaml_registrations():
    import importlib
    import loom
    from loom.agents.mapping import AGENT_GENERATOR_REGISTRY, AGENT_TRAJECTORY_REGISTRY
    from loom.functional.chat_template import get_templates_path

    assert Path(loom.__file__).resolve() == ROOT / "loom/__init__.py"
    cfg = OmegaConf.load(ROOT / "examples/memory/memagent.yaml")
    assert cfg.agent_cls == "loom.agents.memory.MemAgent"
    for path in [cfg.agent_cls, cfg.task, AGENT_GENERATOR_REGISTRY[cfg.agent_cls], AGENT_TRAJECTORY_REGISTRY[cfg.agent_cls]]:
        module, name = path.rsplit(".", 1)
        assert getattr(importlib.import_module(module), name)
    assert (Path(get_templates_path()) / "qwen3_acc_thinking.jinja2").is_file()
