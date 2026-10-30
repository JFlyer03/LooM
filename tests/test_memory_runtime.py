import ast
import asyncio
import os
import json
import subprocess
import sys
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
from loom.integrations.skyrl_train.memory_config import prepare_memory_config
from loom.memory.recurrent import ANSWER_PROMPT, UPDATE_PROMPT
from loom.modules import BaseModule, RecurrentMemoryModule
from loom.tasks.memory import LongContextMemoryTask, extract_boxed_content
from loom.tasks.prepare_memory_data import convert_records
from loom.tasks.verifiers.memagent import compute_score

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
        self.truncated_memory = False
        self.sampling_params = []

    async def async_generate_ids(self, input_ids, sampling_params, request_id):
        prompt = Tokenizer().decode(input_ids)
        self.calls.append((request_id, prompt))
        self.sampling_params.append(dict(sampling_params))
        if "<section>" in prompt:
            text = f"<think>analysis</think>memory-{request_id}"
        else:
            text = r"The answer is \boxed{France}"
        ids = Tokenizer().encode(text)
        await asyncio.sleep(0)
        return text, {
            "output_tokens": ids,
            "finish_reason": "length" if self.truncated or self.truncated_memory and "<section>" in prompt else "stop",
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
    cfg = OmegaConf.load(ROOT / "examples/MemAgent/memagent.yaml")
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
    assert out["episode_nums"] == [3, 3]
    assert all(not any(mask) for mask in out["loss_masks"])


def test_memory_at_its_token_limit_still_reads_remaining_chunks(task_yaml):
    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    runner.infer_engine.truncated_memory = True
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["traj_rewards"] == [1.0, 1.0]
    assert out["stop_reasons"] == ["length", "length", "stop"] * 2
    assert out["episode_nums"] == [3, 3]
    assert all(all(mask) for mask in out["loss_masks"])


def test_stage_limits_are_applied_without_resampling_groups(task_yaml):
    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    runner.cfg.memory.max_memorization_length = 32
    runner.cfg.memory.max_final_response_length = 48
    asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    for (_, prompt), params in zip(runner.infer_engine.calls, runner.infer_engine.sampling_params):
        assert params["max_tokens"] == (32 if "<section>" in prompt else 48)
        assert params["n"] == 1
    assert runner.cfg.generator.sampling_params.max_tokens == 1024


def test_configured_dispatcher_limits_concurrent_generation(task_yaml):
    class CountingBackend(Backend):
        active = 0
        peak = 0

        async def async_generate_ids(self, **kwargs):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                return await super().async_generate_ids(**kwargs)
            finally:
                self.active -= 1

    runner = AutoAgentRunner.from_task(str(task_yaml), None, Tokenizer())
    runner.infer_engine = CountingBackend()
    runner.cfg.dispatcher.max_parallel_agents = 1
    asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert runner.infer_engine.peak == 1
    assert runner.infer_engine.active == 0


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
    forbidden = ("loom_agent", "skyrl_agent", "loom.skyrl_agent_1", "loom.loom")
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
    cfg = OmegaConf.load(ROOT / "examples/MemAgent/memagent.yaml")
    assert cfg.agent_cls == "loom.agents.memory.MemAgent"
    module, name = cfg.memory.module_cls.rsplit(".", 1)
    memory_cls = getattr(importlib.import_module(module), name)
    assert memory_cls is RecurrentMemoryModule
    assert issubclass(memory_cls, BaseModule)
    for path in [cfg.agent_cls, cfg.task, AGENT_GENERATOR_REGISTRY[cfg.agent_cls], AGENT_TRAJECTORY_REGISTRY[cfg.agent_cls]]:
        module, name = path.rsplit(".", 1)
        assert getattr(importlib.import_module(module), name)
    assert (Path(get_templates_path()) / "qwen3_acc_thinking.jinja2").is_file()


@pytest.mark.parametrize(("prediction", "answers", "score"), [
    (r"Reasoning. \boxed{FRANCE}", ["France"], 1.0),
    (r"\boxed{The France}", ["France"], 0.0),
    (r"\boxed{France.}", ["France"], 0.0),
    (r"\boxed{ French Republic }", ["France", "French Republic"], 1.0),
    (r"\boxed{wrong} then \boxed{France}", ["France"], 1.0),
    (r"\boxed{France} then \boxed{wrong}", ["France"], 0.0),
    (r"\boxed France$", ["France"], 1.0),
    ("France", ["France"], 0.0),
    (r"\boxed{France", ["France"], 0.0),
    (r"\boxed{France}" + "x" * 300, ["France"], 0.0),
])
def test_memagent_uses_its_training_verifier(prediction, answers, score):
    assert compute_score(prediction, answers) == score
    result = asyncio.run(LongContextMemoryTask.evaluate_result(
        {"answer": prediction}, {"answers": answers}, {"reward_mode": "memagent"},
    ))
    assert result == score


@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize(("folder", "script"), [
    ("MemAgent", "run_memagent.sh"),
    ("RMM", "run_rmm.sh"),
    ("Memory-R1", "run_memory_r1.sh"),
])
def test_launch_script_forwards_overrides_and_uses_python_config_mapping(tmp_path, with_validation, folder, script):
    dataset = tmp_path / "data with spaces.parquet"
    dataset.touch()
    capture = tmp_path / "command.json"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
    )
    executable.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "CAPTURE": str(capture)}
    overrides = ["trainer.policy.model.path=/local/model with spaces",
                 f"data.train_data={json.dumps([str(dataset)])}", "trainer.policy.optimizer_config.lr=2e-6"]
    if with_validation:
        overrides += [f"data.val_data={json.dumps([str(dataset)])}", "trainer.eval_interval=5"]
    subprocess.run(
        ["bash", str(ROOT / "examples" / folder / script), *overrides],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    command = json.loads(capture.read_text())
    assert command["cwd"] == str(ROOT)
    assert command["args"][:2] == ["-m", "loom.integrations.skyrl_train.memory_main"]
    # The entrypoint resolves task-relative paths after the launcher's cd.
    args = command["args"][2:]
    task_path = ROOT / OmegaConf.from_dotlist(args).generator.task
    assert task_path.parent == ROOT / "examples" / folder
    args.append(f"generator.task={task_path}")
    cfg = OmegaConf.create(prepare_memory_config(args))
    task = OmegaConf.load(cfg.generator.task)
    assert cfg.trainer.policy.model.path == "/local/model with spaces"
    assert cfg.trainer.policy.optimizer_config.lr == 2e-6
    assert cfg.data.train_data == [str(dataset)]
    assert cfg.data.val_data == ([str(dataset)] if with_validation else [])
    assert cfg.trainer.eval_interval == (5 if with_validation else -1)
    assert cfg.generator.n_samples_per_prompt == task.generator.num_trajectories
    assert cfg.generator.eval_n_samples_per_prompt == task.generator.val_config.num_trajectories
    assert cfg.generator.step_wise_trajectories is True
    assert cfg.generator.merge_stepwise_output is False
    assert cfg.trainer.algorithm.grpo_norm_by_std == (folder != "MemAgent")
    assert cfg.trainer.algorithm.advantage_batch_normalize is False
    assert cfg.trainer.algorithm.loss_reduction == ("token_mean" if folder == "MemAgent" else "sequence_mean")
    assert cfg.trainer.algorithm.kl_loss_coef == (0.001 if folder == "MemAgent" else 0.01)
    assert cfg.trainer.algorithm.policy_loss_type == "regular"
    if folder == "MemAgent":
        assert task.memory.reward_mode == "memagent"
    else:
        assert task.generator.num_trajectories == 8
        assert task.memory.trainable_stages == (["rerank"] if folder == "RMM" else ["update"])


def test_python_entrypoint_maps_custom_task_sampling_and_limits(tmp_path):
    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent.yaml")
    task.generator.num_trajectories = 3
    task.generator.val_config.num_trajectories = 2
    task.generator.max_prompt_length = 4096
    task.memory.max_memorization_length = 128
    task.memory.max_final_response_length = 256
    task.generator.sampling_params.temperature = 0.8
    task.generator.val_config.sampling_params.top_p = 0.9
    path = tmp_path / "custom.yaml"
    OmegaConf.save(task, path)
    cfg = OmegaConf.create(prepare_memory_config([f"generator.task={path}", "trainer.epochs=5"]))
    assert cfg.generator.n_samples_per_prompt == 3
    assert cfg.generator.eval_n_samples_per_prompt == 2
    assert cfg.trainer.max_prompt_length == cfg.generator.max_input_length == 4096
    assert cfg.generator.sampling_params.max_generate_length == 256
    assert cfg.generator.eval_sampling_params.max_generate_length == 256
    assert cfg.generator.sampling_params.temperature == 0.8
    assert cfg.generator.eval_sampling_params.top_p == 0.9
    assert cfg.trainer.epochs == 5


@pytest.mark.parametrize("override", [
    "generator.n_samples_per_prompt=8",
    "generator.eval_n_samples_per_prompt=1",
    "generator.max_input_length=100",
])
def test_python_entrypoint_rejects_shared_settings_that_conflict_with_yaml(override):
    path = ROOT / "examples/MemAgent/memagent.yaml"
    with pytest.raises(ValueError, match="must match"):
        prepare_memory_config([f"generator.task={path}", override])


def test_python_entrypoint_rejects_invalid_sampling_counts(tmp_path):
    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent.yaml")
    task.generator.num_trajectories = 0
    path = tmp_path / "invalid.yaml"
    OmegaConf.save(task, path)
    with pytest.raises(ValueError, match="positive integers"):
        prepare_memory_config([f"generator.task={path}"])
