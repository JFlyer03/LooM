import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from loom import AutoAgentRunner
from loom.agents.memory import MemAgent, MemoryR1Agent, MemoryRMMAgent
from loom.common.registry import AGENT_REGISTRY, ENV_REGISTRY, MODULE_REGISTRY
from loom.envs.memory_bank_utils import MemoryItem, get_memory_store
from loom.integrations.base import BackendSpec, register_backend
from loom.integrations.skyrl_train.memory_generator import MemoryGenerator
from loom.modules import NaiveExtractor, NaiveUpdater, RerankRetriever, build_module
from loom.modules.memory_extractor import EXTRACTION_PROMPT_EN
from loom.modules.memory_updater import UPDATE_MEMORY_PROMPT
from loom.modules.memory_retriever import RERANK_PROMPT
from loom.tasks.prepare_memory_data import convert_records
from test_memory_runtime import Tokenizer, Input, Output, ID

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = json.dumps({"memory_list": [{"key": "country", "value": "France", "memory_type": "LongTermMemory", "tags": ["travel"]}]})
UPDATE = json.dumps({"operations": [{"id": 1, "op": "DEL"}, {"id": 2, "op": "ADD"}]})


def memory():
    return {"id": "db-memory-abc", "key": "country", "value": "Spain", "memory_type": "UserMemory", "tags": ["old"]}


def instance():
    return {"M": [memory()], "f": [{"role": "user", "content": "I moved to France.", "timestamp": "2026-01-01"}],
            "q": "Where do I live?", "a": "France", "context_memory": [memory()]}


class Backend:
    def __init__(self, **kwargs):
        self.calls = []

    async def async_generate_ids(self, input_ids, **kwargs):
        prompt = Tokenizer().decode(input_ids)
        self.calls.append(prompt)
        response = EXTRACTION if "memory extraction expert" in prompt else UPDATE if "smart memory manager" in prompt else "<think>relevant</think>[1]"
        tokens = Tokenizer().encode(response)
        await asyncio.sleep(0)
        return response, {"output_tokens": tokens, "logprobs": [-0.5] * len(tokens), "finish_reason": "stop"}


class FixedLLM:
    calls = []

    def chat(self, system, prompt):
        self.calls.append(system)
        if system == "You are a memory extraction expert.":
            return EXTRACTION
        if system == "You are a smart memory manager.":
            return UPDATE
        if system == "You are an impartial judge.":
            return "True"
        return json.dumps({"answer": "France", "cited_ids": [1]})


def task_yaml(tmp_path, algorithm, trainable_stages=None):
    register_backend("modules-test", BackendSpec(Backend, Output, Input))
    cfg = OmegaConf.load(ROOT / "examples/MemAgent/memagent.yaml")
    cfg.agent_cls = f"loom.agents.memory.{algorithm}"
    cfg.task = "loom.tasks.memory_bank." + ("MemoryBankTask" if algorithm == "MemoryR1Agent" else "RerankBankTask")
    cfg.generator.infer_backend = "modules-test"
    cfg.generator.num_trajectories = 2
    if trainable_stages is not None:
        cfg.memory.trainable_stages = trainable_stages
    path = tmp_path / "bank.yaml"
    OmegaConf.save(cfg, path)
    return path


def test_all_original_module_and_agent_names_are_registered():
    for name in ["naive_extractor", "naive_updater", "naive_retriever", "rerank_retriever", "recurrent_memory_module"]:
        assert type(build_module(name, Tokenizer())) is MODULE_REGISTRY.get(name)
    assert AGENT_REGISTRY.get("memagent") is MemAgent
    assert AGENT_REGISTRY.get("memory_r1_agent") is MemoryR1Agent
    assert AGENT_REGISTRY.get("memory_rmm_agent") is MemoryRMMAgent
    assert ENV_REGISTRY.get("memory_bank").__module__.startswith("loom.")
    assert ENV_REGISTRY.get("rerank_bank").__module__.startswith("loom.")


def test_copied_prompts_match_all_old_memory_modules():
    for filename, name, expected in [
        ("memory_extractor.py", "EXTRACTION_PROMPT_EN", EXTRACTION_PROMPT_EN),
        ("memory_updater.py", "UPDATE_MEMORY_PROMPT", UPDATE_MEMORY_PROMPT),
        ("memory_retriever.py", "RERANK_PROMPT", RERANK_PROMPT),
    ]:
        tree = ast.parse((ROOT / "loom_agent/modules" / filename).read_text())
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
        assert values[name] == expected


def test_numbered_ids_and_update_operations_preserve_memory_metadata():
    updater = NaiveUpdater(Tokenizer())
    existing, candidates, id_map = updater.prepare_memory_lists([memory()], EXTRACTION)
    assert existing[0]["id"] == 1 and candidates[0]["id"] == 2
    assert id_map[1][1].id == "db-memory-abc"
    assert id_map[2][1]["memory_type"] == "LongTermMemory"
    store = get_memory_store()
    store.from_list([memory()])
    updater.apply_update_plan(store, [memory()], EXTRACTION, UPDATE)
    saved, = store.get_all()
    assert saved.id != "2" and isinstance(saved.id, str)
    assert saved.value == "France"
    assert saved.memory_type == "LongTermMemory" and saved.tags == ["travel"]
    assert RerankRetriever(Tokenizer()).selected_memories("<think>choose</think>[1]", [saved]) == [saved]


def test_invalid_update_is_rejected_before_any_writes():
    store = get_memory_store()
    store.from_list([memory()])
    before = store.to_list()
    plan = json.dumps({"operations": [{"id": 1, "op": "DEL"}, {"id": 99, "op": "ADD"}]})
    with pytest.raises(ValueError, match="Unknown memory"):
        NaiveUpdater(Tokenizer()).apply_update_plan(store, [memory()], EXTRACTION, plan)
    assert store.to_list() == before


def test_stores_are_isolated_and_deserialization_does_not_mutate_source():
    source = memory()
    source["embedding"] = "<vector dim=1024>"
    first, second = get_memory_store(), get_memory_store()
    assert first is not second
    first.from_list([source])
    second.from_list([source])
    first.get(source["id"]).tags.append("changed")
    first.delete(source["id"])
    assert second.get(source["id"]).tags == ["old"]
    assert source["embedding"] == "<vector dim=1024>" and source["tags"] == ["old"]


def test_memory_r1_composes_extraction_and_update_with_separate_rewards(tmp_path):
    runner = AutoAgentRunner.from_task(str(task_yaml(tmp_path, "MemoryR1Agent")), None, Tokenizer())
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["episode_nums"] == [2, 2]
    assert [row["stage"] for row in out["env_metrics"]] == ["extraction", "update"] * 2
    assert out["rewards"] == [0.5] * 4
    assert all(all(mask) for mask in out["loss_masks"])


def test_memory_r1_native_adapter_requires_one_reward_objective(tmp_path):
    cfg = SimpleNamespace(step_wise_trajectories=True, task=str(task_yaml(tmp_path, "MemoryR1Agent")))
    with pytest.raises(ValueError, match="one trainable_stages"):
        MemoryGenerator(cfg, None, Tokenizer())
    cfg.task = str(task_yaml(tmp_path, "MemoryR1Agent", ["extraction"]))
    generator = MemoryGenerator(cfg, None, Tokenizer())
    out = asyncio.run(generator.runner.run({"env_extras": [{"instance": instance()}]}))
    assert [any(mask) for mask in out["loss_masks"]] == [True, False] * 2


def test_selected_extraction_reward_is_preserved_when_update_reward_differs(tmp_path):
    class InvalidUpdateBackend(Backend):
        async def async_generate_ids(self, input_ids, **kwargs):
            response, meta = await super().async_generate_ids(input_ids, **kwargs)
            if "smart memory manager" in Tokenizer().decode(input_ids):
                response = '{"operations": [{"id": 999, "op": "DEL"}]}'
                tokens = Tokenizer().encode(response)
                meta.update(output_tokens=tokens, logprobs=[-0.5] * len(tokens))
            return response, meta

    path = task_yaml(tmp_path, "MemoryR1Agent", ["extraction"])
    runner = AutoAgentRunner.from_task(str(path), None, Tokenizer())
    runner.infer_engine = InvalidUpdateBackend()
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["rewards"] == [0.5] * 4
    assert [row["stage_reward"] for row in out["env_metrics"]] == [0.5, 0.0] * 2


def test_rmm_uses_fixed_extraction_update_and_trains_only_reranking(tmp_path, monkeypatch):
    FixedLLM.calls = []
    monkeypatch.setattr("loom.agents.memory.rmm_agent.LLMClient", FixedLLM)
    monkeypatch.setattr("loom.envs.memory_bank.LLMClient", FixedLLM)
    runner = AutoAgentRunner.from_task(str(task_yaml(tmp_path, "MemoryRMMAgent")), None, Tokenizer())
    out = asyncio.run(runner.run({"env_extras": [{"instance": instance()}]}))
    assert out["episode_nums"] == [1, 1]
    assert [row["stage"] for row in out["env_metrics"]] == ["rerank", "rerank"]
    assert out["traj_rewards"] == [1.625, 1.625]
    assert len(runner.infer_engine.calls) == 2
    assert runner.infer_engine.calls[0] == runner.infer_engine.calls[1]
    assert FixedLLM.calls.count("You are a memory extraction expert.") == 1
    assert FixedLLM.calls.count("You are a smart memory manager.") == 1


def test_memory_bank_data_aliases_convert_without_old_framework_imports():
    out = convert_records([instance()], task="memory_bank")
    assert out[0]["instance"]["query"] == "Where do I live?"
    assert out[0]["instance"]["memory"][0]["id"] == "db-memory-abc"
    assert out[0]["data_source"] == "loom_memory_bank"


@pytest.mark.parametrize(("folder", "filename", "stages", "score"), [
    ("Memory-R1", "memory_r1.yaml", ["extraction", "update"], 0.5),
    ("RMM", "rmm.yaml", ["rerank"], 1.625),
])
@pytest.mark.parametrize("phase", ["train", "eval"])
def test_bank_examples_generate_native_training_and_eval_rows(tmp_path, monkeypatch, folder, filename, stages, score, phase):
    register_backend("modules-test", BackendSpec(Backend, Output, Input))
    monkeypatch.setattr("loom.agents.memory.rmm_agent.LLMClient", FixedLLM)
    monkeypatch.setattr("loom.envs.memory_bank.LLMClient", FixedLLM)
    FixedLLM.calls = []
    cfg = OmegaConf.load(ROOT / "examples" / folder / filename)
    cfg.generator.infer_backend = "modules-test"
    path = tmp_path / filename
    OmegaConf.save(cfg, path)
    generator = MemoryGenerator(SimpleNamespace(step_wise_trajectories=True, task=str(path)), None, Tokenizer())
    sampling = cfg.generator.val_config if phase == "eval" else cfg.generator
    count = sampling.num_trajectories
    input_batch = {
        "trajectory_ids": [ID("bank-question", repetition) for repetition in range(count)],
        "env_extras": [{"instance": instance()} for _ in range(count)],
        "batch_metadata": SimpleNamespace(training_phase=phase),
        "sampling_params": OmegaConf.to_container(sampling.sampling_params, resolve=True),
    }
    out = asyncio.run(generator.generate(input_batch))
    assert len(out["response_ids"]) == count * len(stages)
    assert [row["stage"] for row in out["env_metrics"]] == stages * count
    assert sum(out["is_last_step"]) == count
    assert out["rewards"] == [score] * (count * len(stages))
    assert [any(mask) for mask in out["loss_masks"]] == ([False, True] if folder == "Memory-R1" else [True]) * count
    assert len(out["trajectory_ids"]) == len(out["response_ids"])
    if folder == "RMM":
        assert FixedLLM.calls.count("You are a memory extraction expert.") == 1
        assert FixedLLM.calls.count("You are a smart memory manager.") == 1
