"""Adapt the migrated memory-bank environments to the new Task interface."""

import asyncio
from copy import deepcopy

from .base import BaseTask
from ..envs.memory_bank import MemoryBankEnv, RerankBankEnv


class MemoryBankTask(BaseTask):
    environment_cls = MemoryBankEnv

    @staticmethod
    def normalize_instance(instance):
        result = {
            "memory": instance.get("memory", instance.get("M", [])),
            "fact": instance.get("fact", instance.get("f", [])),
            "query": instance.get("query", instance.get("q", "")),
            "answer": instance.get("answer", instance.get("a", "")),
            "context_memory": instance.get("context_memory", []),
        }
        if not all(isinstance(result[key], list) for key in ("memory", "fact", "context_memory")):
            raise ValueError("Memory-bank memory, fact and context_memory must be lists")
        if not isinstance(result["query"], str) or not isinstance(result["answer"], str):
            raise ValueError("Memory-bank query and answer must be strings")
        return deepcopy(result)

    @classmethod
    async def initialize_runtime(cls, instance):
        return cls.normalize_instance(instance)

    @classmethod
    def get_instruction(cls, instance):
        return [{"role": "user", "content": cls.normalize_instance(instance)["query"]}]

    @classmethod
    def complete_runtime(cls, result):
        return result

    @classmethod
    async def evaluate_result(cls, result, instance, options=None):
        batch = {key: [value] for key, value in instance.items()}
        predictions = {key: [result[key]] for key in ("extraction", "update")}
        environment = cls.environment_cls(None, None)
        rewards = await asyncio.to_thread(environment.compute_reward, predictions, batch, 1)
        return {stage: float(values[0]) for stage, values in rewards.items()}


class RerankBankTask(MemoryBankTask):
    environment_cls = RerankBankEnv

    @classmethod
    async def evaluate_result(cls, result, instance, options=None):
        batch = {key: [value] for key, value in instance.items()}
        predictions = {"rerank": [result["rerank"]], "candidates": [result["candidates"]]}
        environment = cls.environment_cls(None, None)
        rewards = await asyncio.to_thread(environment.compute_reward, predictions, batch, 1)
        return float(rewards["rerank"][0])
