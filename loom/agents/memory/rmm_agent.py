"""Fixed extractor/updater -> policy reranker, migrated from old LooM."""

import asyncio

from ...common.registry import AGENT_REGISTRY
from ...common.utils import LLMClient
from ...envs.memory_bank_utils import get_memory_store
from ...modules import build_module
from .policy import MemoryPolicyAgent


@AGENT_REGISTRY.register("memory_rmm_agent")
class MemoryRMMAgent(MemoryPolicyAgent):
    def __init__(self, traj_config, infer_engine, tokenizer):
        super().__init__(traj_config, infer_engine, tokenizer)
        self.extractor = build_module(self.options.get("extractor", "naive_extractor"), tokenizer)
        self.updater = build_module(self.options.get("updater", "naive_updater"), tokenizer)
        self.retriever = build_module(self.options.get("retriever", "rerank_retriever"), tokenizer)
        self.llm_client = LLMClient()
        self.store = get_memory_store()

    async def prepare_shared(self, instance):
        """Old RMM prepares one fixed candidate set before sampling G reranks."""
        extraction = await asyncio.to_thread(
            self.llm_client.chat, "You are a memory extraction expert.",
            self.extractor.build_prompt(instance["fact"]),
        )
        update = await asyncio.to_thread(
            self.llm_client.chat, "You are a smart memory manager.",
            self.updater.build_prompt(instance["context_memory"], extraction),
        )
        candidates = self.retriever.prepare_candidates(instance, extraction, update, self.store)
        return {"extraction": extraction, "update": update, "candidates": candidates}

    async def run(self, instance, prepared=None):
        self._check_new_trajectory()
        if prepared is None:
            prepared = await self.prepare_shared(instance)
        candidates = prepared["candidates"]
        response, reason = await self.retriever.rollout(instance["query"], candidates, self._generate)
        self.transitions[-1].episode_done = True
        finish = "completed" if reason == "stop" else "CONTEXT_WINDOW_EXCEEDED"
        return finish, {**prepared, "rerank": response}
