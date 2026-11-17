"""Store retrieval and learned reranking migrated from old LooM."""

import re

from ..common.registry import MODULE_REGISTRY
from ..common.utils import parse_json_from_text
from ..envs.memory_bank_utils import MemoryItem
from .base import BaseModule
from .memory_updater import NaiveUpdater

RERANK_PROMPT = """You are an expert memory retriever.
Your task is to select the most relevant memories to answer the user's query.

User Query: {query}

Candidate Memories:
{candidates}

Select the exact IDs of the most useful memories (max 8 items). Provide your reasoning in <think> tags, then output a JSON list of the selected IDs.
Example output:
<think> memory 1 is relevant because... </think>
[1, 3, 4]
"""


@MODULE_REGISTRY.register("naive_retriever")
class NaiveRetriever(BaseModule):
    def retrieve(self, query, store, top_k=3):
        return [memory for memory, _ in store.search_similar(query, top_k=top_k)]

    async def rollout(self, query, store, top_k=3, **kwargs):
        return self.retrieve(query, store, top_k)


@MODULE_REGISTRY.register("rerank_retriever")
class RerankRetriever(NaiveRetriever):
    def get_id_map(self, context_memory, extraction_output):
        context_memory_objs = [ MemoryItem.from_dict(mem) for mem in context_memory ]
        id_counter = 1
        id_map = {}

        for mem in context_memory_objs:
            temp_id = id_counter
            id_counter += 1
            id_map[temp_id] = ("context", mem)

        ext_json = parse_json_from_text(extraction_output)
        if isinstance(ext_json, dict) and "memory_list" in ext_json and isinstance(ext_json["memory_list"], list):
            for item in ext_json["memory_list"]:
                if not isinstance(item, dict): continue
                temp_id = id_counter
                id_counter += 1
                id_map[temp_id] = ("candidate", item)

        return id_map

    def prepare_candidates(self, instance, extraction_output, update_output, store):
        store.from_list(instance["memory"])
        NaiveUpdater(self.tokenizer).apply_update_plan(
            store, instance["context_memory"], extraction_output, update_output,
        )
        return self.retrieve(instance["query"], store, top_k=30)

    def build_prompt(self, query, candidates):
        text = "\n".join(f"[ID: {i + 1}] {memory.key}: {memory.value}" for i, memory in enumerate(candidates))
        return RERANK_PROMPT.format(query=query, candidates=text)

    def selected_memories(self, response, candidates):
        match = re.search(r"\[(.*?)\]", response.split("</think>")[-1])
        ids = [int(value.strip()) for value in match.group(1).split(",") if value.strip().isdigit()] if match else []
        return [memory for index, memory in enumerate(candidates, 1) if index in ids]

    async def rollout(self, query, candidates, generate, **kwargs):
        return await generate(self.build_prompt(query, candidates), "rerank")
