"""MemAgent's memory workflow copied from loom_agent/modules/memory_agent.py.

The original prompts, chunk order, memory replacement and thinking-text removal
are retained here. An async callback replaces local PyTorch generation, and the
training backend computes advantages from the final task reward.
"""

from dataclasses import dataclass

from .base import BaseModule
from ..common.registry import MODULE_REGISTRY
from ..common.utils import TEMPLATE as UPDATE_PROMPT, TEMPLATE_FINAL_BOXED as ANSWER_PROMPT


@dataclass
class TextMemory:
    value: str = "No previous memory"

    def update(self, response: str):
        # Preserve the legacy method's thinking-text removal, including its fallback.
        if "<think>" in response:
            response = response.split("</think>")[-1].strip() if "</think>" in response else response[-100:].strip()
        self.value = response


@MODULE_REGISTRY.register("recurrent_memory_module")
class RecurrentMemoryModule(BaseModule):
    def __init__(self, tokenizer, chunk_size=2500, max_chunk_number=6):
        super().__init__(tokenizer)
        if chunk_size < 1 or max_chunk_number < 1:
            raise ValueError("chunk_size and max_chunk_number must be positive")
        self.chunk_size = chunk_size
        self.max_chunk_number = max_chunk_number
        self.memory = TextMemory()
        self.memory_history = []

    def chunks(self, context):
        ids = self.tokenizer.encode(context, add_special_tokens=False)
        count = (len(ids) + self.chunk_size - 1) // self.chunk_size
        if count > self.max_chunk_number:
            raise ValueError(f"num_chunks({count}) > max_chunk_number({self.max_chunk_number}); no silent truncation")
        return [
            self.tokenizer.decode(ids[i:i + self.chunk_size], skip_special_tokens=True)
            for i in range(0, len(ids), self.chunk_size)
        ]

    def update_prompt(self, question, memory, chunk):
        return UPDATE_PROMPT.format(prompt=question, memory=memory.value, chunk=chunk)

    def answer_prompt(self, question, memory):
        return ANSWER_PROMPT.format(prompt=question, memory=memory.value)

    async def rollout(self, instance, generate, max_iterations):
        chunks = self.chunks(instance["context"])
        if len(chunks) + 1 > max_iterations:
            raise ValueError("max_iterations must allow all memory updates and the final answer")
        for chunk in chunks:
            response, _ = await generate(
                self.update_prompt(instance["question"], self.memory, chunk), "memory_update"
            )
            # A memory can fill its fixed token budget; the next chunk still follows.
            self.memory.update(response)
            self.memory_history.append(self.memory.value)
        response, reason = await generate(
            self.answer_prompt(instance["question"], self.memory), "answer"
        )
        finish_reason = "completed" if reason == "stop" else "CONTEXT_WINDOW_EXCEEDED"
        return finish_reason, {"answer": response, "memory": self.memory.value}
