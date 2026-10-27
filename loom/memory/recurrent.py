"""The fixed MemAgent workflow migrated from loom_agent/modules/memory_agent.py."""

from dataclasses import dataclass


UPDATE_PROMPT = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

ANSWER_PROMPT = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


@dataclass
class TextMemory:
    value: str = "No previous memory"

    def update(self, response: str):
        # Preserve the legacy method's thinking-text removal, including its fallback.
        if "<think>" in response:
            response = response.split("</think>")[-1].strip() if "</think>" in response else response[-100:].strip()
        self.value = response


class RecurrentMemoryModule:
    def __init__(self, tokenizer, chunk_size=2500, max_chunk_number=6):
        if chunk_size < 1 or max_chunk_number < 1:
            raise ValueError("chunk_size and max_chunk_number must be positive")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_chunk_number = max_chunk_number

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
