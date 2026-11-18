"""Question-conditioned text memory used by MemAgent."""


class RecurrentMemory:
    initial_state = "No previous memory"

    @staticmethod
    async def update(state, question, chunk, generate):
        prompt = f"""You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>\x20
{question}
</problem>

<memory>
{state}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""
        response = await generate(prompt, phase="memory_update")
        if "<think>" in response and "</think>" not in response:
            return response[-100:].strip()
        return response.rsplit("</think>", 1)[-1].strip()
