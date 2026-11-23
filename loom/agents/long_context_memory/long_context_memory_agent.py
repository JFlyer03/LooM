"""Long-context question answering with chunk-wise recurrent memory."""

from uuid import uuid4

from loguru import logger

from loom.functional.history import MessageEncoder
from loom.functional.utils import record_transition
from loom.memory import RecurrentMemory


ANSWER_PROMPT = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>\x20
{question}
</problem>

<memory>
{state}
</memory>

Your answer:
"""


class MemAgent:
    def __init__(self, traj_config, infer_engine, tokenizer):
        self.cfg = traj_config
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        self.encoder = MessageEncoder(
            tokenizer, traj_config.qwen3_enable_thinking, traj_config.qwen3_acc_thinking
        )
        self.memory = RecurrentMemory()
        self.chunk_size = traj_config.agent_kwargs.get("chunk_size", 2048)
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("agent_kwargs.chunk_size must be a positive integer")
        self.transitions = []
        self.messages = []

    @record_transition
    async def _generate_with_recording(self, *, input_ids):
        text, metadata = await self.infer_engine.async_generate_ids(
            input_ids=input_ids,
            sampling_params=dict(self.cfg.sampling_params),
            request_id=uuid4().hex,
        )
        if metadata.get("output_tokens") is None or len(metadata["output_tokens"]) == 0:
            raise ValueError("MemAgent requires generated token IDs from its inference backend")
        return text, metadata

    async def _generate(self, prompt: str, *, phase: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.encoder.encode_messages(messages, [], is_first_message=True)
        if len(input_ids) > self.cfg.max_prompt_length:
            raise ValueError(f"MemAgent {phase} prompt exceeds max_prompt_length")
        response, _ = await self._generate_with_recording(input_ids=input_ids)
        self.transitions[-1].metrics.update(
            phase=phase, policy_id="actor", reset_context=True,
        )
        self.messages.extend([*messages, {"role": "assistant", "content": response}])
        return response

    async def run(self, instruction, context: str, state: str) -> str:
        """Run a rollout and record the actual generated tokens."""
        question = instruction[-1]["content"]
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        token_budget = self.chunk_size * self.cfg.max_iterations
        if token_budget < 0:
            raise ValueError("MemAgent max_iterations must be nonnegative")
        if len(context_ids) > token_budget:
            logger.warning(
                "MemAgent center-truncating context from {} to {} tokens before chunking",
                len(context_ids),
                token_budget,
            )
            head_tokens = (token_budget + 1) // 2
            tail_start = len(context_ids) - (token_budget - head_tokens)
            context_ids = context_ids[:head_tokens] + context_ids[tail_start:]
        for start in range(0, len(context_ids), self.chunk_size):
            chunk = self.tokenizer.decode(
                context_ids[start:start + self.chunk_size], skip_special_tokens=True,
            )
            state = await self.memory.update(state, question, chunk, self._generate)

        answer = await self._generate(ANSWER_PROMPT.format(question=question, state=state), phase="answer")
        self.transitions[-1].episode_done = True
        return answer

    async def infer(self, instruction, context: str) -> str:
        """Answer with a text backend and a fresh memory state."""
        question = instruction[-1]["content"]
        state = self.memory.create_state()
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        token_budget = self.chunk_size * self.cfg.max_iterations
        if token_budget < 0:
            raise ValueError("MemAgent max_iterations must be nonnegative")
        if len(context_ids) > token_budget:
            logger.warning(
                "MemAgent center-truncating context from {} to {} tokens before chunking",
                len(context_ids),
                token_budget,
            )
            head_tokens = (token_budget + 1) // 2
            tail_start = len(context_ids) - (token_budget - head_tokens)
            context_ids = context_ids[:head_tokens] + context_ids[tail_start:]

        async def generate(prompt: str, *, phase: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.encoder.encode_messages(messages, [], is_first_message=True)
            if len(input_ids) > self.cfg.max_prompt_length:
                raise ValueError(f"MemAgent {phase} prompt exceeds max_prompt_length")
            response = await self.infer_engine.async_generate_prompts(
                prompts=prompt, sampling_params=dict(self.cfg.sampling_params),
            )
            self.messages.extend([*messages, {"role": "assistant", "content": response}])
            return response

        for start in range(0, len(context_ids), self.chunk_size):
            chunk = self.tokenizer.decode(
                context_ids[start:start + self.chunk_size], skip_special_tokens=True,
            )
            state = await self.memory.update(state, question, chunk, generate)

        return await generate(ANSWER_PROMPT.format(question=question, state=state), phase="answer")

