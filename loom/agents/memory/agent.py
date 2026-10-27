from copy import deepcopy
from uuid import uuid4

from ...functional.utils import Observation, TokensWithLogprobs, Transition
from ...memory import RecurrentMemoryModule, TextMemory


class MemAgent:
    """One independent sample: read chunks, overwrite memory, answer.

    The runner owns sampling groups; the training backend owns advantages.
    """

    def __init__(self, traj_config, infer_engine, tokenizer):
        self.cfg = traj_config
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        options = traj_config.memory or {}
        self.module = RecurrentMemoryModule(
            tokenizer, options.get("chunk_size", 2500), options.get("max_chunk_number", 6)
        )
        self.memory = TextMemory()
        self.transitions = []
        self.messages = []
        self.memory_history = []
        self.agent_id = uuid4().hex

    async def _generate(self, prompt, stage):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.cfg.qwen3_enable_thinking,
        )
        if len(input_ids) > self.cfg.max_prompt_length:
            raise ValueError(f"{stage} input has {len(input_ids)} tokens; limit is {self.cfg.max_prompt_length}")
        response, meta = await self.infer_engine.async_generate_ids(
            input_ids=input_ids,
            sampling_params=deepcopy(self.cfg.sampling_params),
            request_id=self.agent_id,
        )
        token_ids = meta.get("output_tokens")
        logprobs = meta.get("logprobs")
        if not isinstance(response, str) or not isinstance(token_ids, list) or not token_ids:
            raise ValueError("Backend must return response text and nonempty output_tokens")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError("output_tokens and logprobs must have identical lengths")
        reason = meta.get("finish_reason")
        if reason not in ("stop", "length"):
            raise ValueError(f"Unexpected generation finish_reason: {reason!r}")
        self.transitions.append(Transition(
            ob=Observation(input_ids=list(input_ids)),
            ac=TokensWithLogprobs(token_ids=list(token_ids), logprobs=logprobs, text=response),
            reward=0.0,
            episode_done=stage == "answer" or reason == "length",
            metrics={"stage": stage, "memory_before": self.memory.value, "stop_reason": reason},
        ))
        self.messages.extend(messages + [{"role": "assistant", "content": response}])
        return response, reason

    async def run(self, instance):
        if self.transitions:
            raise RuntimeError("Create a new MemAgent for each trajectory")
        chunks = self.module.chunks(instance["context"])
        if len(chunks) + 1 > self.cfg.max_iterations:
            raise ValueError("max_iterations must allow all memory updates and the final answer")
        for chunk in chunks:
            response, reason = await self._generate(
                self.module.update_prompt(instance["question"], self.memory, chunk), "memory_update"
            )
            if reason == "length":
                return "CONTEXT_WINDOW_EXCEEDED", {"answer": "", "memory": self.memory.value}
            self.memory.update(response)
            self.memory_history.append(self.memory.value)
        response, reason = await self._generate(
            self.module.answer_prompt(instance["question"], self.memory), "answer"
        )
        finish_reason = "completed" if reason == "stop" else "CONTEXT_WINDOW_EXCEEDED"
        return finish_reason, {"answer": response, "memory": self.memory.value}
