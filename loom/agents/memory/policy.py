from copy import deepcopy
from uuid import uuid4

from ...functional.utils import Observation, TokensWithLogprobs, Transition
from ...modules.memory_agent import TextMemory


class MemoryPolicyAgent:
    """Record raw policy calls shared by the modular memory Agents.

    The runner owns sampling groups; the training backend owns advantages.
    """

    def __init__(self, traj_config, infer_engine, tokenizer):
        self.cfg = traj_config
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        options = traj_config.memory or {}
        self.options = options
        self.memory = TextMemory()
        self.transitions = []
        self.messages = []
        self.memory_history = []
        self.stage_limits = {
            "memory_update": options.get("max_memorization_length"),
            "answer": options.get("max_final_response_length"),
            "extraction": options.get("max_generate_length"),
            "update": options.get("max_generate_length"),
            "rerank": options.get("max_generate_length"),
        }
        if any(limit is not None and limit < 1 for limit in self.stage_limits.values()):
            raise ValueError("Memory and answer generation limits must be positive")
        self.agent_id = uuid4().hex

    async def _generate(self, prompt, stage):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.cfg.qwen3_enable_thinking,
        )
        if len(input_ids) > self.cfg.max_prompt_length:
            raise ValueError(f"{stage} input has {len(input_ids)} tokens; limit is {self.cfg.max_prompt_length}")
        sampling_params = deepcopy(self.cfg.sampling_params or {})
        if self.stage_limits.get(stage) is not None:
            sampling_params["max_tokens"] = min(
                sampling_params.get("max_tokens", self.stage_limits.get(stage)), self.stage_limits.get(stage)
            )
        sampling_params["n"] = 1
        response, meta = await self.infer_engine.async_generate_ids(
            input_ids=input_ids,
            sampling_params=sampling_params,
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
            episode_done=stage == "answer",
            metrics={
                "stage": stage, "memory_before": self.memory.value, "stop_reason": reason,
                "trainable": stage in self.options.get("trainable_stages", [stage]),
            },
        ))
        self.messages.extend(messages + [{"role": "assistant", "content": response}])
        return response, reason

    def _check_new_trajectory(self):
        if self.transitions:
            raise RuntimeError("Create a new memory Agent for each trajectory")
