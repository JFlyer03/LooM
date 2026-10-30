"""Run reset-context MemAgent calls inside VeRL's native V0 weight-sync context."""

import asyncio

from vllm import SamplingParams
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from loom import AutoAgentRunner
from .rollout_data import pack_rollout


class V0MemoryServer:
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    async def generate(self, request_id, prompt_ids, sampling_params):
        # V0's synchronous engine is called serially on the native worker thread.
        outputs = self.inference_engine.generate(
            prompts=[{"prompt_token_ids": prompt_ids}],
            sampling_params=SamplingParams(**sampling_params), use_tqdm=False,
        )
        result = outputs[0].outputs[0]
        return result.text, {
            "output_tokens": list(result.token_ids), "finish_reason": result.finish_reason, "logprobs": None,
        }


class MemoryV0Rollout:
    def __init__(self, native_rollout, task_yaml, tokenizer, rollout_config):
        self.native_rollout = native_rollout
        self.inference_engine = native_rollout.inference_engine
        self.tokenizer = tokenizer
        self.config = rollout_config
        self.runner = AutoAgentRunner.from_task(task_yaml, V0MemoryServer(self.inference_engine), tokenizer)

    def generate_sequences(self, prompts):
        val_mode = prompts.meta_info.get("val_mode", False)
        rows = asyncio.run(self.runner.run(prompts, val_mode=val_mode))
        return pack_rollout(rows, self.tokenizer, self.config, val_mode)


class MemoryV0Worker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        native_rollout, sharding_manager = super()._build_rollout(trust_remote_code=trust_remote_code)
        rollout = MemoryV0Rollout(
            native_rollout, self.config.rollout.memory_task_yaml, self.tokenizer, self.config.rollout,
        )
        # Keep VeRL's parameter synchronization, offloading and sleep/wake implementation intact.
        return rollout, sharding_manager
