"""MemAgent orchestration copied from the original modular MemoryAgent."""

from ...common.registry import AGENT_REGISTRY
from ...modules import build_module
from .policy import MemoryPolicyAgent


@AGENT_REGISTRY.register("memagent")
class MemAgent(MemoryPolicyAgent):
    def __init__(self, traj_config, infer_engine, tokenizer):
        super().__init__(traj_config, infer_engine, tokenizer)
        self.module = build_module(
            self.options.get("module_cls", "recurrent_memory_module"), tokenizer,
            chunk_size=self.options.get("chunk_size", 2500),
            max_chunk_number=self.options.get("max_chunk_number", 6),
        )
        self.memory = self.module.memory
        self.memory_history = self.module.memory_history

    async def run(self, instance):
        self._check_new_trajectory()
        return await self.module.rollout(instance, self._generate, self.cfg.max_iterations)


MemoryAgent = MemAgent
