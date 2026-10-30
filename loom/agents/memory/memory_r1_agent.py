"""Extractor -> Updater composition migrated from loom_agent/agents."""

from ...common.registry import AGENT_REGISTRY
from ...modules import build_module
from .policy import MemoryPolicyAgent


@AGENT_REGISTRY.register("memory_r1_agent")
class MemoryR1Agent(MemoryPolicyAgent):
    def __init__(self, traj_config, infer_engine, tokenizer):
        super().__init__(traj_config, infer_engine, tokenizer)
        self.extractor = build_module(self.options.get("extractor", "naive_extractor"), tokenizer)
        self.updater = build_module(self.options.get("updater", "naive_updater"), tokenizer)

    async def run(self, instance):
        self._check_new_trajectory()
        if self.cfg.max_iterations < 2:
            raise ValueError("Memory-R1 requires extraction and update steps")
        extraction, ext_stop = await self.extractor.rollout(instance["fact"], self._generate)
        update, upd_stop = await self.updater.rollout(instance["context_memory"], extraction, self._generate)
        self.transitions[-1].episode_done = True
        finish = "completed" if ext_stop == upd_stop == "stop" else "CONTEXT_WINDOW_EXCEEDED"
        return finish, {"extraction": extraction, "update": update}
