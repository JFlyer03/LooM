"""Trajectory execution and reward assignment for long-context memory agents."""

from loom.agents.base import BaseTrajectory
from .long_context_memory_agent import MemAgent


class LongContextMemoryTrajectory(BaseTrajectory):
    async def initialize_trajectory(self):
        pass

    async def generate_trajectory(self):
        self.instance = await self.task.initialize_runtime(self.data["instance"])
        self.agent: MemAgent = self.agent_cls(
            traj_config=self.cfg,
            infer_engine=self.infer_engine,
            tokenizer=self.tokenizer,
        )
        self.initial_memory = self.agent.memory.create_state()
        answer = await self.agent.run(
            self.task.get_instruction(self.instance), self.instance["context"], self.initial_memory,
        )
        self.result = {
            "instance_id": self.cfg.instance_id,
            "trajectory_id": self.cfg.trajectory_id,
            "messages": self.agent.messages,
            "transitions": self.agent.transitions,
            "results": self.task.complete_runtime(answer),
            "finish_reason": self.agent.transitions[-1].metrics["finish_reason"],
        }

    async def evaluate_trajectory(self):
        reward = await self.task.evaluate_result(self.result["results"], self.instance)
        self.result["reward"] = reward
        self.agent.transitions[-1].reward = reward
