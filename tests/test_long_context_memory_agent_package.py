"""Long-context memory agents must resolve through their public registered paths."""

import importlib
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from loom.agents.mapping import AGENT_GENERATOR_REGISTRY, AGENT_TRAJECTORY_REGISTRY
from loom.agents.long_context_memory import MemAgent, LongContextMemoryTrajectory
from loom.agents.long_context_memory.long_context_memory_agent import MemAgent as Implementation
from loom.agents.long_context_memory.long_context_memory_runner import LongContextMemoryTrajectory as TrajectoryImplementation


def test_long_context_memory_exports_and_registered_paths():
    assert MemAgent is Implementation
    assert LongContextMemoryTrajectory is TrajectoryImplementation
    assert AGENT_GENERATOR_REGISTRY["loom.agents.long_context_memory.MemAgent"] == "loom.agents.base.AgentRunner"
    trajectory_path = AGENT_TRAJECTORY_REGISTRY["loom.agents.long_context_memory.MemAgent"]
    module, name = trajectory_path.rsplit(".", 1)
    assert getattr(importlib.import_module(module), name) is LongContextMemoryTrajectory


def test_trajectory_initializes_runtime_and_agent_when_generating():
    instance = {"context": "long article"}
    instruction = [{"role": "user", "content": "question"}]
    agent = SimpleNamespace(
        memory=SimpleNamespace(initial_state="No previous memory"),
        run=AsyncMock(return_value="answer"),
        messages=[],
        transitions=[SimpleNamespace(metrics={"finish_reason": "stop"})],
    )
    trajectory = LongContextMemoryTrajectory.__new__(LongContextMemoryTrajectory)
    trajectory.data = {"instance": instance}
    trajectory.cfg = SimpleNamespace(instance_id="sample", trajectory_id=0)
    trajectory.infer_engine = object()
    trajectory.tokenizer = object()
    trajectory.agent_cls = Mock(return_value=agent)
    trajectory.task = SimpleNamespace(
        initialize_runtime=AsyncMock(return_value=instance),
        get_instruction=Mock(return_value=instruction),
        complete_runtime=Mock(return_value="answer"),
    )

    async def run():
        await trajectory.initialize_trajectory()
        trajectory.task.initialize_runtime.assert_not_awaited()
        trajectory.agent_cls.assert_not_called()
        await trajectory.generate_trajectory()

    asyncio.run(run())
    trajectory.task.initialize_runtime.assert_awaited_once_with(instance)
    trajectory.agent_cls.assert_called_once_with(
        traj_config=trajectory.cfg,
        infer_engine=trajectory.infer_engine,
        tokenizer=trajectory.tokenizer,
    )
    agent.run.assert_awaited_once_with(instruction, "long article", "No previous memory")
    assert trajectory.result["results"] == "answer"
