"""The memory task package must remain usable by existing dataset and rollout code."""

import asyncio

from loom.tasks.long_context_memory import LongContextMemoryTask
from loom.tasks.long_context_memory.utils import LongContextMemoryTask as Implementation
from loom.tasks.prepare_memory_data import convert_records


def test_memory_task_package_preserves_data_preparation_and_scoring():
    assert LongContextMemoryTask is Implementation
    record = {
        "context": "Paris is the capital of France.",
        "extra_info": {"question": "What is the capital of France?"},
        "reward_model": {"ground_truth": "Paris"},
    }
    converted, = convert_records([record])
    instance = asyncio.run(LongContextMemoryTask.initialize_runtime(converted["instance"]))
    assert converted["prompt"] == LongContextMemoryTask.get_instruction(instance)
    assert instance["answers"] == ["Paris"]
    answer = LongContextMemoryTask.complete_runtime(r"\boxed{Paris}")
    assert asyncio.run(LongContextMemoryTask.evaluate_result(answer, instance)) == 1.0
