"""Memory-module boundary adapted from loom_agent/modules/base.py.

Modules own memory operations and orchestration. Generation is supplied by the
Agent; sampling groups, advantages and training tensors belong to the backend.
"""

from abc import ABC, abstractmethod


class BaseModule(ABC):
    def __init__(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer

    @abstractmethod
    async def rollout(self, *args, **kwargs):
        """Run one sample using an async generate(prompt, stage) callback."""
        raise NotImplementedError

    async def inference(self, *args, **kwargs):
        return await self.rollout(*args, **kwargs)
