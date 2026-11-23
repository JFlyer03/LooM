"""State initialization and generation contracts for memory mechanisms."""

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT")


class GenerateMemoryText(Protocol):
    async def __call__(self, prompt: str, *, phase: str) -> str: ...


class BaseMemory(ABC, Generic[StateT]):
    @abstractmethod
    def create_state(self) -> StateT:
        """Create the initial memory state for one execution."""
        ...
