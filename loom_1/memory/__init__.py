"""Memory mechanisms independent of the training backend."""

from .recurrent import RecurrentMemoryModule, TextMemory

__all__ = ["RecurrentMemoryModule", "TextMemory"]
