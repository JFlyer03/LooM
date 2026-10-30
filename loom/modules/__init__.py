"""Memory modules owned by loom, migrated from the original implementation."""

from .base import BaseModule
from .memory_agent import RecurrentMemoryModule, TextMemory
from .memory_extractor import NaiveExtractor
from .memory_updater import NaiveUpdater
from .memory_retriever import NaiveRetriever, RerankRetriever
from .placeholders import MemoryExtractor, MemoryUpdater, MemoryRetriever
from ..common.registry import MODULE_REGISTRY


def build_module(name, tokenizer=None, **kwargs):
    """Build a migrated module by its original registry name or a class path."""
    if "." in name:
        from ..integrations.base import _import_object
        module_cls = _import_object(name)
    else:
        module_cls = MODULE_REGISTRY.get(name)
    return module_cls(tokenizer=tokenizer, **kwargs)


__all__ = [
    "BaseModule", "RecurrentMemoryModule", "TextMemory", "NaiveExtractor",
    "NaiveUpdater", "NaiveRetriever", "RerankRetriever", "MemoryExtractor",
    "MemoryUpdater", "MemoryRetriever", "build_module",
]
