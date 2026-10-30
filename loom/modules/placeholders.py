"""Extension names retained from old LooM, with usable default implementations."""

from .memory_extractor import NaiveExtractor
from .memory_updater import NaiveUpdater
from .memory_retriever import NaiveRetriever


class MemoryExtractor(NaiveExtractor):
    pass


class MemoryUpdater(NaiveUpdater):
    pass


class MemoryRetriever(NaiveRetriever):
    pass
