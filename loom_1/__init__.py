"""loom's Agent runtime, derived from SkyRL-Agent."""

__all__ = ["AutoAgentRunner"]


def __getattr__(name):
    if name == "AutoAgentRunner":
        from .auto import AutoAgentRunner

        return AutoAgentRunner
    raise AttributeError(name)
