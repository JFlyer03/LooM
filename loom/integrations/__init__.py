"""Load optional backends only when they are requested."""

from importlib import import_module

__all__ = ["skyrl_train", "verl", "openai"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register the lightweight adapter used by build_backend("verl").
from . import verl
