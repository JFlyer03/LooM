"""An in-memory text entry bank with keyword recall."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .base import BaseMemory

MemoryBankState = dict[str, str]


@dataclass(frozen=True)
class MemoryEdit:
    action: Literal["add", "update", "delete"]
    entry_id: str
    text: str | None = None


class MemoryBank(BaseMemory[MemoryBankState]):
    def create_state(self) -> MemoryBankState:
        return {}

    def store(self, state: MemoryBankState, entries: Mapping[str, str]) -> MemoryBankState:
        """Insert new entries and return the next state."""
        return self.apply_update(state, (MemoryEdit("add", key, text) for key, text in entries.items()))

    def recall(self, state: MemoryBankState, query: str, *, top_k: int) -> MemoryBankState:
        """Return matching entries by descending keyword overlap, oldest first on ties."""
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        keywords = set(re.findall(r"\w+", query.casefold()))
        if top_k == 0 or not keywords:
            return {}

        matches = []
        for key, text in state.items():
            score = len(keywords.intersection(re.findall(r"\w+", text.casefold())))
            if score:
                matches.append((score, key))
        matches.sort(key=lambda match: match[0], reverse=True)
        return {key: state[key] for _, key in matches[:top_k]}

    def apply_update(self, state: MemoryBankState, plan: Iterable[MemoryEdit]) -> MemoryBankState:
        """Apply edits in order and return a new state after the whole plan succeeds."""
        updated = state.copy()
        for edit in plan:
            action, key, text = edit.action, edit.entry_id, edit.text
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Memory entry IDs must be nonempty strings")
            if action == "add":
                if key in updated:
                    raise ValueError(f"Memory entry already exists: {key!r}")
            elif action in ("update", "delete"):
                if key not in updated:
                    raise ValueError(f"Unknown memory entry: {key!r}")
            else:
                raise ValueError(f"Unsupported memory action: {action!r}")

            if action == "delete":
                if text is not None:
                    raise ValueError("Delete edits take an entry ID only")
                del updated[key]
            else:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Memory entry text must be a nonempty string")
                updated[key] = text
        return updated
