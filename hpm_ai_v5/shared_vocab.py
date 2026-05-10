"""Dependency-light shared token vocabulary for v5 adapters and polygraphs."""

from __future__ import annotations


class UnifiedVocabulary:
    """Shared mapping for string tokens to numeric IDs."""

    _type_map: dict[str, float] = {}

    @classmethod
    def get_id(cls, name: str) -> float:
        if name not in cls._type_map:
            cls._type_map[name] = float(len(cls._type_map) + 1)
        return cls._type_map[name]
