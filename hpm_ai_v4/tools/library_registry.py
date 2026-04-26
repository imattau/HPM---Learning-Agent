"""Lightweight registry for curated HPM pattern libraries."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class LibraryEntry:
    """Metadata for one saved library bundle."""

    name: str
    path: str
    domain: str
    status: str = "seed"
    source: str = "unknown"
    density_mean: float = 0.0
    density_min: float = 0.0
    density_max: float = 0.0
    pattern_count: int = 0
    created_at: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LibraryEntry":
        return LibraryEntry(
            name=str(data.get("name", "")),
            path=str(data.get("path", "")),
            domain=str(data.get("domain", "text")),
            status=str(data.get("status", "seed")),
            source=str(data.get("source", "unknown")),
            density_mean=float(data.get("density_mean", 0.0)),
            density_min=float(data.get("density_min", 0.0)),
            density_max=float(data.get("density_max", 0.0)),
            pattern_count=int(data.get("pattern_count", 0)),
            created_at=str(data.get("created_at", "")),
            notes=str(data.get("notes", "")),
        )


class LibraryRegistry:
    """JSON registry of reusable HPM libraries."""

    def __init__(self, path: str):
        self.path = path
        self.entries: Dict[str, LibraryEntry] = {}
        self.load()

    def register(self, entry: LibraryEntry) -> LibraryEntry:
        self.entries[entry.name] = entry
        self.save()
        return entry

    def upsert(
        self,
        name: str,
        path: str,
        domain: str,
        status: str = "seed",
        source: str = "unknown",
        density_mean: float = 0.0,
        density_min: float = 0.0,
        density_max: float = 0.0,
        pattern_count: int = 0,
        created_at: str = "",
        notes: str = "",
    ) -> LibraryEntry:
        entry = LibraryEntry(
            name=name,
            path=path,
            domain=domain,
            status=status,
            source=source,
            density_mean=density_mean,
            density_min=density_min,
            density_max=density_max,
            pattern_count=pattern_count,
            created_at=created_at,
            notes=notes,
        )
        return self.register(entry)

    def promote(self, name: str, notes: str = "") -> LibraryEntry:
        entry = self.require(name)
        entry.status = "promoted"
        if notes:
            entry.notes = notes
        self.save()
        return entry

    def validate(self, name: str, notes: str = "") -> LibraryEntry:
        entry = self.require(name)
        entry.status = "validated"
        if notes:
            entry.notes = notes
        self.save()
        return entry

    def seed(self, name: str, notes: str = "") -> LibraryEntry:
        entry = self.require(name)
        entry.status = "seed"
        if notes:
            entry.notes = notes
        self.save()
        return entry

    def require(self, name: str) -> LibraryEntry:
        if name not in self.entries:
            raise KeyError(f"Unknown library entry: {name!r}")
        return self.entries[name]

    def list(self, domain: Optional[str] = None, status: Optional[str] = None) -> List[LibraryEntry]:
        entries = list(self.entries.values())
        if domain is not None:
            entries = [entry for entry in entries if entry.domain == domain]
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        return sorted(entries, key=lambda entry: (entry.domain, entry.status, entry.name))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                [entry.to_dict() for entry in self.list()],
                f,
                indent=2,
                sort_keys=True,
            )

    def load(self) -> None:
        self.entries = {}
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for raw in data if isinstance(data, list) else []:
            entry = LibraryEntry.from_dict(raw)
            if entry.name:
                self.entries[entry.name] = entry

    def describe(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self.list()]
