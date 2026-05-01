"""Lightweight registry and resolver for curated HPM pattern libraries."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class LibraryEntry:
    """Metadata for one saved library bundle."""

    name: str
    path: str
    domain: str
    status: str = "seed"
    bundle_kind: str = "auto"
    level_contract: str = ""
    obs_dims: List[int] = field(default_factory=list)
    decoder_families: List[str] = field(default_factory=list)
    source: str = "unknown"
    density_mean: float = 0.0
    density_min: float = 0.0
    density_max: float = 0.0
    pattern_count: int = 0
    created_at: str = ""
    notes: str = ""
    ingest_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LibraryEntry":
        return LibraryEntry(
            name=str(data.get("name", "")),
            path=str(data.get("path", "")),
            domain=str(data.get("domain", "text")),
            status=str(data.get("status", "seed")),
            bundle_kind=str(data.get("bundle_kind", "auto")),
            level_contract=str(data.get("level_contract", "")),
            obs_dims=[int(v) for v in data.get("obs_dims", []) or []],
            decoder_families=[str(v) for v in data.get("decoder_families", []) or []],
            source=str(data.get("source", "unknown")),
            density_mean=float(data.get("density_mean", 0.0)),
            density_min=float(data.get("density_min", 0.0)),
            density_max=float(data.get("density_max", 0.0)),
            pattern_count=int(data.get("pattern_count", 0)),
            created_at=str(data.get("created_at", "")),
            notes=str(data.get("notes", "")),
            ingest_state=dict(data.get("ingest_state", {}) or {}),
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
        bundle_kind: str = "auto",
        level_contract: str = "",
        obs_dims: Optional[Sequence[int]] = None,
        decoder_families: Optional[Sequence[str]] = None,
        source: str = "unknown",
        density_mean: float = 0.0,
        density_min: float = 0.0,
        density_max: float = 0.0,
        pattern_count: int = 0,
        created_at: str = "",
        notes: str = "",
        ingest_state: Optional[Dict[str, Any]] = None,
    ) -> LibraryEntry:
        entry = LibraryEntry(
            name=name,
            path=path,
            domain=domain,
            status=status,
            bundle_kind=bundle_kind,
            level_contract=level_contract,
            obs_dims=[int(v) for v in (obs_dims or [])],
            decoder_families=[str(v) for v in (decoder_families or [])],
            source=source,
            density_mean=density_mean,
            density_min=density_min,
            density_max=density_max,
            pattern_count=pattern_count,
            created_at=created_at,
            notes=notes,
            ingest_state=dict(ingest_state or {}),
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

    def ingest_gate(self, name: str, *, adapter: Any = None, lowercase: bool = True):
        from hpm_ai_v4.tools.ingest import TextIngestGate

        entry = self.require(name)
        return TextIngestGate.from_snapshot(entry.ingest_state, adapter=adapter, lowercase=lowercase)

    def resolve_bundle(
        self,
        view: Optional[str] = None,
        *,
        domain: Optional[str] = None,
        task_family: Optional[str] = None,
        bundle_kind: Optional[str] = None,
        level_contract: Optional[str] = None,
        obs_dims: Optional[Sequence[int]] = None,
        status: Optional[str] = None,
        name: Optional[str] = None,
        require_exists: bool = True,
    ) -> Optional["BundleResolution"]:
        return BundleResolver(self).resolve(
            view=view,
            domain=domain,
            task_family=task_family,
            bundle_kind=bundle_kind,
            level_contract=level_contract,
            obs_dims=obs_dims,
            status=status,
            name=name,
            require_exists=require_exists,
        )


@dataclass(frozen=True)
class BundleResolution:
    """Resolved bundle choice plus scoring rationale."""

    entry: LibraryEntry
    path: str
    score: float
    reasons: Tuple[str, ...]
    view: str = ""


class BundleResolver:
    """Choose a compatible bundle from a registry using view metadata."""

    VIEW_PRESETS: Dict[str, Dict[str, Any]] = {
        "chat": {
            "domains": ("chat", "text"),
            "bundle_kinds": ("stacked", "bundle", "flat"),
            "level_contract": "l1-l5",
        },
        "math_text": {
            "domains": ("math", "structured_text", "text"),
            "bundle_kinds": ("stacked", "bundle", "flat"),
            "level_contract": "l1-l5",
        },
        "code_dsl": {
            "domains": ("code", "text"),
            "bundle_kinds": ("stacked", "bundle", "flat"),
            "level_contract": "l1-l5",
        },
        "structured_text": {
            "domains": ("structured_text", "text"),
            "bundle_kinds": ("stacked", "bundle", "flat"),
            "level_contract": "l1-l5",
        },
        "text": {
            "domains": ("text",),
            "bundle_kinds": ("flat", "stacked", "bundle"),
            "level_contract": "",
        },
    }

    STATUS_RANK = {
        "seed": 1.0,
        "validated": 2.0,
        "promoted": 3.0,
    }

    def __init__(self, registry: LibraryRegistry):
        self.registry = registry

    @staticmethod
    def bundle_base_path(path: str) -> str:
        base = str(path)
        if base.endswith(".l1.pkl"):
            return base[:-7]
        if base.endswith(".pkl"):
            return base[:-4]
        return base

    @classmethod
    def infer_bundle_kind(cls, entry: LibraryEntry) -> str:
        if entry.bundle_kind and entry.bundle_kind != "auto":
            return entry.bundle_kind
        base = cls.bundle_base_path(entry.path)
        if os.path.exists(base + ".l1.pkl"):
            return "stacked"
        if os.path.exists(entry.path):
            return "flat"
        return "unknown"

    @classmethod
    def infer_level_contract(cls, entry: LibraryEntry) -> str:
        if entry.level_contract:
            return entry.level_contract
        kind = cls.infer_bundle_kind(entry)
        if kind == "stacked":
            return "l1-l5"
        if kind == "flat":
            return "l1"
        return ""

    @classmethod
    def resolved_path(cls, entry: LibraryEntry) -> str:
        base = cls.bundle_base_path(entry.path)
        if os.path.exists(base + ".l1.pkl"):
            return base
        if os.path.exists(entry.path):
            return entry.path
        return base

    @staticmethod
    def _normalized_obs_dims(values: Optional[Sequence[int]]) -> Tuple[int, ...]:
        if not values:
            return tuple()
        return tuple(int(v) for v in values if int(v) >= 0)

    @staticmethod
    def _status_rank(status: str) -> float:
        return BundleResolver.STATUS_RANK.get(status, 0.0)

    def resolve(
        self,
        view: Optional[str] = None,
        *,
        domain: Optional[str] = None,
        task_family: Optional[str] = None,
        bundle_kind: Optional[str] = None,
        level_contract: Optional[str] = None,
        obs_dims: Optional[Sequence[int]] = None,
        status: Optional[str] = None,
        name: Optional[str] = None,
        require_exists: bool = True,
    ) -> Optional[BundleResolution]:
        preset = dict(self.VIEW_PRESETS.get(view or "", {}))
        domains = tuple(preset.get("domains", ()))
        bundle_kinds = tuple(preset.get("bundle_kinds", ()))
        preset_level_contract = str(preset.get("level_contract", ""))

        if domain is None and task_family is not None:
            domain = task_family
        if not domains and domain:
            domains = (domain,)
        if bundle_kind is not None:
            bundle_kinds = (bundle_kind,)
        if level_contract is None:
            level_contract = preset_level_contract or None

        target_obs_dims = self._normalized_obs_dims(obs_dims)
        scored: List[BundleResolution] = []
        for entry in self.registry.list():
            if name and entry.name != name:
                continue
            if domains and entry.domain not in domains:
                continue
            if status and entry.status != status:
                continue
            resolved_path = self.resolved_path(entry)
            if require_exists and not self._path_exists(resolved_path, entry):
                continue
            score, reasons = self._score_entry(
                entry,
                view=view or "",
                bundle_kinds=bundle_kinds,
                level_contract=level_contract or "",
                obs_dims=target_obs_dims,
            )
            if score is None:
                continue
            scored.append(
                BundleResolution(
                    entry=entry,
                    path=resolved_path,
                    score=score,
                    reasons=tuple(reasons),
                    view=view or "",
                )
            )

        if not scored:
            return None
        scored.sort(key=lambda item: (item.score, self._status_rank(item.entry.status), item.entry.pattern_count), reverse=True)
        return scored[0]

    def _score_entry(
        self,
        entry: LibraryEntry,
        *,
        view: str,
        bundle_kinds: Sequence[str],
        level_contract: str,
        obs_dims: Tuple[int, ...],
    ) -> Tuple[Optional[float], List[str]]:
        score = 0.0
        reasons: List[str] = []
        status_rank = self._status_rank(entry.status)
        score += 0.6 * status_rank
        if status_rank:
            reasons.append(f"status={entry.status}")

        inferred_kind = self.infer_bundle_kind(entry)
        if bundle_kinds:
            if inferred_kind in bundle_kinds:
                kind_rank = len(bundle_kinds) - bundle_kinds.index(inferred_kind)
                score += 1.2 + 0.25 * kind_rank
                reasons.append(f"bundle_kind={inferred_kind}")
            elif inferred_kind != "unknown":
                score -= 0.4

        inferred_contract = self.infer_level_contract(entry)
        if level_contract:
            if inferred_contract == level_contract:
                score += 1.5
                reasons.append(f"level_contract={inferred_contract}")
            elif inferred_contract and inferred_contract != level_contract:
                score -= 0.75
                reasons.append(f"contract_mismatch={inferred_contract}")

        entry_dims = self._normalized_obs_dims(entry.obs_dims)
        if obs_dims and entry_dims:
            if entry_dims == obs_dims:
                score += 1.0
                reasons.append(f"obs_dims={list(entry_dims)}")
            else:
                overlap = len(set(entry_dims).intersection(obs_dims))
                if overlap:
                    score += 0.35 + 0.15 * overlap
                    reasons.append(f"obs_dim_overlap={overlap}")
                else:
                    score -= 0.3

        if view and view in entry.notes:
            score += 0.1
        if entry.pattern_count:
            score += min(0.4, 0.05 * float(entry.pattern_count) ** 0.5)
        score += min(0.3, max(0.0, entry.density_mean))
        return score, reasons

    @staticmethod
    def _path_exists(resolved_path: str, entry: LibraryEntry) -> bool:
        if not resolved_path:
            return False
        if os.path.exists(resolved_path):
            return True
        base = BundleResolver.bundle_base_path(entry.path)
        if os.path.exists(base + ".l1.pkl"):
            return True
        return os.path.exists(entry.path)
