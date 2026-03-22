"""ArchiveLibrarian: stateless class for archive-query and global-promotion logic.

Extracted from ContextualPatternStore as part of Phase 3 refactor.
"""
from __future__ import annotations

import json
import os
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hpm.store.contextual_store import SubstrateSignature


@dataclass
class CandidateArchive:
    episode_id: str
    archive_path: Path
    signature: SubstrateSignature
    success: bool


class ArchiveLibrarian:
    """Stateless helper for archive queries and global pattern promotion."""

    def query_archive(self, sig: SubstrateSignature, archive_root: Path) -> list[CandidateArchive]:
        """Coarse filter: return candidates matching grid_size and color_count +-1."""
        candidates = []
        archive_root = Path(archive_root)
        if not archive_root.is_dir():
            return candidates
        for run_dir_name in os.listdir(archive_root):
            index_path = archive_root / run_dir_name / "index.json"
            if not index_path.exists():
                continue
            entries = self._load_index(index_path)
            for entry in entries:
                s = entry.get("signature", {})
                stored_size = s.get("grid_size")
                if stored_size is None:
                    continue
                if tuple(stored_size) != sig.grid_size:
                    continue
                stored_cc = s.get("unique_color_count", -999)
                if abs(stored_cc - sig.unique_color_count) > 1:
                    continue
                archive_path = Path(entry.get("archive_path", ""))
                success_metrics = entry.get("success_metrics", {})
                success = bool(success_metrics.get("correct", False))
                sig_stored = SubstrateSignature(
                    grid_size=tuple(stored_size),
                    unique_color_count=s.get("unique_color_count", 0),
                    object_count=s.get("object_count", 0),
                    aspect_ratio_bucket=s.get("aspect_ratio_bucket", "square"),
                )
                candidates.append(CandidateArchive(
                    episode_id=entry.get("context_id", ""),
                    archive_path=archive_path,
                    signature=sig_stored,
                    success=success,
                ))
        return candidates

    def run_global_pass(self, sqlite_store, tier2_patterns: list, context_id: str,
                        weight_threshold: float, promotion_n: int) -> None:
        """Update context_ids; set is_global=True when threshold reached.

        sqlite_store is the db_path string (path to globals.db).
        tier2_patterns is a list of (pattern, weight, agent_id) tuples.
        """
        db_path = sqlite_store
        conn = sqlite3.connect(db_path)
        try:
            for pattern, weight, agent_id in tier2_patterns:
                if weight <= weight_threshold:
                    continue
                mu_blob = pickle.dumps(pattern.mu)
                existing = conn.execute(
                    "SELECT context_ids FROM global_patterns WHERE id=?",
                    (pattern.id,)
                ).fetchone()
                if existing is None:
                    is_global = 1 if 1 >= promotion_n else 0
                    conn.execute(
                        "INSERT INTO global_patterns "
                        "(id, mu, weight, agent_id, is_global, context_ids) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (pattern.id, mu_blob, weight, agent_id,
                         is_global, json.dumps([context_id]))
                    )
                else:
                    ctx_ids = json.loads(existing[0])
                    if context_id not in ctx_ids:
                        ctx_ids.append(context_id)
                    is_global = 1 if len(ctx_ids) >= promotion_n else 0
                    conn.execute(
                        "UPDATE global_patterns "
                        "SET mu=?, weight=?, is_global=?, context_ids=? WHERE id=?",
                        (mu_blob, weight, is_global, json.dumps(ctx_ids), pattern.id)
                    )
            conn.commit()
        finally:
            conn.close()

    def load_global_patterns(self, sqlite_store) -> list:
        """Return all patterns with is_global=True from SQLiteStore.

        sqlite_store is the db_path string (path to globals.db).
        Returns list of (id, mu_blob, weight, agent_id) tuples.
        """
        db_path = sqlite_store
        if not db_path or not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT id, mu, weight, agent_id FROM global_patterns WHERE is_global=1"
            ).fetchall()
        finally:
            conn.close()
        return list(rows)

    def _load_index(self, index_path: Path) -> list:
        if not index_path.exists():
            return []
        with open(index_path) as f:
            return json.load(f)
