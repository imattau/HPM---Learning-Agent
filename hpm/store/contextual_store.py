"""ContextualPatternStore: wraps TieredStore to archive and warm-start Tier 2 across episodes.

Implements the HPM spec: archive Tier 2 state per episode, warm-start from structurally
similar past tasks (coarse + fine NLL filter), promote globally useful patterns to SQLite.

Phase 1: Archive + warm-start
Phase 2: Global pass (SQLite) + injection
Phase 3: Delegation to Librarian/Forecaster (done in contextual_store, librarian, forecaster)
"""
from __future__ import annotations

import datetime
import json
import os
import pickle
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SubstrateSignature:
    """Structural fingerprint of an ARC task grid."""
    grid_size: tuple[int, int]
    unique_color_count: int
    object_count: int
    aspect_ratio_bucket: str  # "square" | "landscape" | "portrait"


def extract_signature(grid: np.ndarray) -> SubstrateSignature:
    """Compute a structural fingerprint for an ARC task grid.

    Background value is 0. Objects are 4-connected non-zero components.
    aspect_ratio_bucket: square (0.8 <= r <= 1.25), landscape (< 0.8), portrait (> 1.25)
    """
    rows, cols = grid.shape
    unique_colors = set(int(v) for v in grid.flat if int(v) != 0)
    unique_color_count = len(unique_colors)

    # 4-connected component labelling for non-zero cells
    visited = np.zeros_like(grid, dtype=bool)
    object_count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                object_count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or grid[cr, cc] == 0:
                        continue
                    visited[cr, cc] = True
                    stack.extend([(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)])

    ratio = rows / cols if cols > 0 else 1.0
    if ratio < 0.8:
        bucket = "landscape"
    elif ratio > 1.25:
        bucket = "portrait"
    else:
        bucket = "square"

    return SubstrateSignature(
        grid_size=(rows, cols),
        unique_color_count=unique_color_count,
        object_count=object_count,
        aspect_ratio_bucket=bucket,
    )


class ContextualPatternStore:
    """Wraps TieredStore to archive Tier 2 across episodes and warm-start from past tasks.

    All TieredStore public methods are delegated unchanged (transparent to Agent.step()).
    Lifecycle hooks (begin_context / end_context) are called explicitly by the benchmark harness.

    Archive uses pickle for Tier 2 GaussianPattern objects (spec-mandated: same objects
    TieredStore holds in memory during a run).
    """

    def __init__(
        self,
        tiered_store,
        archive_dir: str,
        run_id: Optional[str] = None,
        fingerprint_nll_threshold: float = 50.0,
        global_weight_threshold: float = 0.6,
        global_promotion_n: int = 5,
        l3_agents: list | None = None,
    ):
        self._store = tiered_store
        self._archive_dir = archive_dir
        self._run_id = run_id or str(uuid.uuid4())
        self._fingerprint_nll_threshold = fingerprint_nll_threshold
        self._global_weight_threshold = global_weight_threshold
        self._global_promotion_n = global_promotion_n
        self._last_sig: Optional[SubstrateSignature] = None
        self._l3_agents: list | None = l3_agents

        # Phase 3: delegate to dedicated classes
        from hpm.store.archive_librarian import ArchiveLibrarian
        from hpm.store.archive_forecaster import ArchiveForecaster
        self._librarian = ArchiveLibrarian()
        self._forecaster = ArchiveForecaster()

        self._init_db()

    def _init_db(self) -> None:
        """Create global_patterns table in globals.db if not exists."""
        run_dir = os.path.join(self._archive_dir, self._run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._db_path = os.path.join(run_dir, "globals.db")
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_patterns (
                id TEXT PRIMARY KEY,
                mu BLOB NOT NULL,
                weight REAL NOT NULL,
                agent_id TEXT NOT NULL,
                is_global INTEGER DEFAULT 0,
                context_ids TEXT DEFAULT '[]'
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Archive lifecycle hooks (called by benchmark harness)
    # ------------------------------------------------------------------

    def begin_context(self, sig: SubstrateSignature, first_obs: list) -> str:
        """Warm-start Tier 2 and L3 agents from best matching past episode; return context_id."""
        context_id = str(uuid.uuid4())
        self._last_sig = sig

        # Coarse filter: substrate signature
        candidates = self._librarian.query_archive(sig, Path(self._archive_dir))

        # Fine ranking: L3 cosine similarity if l3_agents available, else NLL (existing path).
        # Note: _rank_by_l3 returns raw candidate objects (candidate.archive_path).
        # The forecaster.rank() path returns RankedCandidate wrappers (best.candidate.archive_path).
        l3_bundles = []
        if self._l3_agents:
            from hpm.agents.hierarchical import extract_bundle, encode_bundle
            current_l3_vecs = [encode_bundle(extract_bundle(a)) for a in self._l3_agents]
            ranked_candidates = self._rank_by_l3(candidates, current_l3_vecs)
            if ranked_candidates:
                best = ranked_candidates[0]
                l3_bundles = self._load_archive(str(best.archive_path))
        else:
            ranked = self._forecaster.rank(
                candidates, first_obs, nll_threshold=self._fingerprint_nll_threshold
            )
            if ranked:
                best = ranked[0]
                l3_bundles = self._load_archive(str(best.candidate.archive_path))

        # Inject L3 bundles as seed patterns (no-op if l3_agents=None or l3_bundles=[])
        self._inject_l3(l3_bundles)

        # Phase 3: delegate global injection to librarian
        self._inject_globals()

        # Delegate to inner TieredStore so Tier 1 context is created and agents can save patterns
        self._store.begin_context(context_id)
        return context_id

    def end_context(self, context_id: str, success_metrics: dict,
                    similarity_threshold: float = 0.95) -> None:
        """Serialise Tier 2 to archive, update index, run global pass."""
        run_dir = os.path.join(self._archive_dir, self._run_id)
        os.makedirs(run_dir, exist_ok=True)

        archive_path = os.path.join(run_dir, f"{context_id}.pkl")
        tmp_path_pkl = archive_path + ".tmp.pkl"
        tier2_state = self._store.query_tier2_all()

        correct = success_metrics.get("correct", False)
        if self._l3_agents and correct:
            from hpm.agents.hierarchical import extract_bundle
            l3_bundles = []
            for a in self._l3_agents:
                b = extract_bundle(a)  # cache result — read all fields from same bundle
                l3_bundles.append((b.mu.copy(), float(b.weight), float(b.epistemic_loss)))
            payload = {"tier2": tier2_state, "l3_bundles": l3_bundles}
        else:
            payload = tier2_state  # old list format when no l3_agents or task failed

        with open(tmp_path_pkl, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path_pkl, archive_path)

        index_path = os.path.join(run_dir, "index.json")
        index = self._load_index(index_path)
        sig = self._last_sig
        entry = {
            "context_id": context_id,
            "signature": {
                "grid_size": list(sig.grid_size) if sig else None,
                "unique_color_count": sig.unique_color_count if sig else None,
                "object_count": sig.object_count if sig else None,
                "aspect_ratio_bucket": sig.aspect_ratio_bucket if sig else None,
            },
            "success_metrics": success_metrics,
            "archive_path": archive_path,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        index.append(entry)
        with open(index_path, "w") as f:
            json.dump(index, f)

        # Delegate to inner TieredStore so similarity_merge / negative_merge run
        correct = success_metrics.get("correct", False)
        self._store.end_context(context_id, correct=correct,
                                similarity_threshold=similarity_threshold)

        # Phase 3: delegate global pass to librarian (after end_context promotes patterns)
        self._librarian.run_global_pass(
            sqlite_store=self._db_path,
            tier2_patterns=self._store.query_tier2_all(),
            context_id=context_id,
            weight_threshold=self._global_weight_threshold,
            promotion_n=self._global_promotion_n,
        )

    # ------------------------------------------------------------------
    # Internal helpers — Phase 1
    # ------------------------------------------------------------------

    def _load_archive(self, archive_path: str) -> list:
        """Load archive from disk. Replace Tier 2 with archived patterns.

        Returns l3_bundles: list of (mu_array, weight, epistemic_loss) tuples.
        Returns [] for old list-format archives (backward compatible).
        """
        with open(archive_path, "rb") as f:
            records = pickle.load(f)

        if isinstance(records, list):
            # Old format: plain list of (pattern, weight, agent_id)
            tier2_records = records
            l3_bundles = []
        else:
            # New format: dict with "tier2" and "l3_bundles" keys
            tier2_records = records.get("tier2", [])
            l3_bundles = records.get("l3_bundles", [])

        # Replace Tier 2 (clean REPLACE, not additive)
        self._store._tier2._data.clear()
        for pattern, weight, agent_id in tier2_records:
            self._store.promote_to_tier2(pattern, weight, agent_id)

        return l3_bundles

    def _inject_l3(self, l3_bundles: list) -> None:
        """Inject stored L3 bundle arrays as GaussianPattern seeds into L3 agents' stores.

        Each bundle is (mu_array, weight, epistemic_loss). Uses identity covariance —
        weight already encodes certainty; broad prior avoids over-constraining agents
        before they see the new task's training data.
        """
        if not self._l3_agents or not l3_bundles:
            return
        from hpm.patterns.factory import make_pattern
        for mu, weight, epistemic_loss in l3_bundles:
            pattern = make_pattern(mu=mu, scale=np.eye(len(mu)), pattern_type="gaussian")
            for agent in self._l3_agents:
                agent.store.save(pattern, weight, agent.agent_id)

    def _rank_by_l3(self, candidates: list, current_l3_vecs: list) -> list:
        """Re-rank candidates by cosine similarity of stored L3 bundles to current L3 state.

        candidates: raw candidate objects from ArchiveLibrarian (have .archive_path attribute)
        current_l3_vecs: list of encoded L3 bundle vectors (one per L3 agent)

        Candidates without L3 bundles (old-format archives) are moved to the end.
        Returns re-ranked candidates list (most similar first).

        Note: loads the full archive pkl per candidate (pickle does not support partial
        key loading). Acceptable at benchmark scale (hundreds of tasks).
        """
        def _score(candidate) -> float:
            try:
                with open(str(candidate.archive_path), "rb") as f:
                    records = pickle.load(f)
            except Exception:
                return -1.0
            if isinstance(records, list):
                return -1.0  # old format — no L3 bundles
            l3_bundles = records.get("l3_bundles", [])
            if not l3_bundles or not current_l3_vecs:
                return -1.0
            sims = []
            for mu, _w, _eps in l3_bundles:
                norm_stored = np.linalg.norm(mu)
                for vec in current_l3_vecs:
                    norm_cur = np.linalg.norm(vec)
                    if norm_stored < 1e-12 or norm_cur < 1e-12:
                        continue
                    if len(mu) != len(vec):
                        continue  # dimension mismatch — skip
                    sims.append(float(np.dot(mu, vec) / (norm_stored * norm_cur)))
            return float(np.mean(sims)) if sims else -1.0

        scored = [(candidate, _score(candidate)) for candidate in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def _load_index(self, index_path: str) -> list:
        if not os.path.exists(index_path):
            return []
        with open(index_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Internal helpers — Phase 2
    # ------------------------------------------------------------------

    def _inject_globals(self) -> None:
        """Load all is_global=True patterns from globals.db into Tier 2."""
        # Phase 3: delegate to librarian
        rows = self._librarian.load_global_patterns(
            sqlite_store=self._db_path if hasattr(self, "_db_path") else None
        )
        for pid, mu_blob, weight, agent_id in rows:
            # Skip if already present in Tier 2
            if self._store._tier2.has(pid):
                continue
            mu = pickle.loads(mu_blob)
            from hpm.patterns.factory import make_pattern, pattern_from_dict
            p = make_pattern(mu=mu, scale=np.eye(len(mu)), pattern_type="gaussian")
            p_dict = p.to_dict()
            p_dict["id"] = pid
            p_restored = pattern_from_dict(p_dict)
            self._store.promote_to_tier2(p_restored, weight, agent_id)

    # ------------------------------------------------------------------
    # TieredStore delegation (transparent to Agent.step())
    # ------------------------------------------------------------------

    def save(self, pattern, weight: float, agent_id: str) -> None:
        self._store.save(pattern, weight, agent_id)

    def load(self, pattern_id: str) -> tuple:
        return self._store.load(pattern_id)

    def query(self, agent_id: str) -> list:
        return self._store.query(agent_id)

    def query_all(self) -> list:
        return self._store.query_all()

    def query_tier2(self, agent_id: str) -> list:
        return self._store.query_tier2(agent_id)

    def query_tier2_all(self) -> list:
        return self._store.query_tier2_all()

    def delete(self, pattern_id: str) -> None:
        self._store.delete(pattern_id)

    def update_weight(self, pattern_id: str, weight: float) -> None:
        self._store.update_weight(pattern_id, weight)

    def similarity_merge(self, context_id: str, **kwargs) -> None:
        self._store.similarity_merge(context_id, **kwargs)

    def promote_to_tier2(self, pattern, weight: float, agent_id: str, **kwargs) -> None:
        self._store.promote_to_tier2(pattern, weight, agent_id, **kwargs)

    def query_negative(self, agent_id: str) -> list:
        return self._store.query_negative(agent_id)

    def query_tier2_negative_all(self) -> list:
        return self._store.query_tier2_negative_all()

    def negative_merge(self, context_id: str, **kwargs) -> None:
        self._store.negative_merge(context_id, **kwargs)
