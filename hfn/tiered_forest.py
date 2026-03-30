"""
TieredForest — two-level memory tiering for the HFN Forest.

Hot cache (RAM): LRU-capped OrderedDict of full HFN nodes.
Cold store (disk): {cold_dir}/{node_id}.npz for evicted nodes.
mu_index (RAM): mu vectors for ALL nodes (hot + cold) — fast distance screening.

See docs/superpowers/specs/2026-03-29-tiered-forest-design.md
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import psutil

from hfn.forest import Forest
from hfn.hfn import HFN


class TieredForest(Forest):

    def __init__(
        self,
        D: int,
        forest_id: str = "forest",
        cold_dir: str | Path = "/tmp/hfn_cold",
        max_hot: int = 500,
        persistence_floor: float = 0.1,
        sweep_every: int = 100,
        min_free_ram_mb: int = 500,
        protected_ids: set[str] | None = None,
    ) -> None:
        super().__init__(D=D, forest_id=forest_id)
        self._cold_dir = Path(cold_dir)
        self._cold_dir.mkdir(parents=True, exist_ok=True)
        # Clear any stale .npz files from a previous run
        for f in self._cold_dir.glob("*.npz"):
            f.unlink()

        self._max_hot = max_hot
        self._persistence_floor = persistence_floor
        self._sweep_every = sweep_every
        self._min_free_ram_mb = min_free_ram_mb
        self._protected_ids: set[str] = set(protected_ids or [])

        # Hot cache: OrderedDict maintains LRU order (most-recently-used at end)
        self._hot: OrderedDict[str, HFN] = OrderedDict()
        # mu_index: all nodes (hot + cold)
        self._mu_index: dict[str, np.ndarray] = {}
        # Observation counter for sweep triggering
        self._obs_count: int = 0

    # ------------------------------------------------------------------
    # Forest interface overrides
    # ------------------------------------------------------------------

    def register(self, node: HFN) -> None:
        if node.id in self._mu_index:
            return  # idempotent
        self._mu_index[node.id] = node.mu.copy()
        self._hot[node.id] = node
        self._evict_lru_if_needed()
        self._sync_gaussian()

    def deregister(self, node_id: str) -> None:
        if node_id in self._protected_ids:
            return  # no-op for protected nodes
        self._mu_index.pop(node_id, None)
        self._hot.pop(node_id, None)
        cold_path = self._cold_path(node_id)
        if cold_path.exists():
            cold_path.unlink()
        self._sync_gaussian()

    def active_nodes(self) -> list[HFN]:
        """Return hot nodes only. See spec Section 3 for HPM rationale."""
        return list(self._hot.values())

    def __contains__(self, node_id: str) -> bool:  # type: ignore[override]
        return node_id in self._mu_index

    def __len__(self) -> int:
        return len(self._mu_index)

    # ------------------------------------------------------------------
    # New methods
    # ------------------------------------------------------------------

    def set_protected(self, ids: set[str]) -> None:
        self._protected_ids = set(ids)

    def hot_count(self) -> int:
        return len(self._hot)

    def cold_count(self) -> int:
        return len(self._mu_index) - len(self._hot)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cold_path(self, node_id: str) -> Path:
        safe = node_id.replace("/", "_").replace("\\", "_")
        return self._cold_dir / f"{safe}.npz"

    def _evict_to_cold(self, node_id: str) -> None:
        """Write a hot node to disk and remove from _hot."""
        node = self._hot.pop(node_id, None)
        if node is None:
            return
        path = self._cold_path(node_id)
        np.savez_compressed(
            path,
            mu=node.mu,
            sigma=node.sigma,
            node_id=np.array([node.id]),
            use_diag=np.array([node.use_diag]),
        )

    def _load_from_cold(self, node_id: str) -> HFN | None:
        """Load a cold node into _hot. Returns the node or None if not found."""
        path = self._cold_path(node_id)
        if not path.exists():
            return None
        data = np.load(path)
        # use_diag may be absent in files serialised before this change
        use_diag = bool(data["use_diag"][0]) if "use_diag" in data else False
        node = HFN(
            mu=data["mu"].copy(),
            sigma=data["sigma"].copy(),
            id=str(data["node_id"][0]),
            use_diag=use_diag,
        )
        path.unlink()
        self._hot[node_id] = node
        self._hot.move_to_end(node_id)
        self._evict_lru_if_needed()
        return node

    def _evict_lru_if_needed(self) -> None:
        """Evict least-recently-used hot nodes until len(_hot) <= max_hot."""
        while len(self._hot) > self._max_hot:
            lru_id, _ = next(iter(self._hot.items()))
            self._evict_to_cold(lru_id)

    def _on_observe(self) -> None:
        """Call once per observation. Triggers sweep every sweep_every calls."""
        self._obs_count += 1
        if self._obs_count % self._sweep_every == 0:
            self._sweep()

    def _sweep(self) -> None:
        """
        Two-step sweep:

        Step 1 — Emergency RAM check: if free RAM < min_free_ram_mb,
        evict the bottom half of hot nodes (LRU = front of OrderedDict).
        Both protected and unprotected go to cold; neither is deleted.

        Step 2 — Persistence floor (operates on pre-step-1 cold state):
        Any node that was already cold before step 1 ran is treated as
        dormant. Unprotected dormant nodes are deleted entirely. Protected
        dormant nodes stay cold (kept in _mu_index and on disk).
        """
        # Snapshot which nodes are already cold before step 1
        pre_cold_ids = [nid for nid in self._mu_index if nid not in self._hot]

        # Step 1: emergency RAM eviction
        free_mb = psutil.virtual_memory().available / (1024 * 1024)
        if free_mb < self._min_free_ram_mb:
            hot_ids = list(self._hot.keys())
            n_evict = max(1, len(hot_ids) // 2)
            for node_id in hot_ids[:n_evict]:
                self._evict_to_cold(node_id)

        # Step 2: delete unprotected nodes that were already cold before this sweep
        for node_id in pre_cold_ids:
            if node_id not in self._protected_ids:
                self._mu_index.pop(node_id, None)
                cold_path = self._cold_path(node_id)
                if cold_path.exists():
                    cold_path.unlink()

    def retrieve(self, x: np.ndarray, k: int = 5) -> list[HFN]:
        """
        Screen _mu_index by Euclidean distance, load cold top-k candidates,
        return up to k full HFN nodes (hot or promoted from cold).
        """
        if not self._mu_index:
            return []
        scored = sorted(
            self._mu_index.keys(),
            key=lambda nid: float(np.linalg.norm(self._mu_index[nid] - x)),
        )
        result = []
        for node_id in scored[:k]:
            if node_id in self._hot:
                self._hot.move_to_end(node_id)  # refresh LRU
                result.append(self._hot[node_id])
            else:
                node = self._load_from_cold(node_id)
                if node is not None:
                    result.append(node)
        return result

    def get(self, node_id: str) -> HFN | None:
        """Return a full HFN by id, loading from cold if needed."""
        if node_id in self._hot:
            self._hot.move_to_end(node_id)
            return self._hot[node_id]
        if node_id in self._mu_index:
            return self._load_from_cold(node_id)
        return None

    def _sync_gaussian(self) -> None:
        """Override: compute from hot nodes only (approximation)."""
        hot = list(self._hot.values())
        if not hot:
            self.mu = np.zeros(self._D)
            self.sigma = np.eye(self._D)
        else:
            mus = np.stack([n.mu for n in hot])
            self.mu = mus.mean(axis=0)
            # Expand diag nodes to full matrices before averaging
            sigmas = [np.diag(n.sigma) if n.use_diag else n.sigma for n in hot]
            self.sigma = np.mean(sigmas, axis=0)
