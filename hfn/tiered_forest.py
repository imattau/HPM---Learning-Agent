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
        )

    def _load_from_cold(self, node_id: str) -> HFN | None:
        """Load a cold node into _hot. Returns the node or None if not found."""
        path = self._cold_path(node_id)
        if not path.exists():
            return None
        data = np.load(path)
        node = HFN(
            mu=data["mu"].copy(),
            sigma=data["sigma"].copy(),
            id=str(data["node_id"][0]),
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

    def _sync_gaussian(self) -> None:
        """Override: compute from hot nodes only (approximation)."""
        hot = list(self._hot.values())
        if not hot:
            self.mu = np.zeros(self._D)
            self.sigma = np.eye(self._D)
        else:
            mus = np.stack([n.mu for n in hot])
            self.mu = mus.mean(axis=0)
            sigmas = np.stack([n.sigma for n in hot])
            self.sigma = sigmas.mean(axis=0)
