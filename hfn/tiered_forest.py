"""
TieredForest — two-level memory tiering for the HFN Forest.

Optimized for SP43 "Fractal Retrieval":
Uses the hierarchical structure of HFNs to prune search space.
"""
from __future__ import annotations
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hfn.hfn import HFN
from hfn.forest import Forest

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore


class TieredForest(Forest):
    """
    A Forest that uses tiering and hierarchical pruning for retrieval.
    """
    def __init__(
        self,
        D: int,
        cold_dir: Path | str,
        hot_cap: int = 100,
        forest_id: str = "forest",
        protected_ids: set[str] | None = None,
        max_hot: int | None = None,
        sweep_every: int = 0,
        min_free_ram_mb: int = 0,
    ):
        super().__init__(D, forest_id=forest_id)
        self._cold_dir = Path(cold_dir)
        if protected_ids:
            self._protected_ids = set(protected_ids)
        if max_hot is not None:
            hot_cap = max_hot
        self._hot_cap = hot_cap
        self._hot: OrderedDict[str, HFN] = OrderedDict()
        self._mu_index: dict[str, np.ndarray] = {}
        self._cold_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_every = sweep_every
        self._min_free_ram_mb = min_free_ram_mb
        self._obs_count: int = 0

        # PERSISTENT CACHES
        self._summary_ids: set[str] = set()
        self._root_ids: set[str] = set()
        
        self._load_index()

    def register(self, node: HFN, skip_cache: bool = False) -> None:
        node_id = node.id
        self._mu_index[node_id] = node.mu
        self._hot[node_id] = node
        self._hot.move_to_end(node_id)
        
        if not skip_cache:
            # Update Hierarchy Cache
            if not self.get_parents(node_id):
                self._root_ids.add(node_id)
            
            if node.children():
                self._summary_ids.add(node_id)
                for child in node.children():
                    self._root_ids.discard(child.id)
            
        if len(self._hot) > self._hot_cap:
            self._evict_lru()

    def rebuild_hierarchy_cache(self) -> None:
        """Efficiently rebuild _root_ids and _summary_ids without O(N^2)."""
        all_ids = list(self._mu_index.keys())
        self._root_ids = set(all_ids)
        self._summary_ids.clear()
        
        for nid in all_ids:
            node = self.get(nid)
            if node and node.children():
                self._summary_ids.add(nid)
                for child in node.children():
                    self._root_ids.discard(child.id)
        
        pass # print(f"      [DEBUG] Forest.rebuild_hierarchy_cache: nodes={len(all_ids)}, roots={len(self._root_ids)}")

    def hot_count(self) -> int:
        """Number of nodes currently in the hot (in-memory) cache."""
        return len(self._hot)

    def cold_count(self) -> int:
        """Number of nodes only in cold storage (not in hot cache)."""
        return len(self._mu_index) - len(self._hot)

    def deregister(self, node_id: str) -> None:
        if node_id in self._protected_ids:
            return  # protected nodes cannot be deregistered
        if node_id in self._mu_index:
            self._mu_index.pop(node_id)
            self._hot.pop(node_id, None)
            self._summary_ids.discard(node_id)
            self._root_ids.discard(node_id)
            cold_path = self._cold_path(node_id)
            if cold_path.exists():
                cold_path.unlink()

    def retrieve(self, x: np.ndarray, k: int = 5) -> list[HFN]:
        """
        Hierarchical retrieval: Use hierarchy to prune by distance.
        """
        if not self._mu_index:
            return []

        # Start with cached roots or fallback
        roots = list(self._root_ids)
        if not roots:
            roots = list(self._mu_index.keys())[:50]

        # DEBUG: Check manifold consistency
        if k > 0:
            sample_id = next(iter(self._mu_index))
            # print(f"      [DEBUG] Forest.retrieve: D={self._D}, index_size={len(self._mu_index)}, sample_id={sample_id[:8]}, x_shape={x.shape}")

        candidates = []
        # FIX: Copy the list so we don't drain 'roots' for the debug print
        frontier = list(roots) 
        seen = set()
        
        # Budget for total nodes to inspect
        budget = max(100, k * 10)

        while frontier and budget > 0:
            try:
                # Sort by distance (mu - x)
                frontier.sort(key=lambda nid: float(np.sum((self._mu_index[nid] - x)**2)))
            except ValueError as e:
                return []
            
            nid = frontier.pop(0)
            if nid in seen: continue
            seen.add(nid)
            budget -= 1
            
            node = self.get(nid)
            if not node:
                continue
            
            candidates.append(node)
            
            # Expand children safely
            children = node.children()
            if children:
                for c in children:
                    if c.id not in seen and c.id in self._mu_index:
                        frontier.append(c.id)

        # Final sort
        candidates.sort(key=lambda n: float(np.sum((n.mu - x)**2)))
        res = candidates[:k]

        # Promote top-k results to hot cache (MRU position)
        for node in res:
            if node.id not in self._hot:
                self._hot[node.id] = node
                if len(self._hot) > self._hot_cap:
                    self._evict_lru()
            else:
                self._hot.move_to_end(node.id)

        return res

    def get(self, node_id: str) -> HFN | None:
        """Return node by id, checking hot cache and then cold store."""
        if node_id in self._hot:
            self._hot.move_to_end(node_id)
            return self._hot[node_id]
        if node_id in self._mu_index:
            return self._load_from_cold(node_id)
        return None

    def active_nodes(self) -> list[HFN]:
        """Return all full HFN nodes from both tiers."""
        nodes = []
        for nid in self._mu_index.keys():
            node = self.get(nid)
            if node: nodes.append(node)
        return nodes

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._mu_index

    def __len__(self) -> int:
        return len(self._mu_index)

    def _load_index(self) -> None:
        """Scan cold_dir to rebuild _mu_index, then restore hierarchy cache."""
        for p in self._cold_dir.glob("*.npz"):
            try:
                data = np.load(p)
                node_id = p.stem
                self._mu_index[node_id] = data["mu"]
            except Exception:
                continue
        if self._mu_index:
            self.rebuild_hierarchy_cache()

    def _cold_path(self, node_id: str) -> Path:
        return self._cold_dir / f"{node_id}.npz"

    def _load_from_cold(self, node_id: str) -> HFN | None:
        path = self._cold_path(node_id)
        if not path.exists(): return None
        try:
            data = np.load(path)
            node = HFN(
                mu=data["mu"],
                sigma=data["sigma"],
                id=node_id,
                use_diag=bool(data["use_diag"])
            )
            # Restore child links if saved
            child_ids_str = str(data["child_ids_str"]) if "child_ids_str" in data else ""
            for cid in (child_ids_str.split(",") if child_ids_str else []):
                child = self.get(cid)
                if child is not None:
                    node.add_edge(node, child, "PART_OF")
            # Re-register locally
            self._hot[node_id] = node
            if len(self._hot) > self._hot_cap: self._evict_lru()
            return node
        except Exception:
            return None

    def _evict_lru(self) -> None:
        # Find the LRU node that isn't protected
        evict_id = None
        for candidate in list(self._hot.keys()):
            if candidate not in self._protected_ids:
                evict_id = candidate
                break
        if evict_id is None:
            return  # All hot nodes are protected — nothing to evict
        node = self._hot.pop(evict_id)
        nid = evict_id
        # Save to cold
        path = self._cold_path(nid)
        # Store child IDs as a comma-joined string to avoid pickle
        child_ids_str = ",".join(c.id for c in node.children()) if node.children() else ""
        np.savez_compressed(
            path,
            mu=node.mu,
            sigma=node.sigma,
            use_diag=node.use_diag,
            child_ids_str=np.array(child_ids_str),
        )

    def _on_observe(self) -> None:
        self._obs_count += 1
        if self._sweep_every > 0 and self._obs_count % self._sweep_every == 0:
            self._sweep()

    def _sweep(self) -> None:
        """Two-step sweep: RAM-pressure eviction, then persistence-floor deletion."""
        # Snapshot cold IDs before step 1 so step 2 doesn't delete freshly evicted nodes
        cold_ids_before = [nid for nid in self._mu_index if nid not in self._hot]

        # Step 1: RAM pressure — evict LRU half of hot nodes to cold
        if self._min_free_ram_mb > 0:
            free_mb = 0
            if _psutil is not None:
                free_mb = _psutil.virtual_memory().available / (1024 * 1024)
            if free_mb < self._min_free_ram_mb:
                n_hot = len(self._hot)
                evict_count = n_hot // 2
                for _ in range(evict_count):
                    if not self._hot:
                        break
                    self._evict_lru()

        # Step 2: Persistence floor — delete cold unprotected nodes that were cold before sweep
        for nid in cold_ids_before:
            if nid in self._protected_ids:
                continue
            self._mu_index.pop(nid, None)
            self._summary_ids.discard(nid)
            self._root_ids.discard(nid)
            cold_path = self._cold_path(nid)
            if cold_path.exists():
                cold_path.unlink()

    def _sync_gaussian(self) -> None:
        hot = list(self._hot.values())
        if not hot:
            self.mu = np.zeros(self._D)
            self.sigma = np.eye(self._D)
        else:
            mus = np.stack([n.mu for n in hot])
            self.mu = mus.mean(axis=0)
            sigmas = [np.diag(n.sigma) if n.use_diag else n.sigma for n in hot]
            self.sigma = np.mean(sigmas, axis=0)
