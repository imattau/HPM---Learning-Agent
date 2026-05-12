from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell


class PatternPager:
    """
    Minimal async disk-backed pattern archive.

    Spills the lowest-weight patterns to per-pattern JSON files on a background
    worker so training does not wait on archive I/O.
    """

    def __init__(
        self,
        cache_dir: str,
        agent_name: str,
        max_active_patterns: int = 5000,
        spill_fraction: float = 0.25,
    ):
        self.cache_dir = cache_dir
        self.agent_name = agent_name
        self.max_active_patterns = max_active_patterns
        self.spill_fraction = spill_fraction
        self.archive_dir = os.path.join(self.cache_dir, self.agent_name)
        os.makedirs(self.archive_dir, exist_ok=True)
        self.journal_path = os.path.join(self.archive_dir, "journal.jsonl")
        self.index_path = os.path.join(self.archive_dir, "index.jsonl")

        self._queue: "Queue[Optional[Dict[str, object]]]" = Queue()
        self._pending: Dict[str, Dict[str, object]] = {}
        self._index: Dict[str, Dict[str, object]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker.start()
        self._replay_journal()
        self._load_index()
        atexit.register(self.close)

    @staticmethod
    def _serialize_cell(cell: Cell) -> Dict[str, object]:
        return {
            "name": cell.name,
            "dim": cell.dim,
            "embedding": cell.as_numpy().tolist(),
            "source": cell.source.name if cell.source is not None else None,
            "target": cell.target.name if cell.target is not None else None,
            "weight": float(cell.weight),
        }

    @staticmethod
    def _deserialize_cell(payload: Dict[str, object], lookup: Dict[str, Cell]) -> Cell:
        source_name = payload.get("source")
        target_name = payload.get("target")
        return Cell(
            name=str(payload["name"]),
            dim=int(payload["dim"]),
            embedding=np.asarray(payload["embedding"], dtype=float),
            source=lookup.get(str(source_name)) if source_name is not None else None,
            target=lookup.get(str(target_name)) if target_name is not None else None,
            weight=float(payload.get("weight", 1.0)),
        )

    def _path_for_name(self, name: str) -> str:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
        return os.path.join(self.archive_dir, f"{digest}.json")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        if denom <= 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _to_numpy(vector: object) -> np.ndarray:
        if isinstance(vector, np.ndarray):
            return vector.astype(float, copy=False)
        try:
            import torch

            if isinstance(vector, torch.Tensor):
                return vector.detach().cpu().numpy().astype(float, copy=False)
        except Exception:
            pass
        return np.asarray(vector, dtype=float)

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                if payload is not None:
                    self._write_payload(payload)
                    with self._lock:
                        self._pending.pop(str(payload["name"]), None)
            finally:
                self._queue.task_done()

    def _write_payload(self, payload: Dict[str, object]) -> None:
        name = str(payload["name"])
        path = self._path_for_name(name)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)

    def _append_journal(self, payload: Dict[str, object]) -> None:
        with open(self.journal_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _append_index(self, payload: Dict[str, object]) -> None:
        with open(self.index_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _upsert_index(self, payload: Dict[str, object]) -> None:
        name = str(payload["name"])
        self._index[name] = payload

    def _load_index(self) -> None:
        self._index = {}
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._upsert_index(payload)
                if self._index:
                    return
            except FileNotFoundError:
                pass

        self._rebuild_index_from_archive()

    def _rebuild_index_from_archive(self) -> None:
        for fname in os.listdir(self.archive_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.archive_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            self._upsert_index(payload)

        if self._index:
            with open(self.index_path, "w", encoding="utf-8") as handle:
                for payload in self._index.values():
                    handle.write(json.dumps(payload) + "\n")

    def _replay_journal(self) -> None:
        if not os.path.exists(self.journal_path):
            return

        replayed: List[Dict[str, object]] = []
        try:
            with open(self.journal_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        replayed.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return

        for payload in replayed:
            self._write_payload(payload)
            self._append_index(payload)
            self._upsert_index(payload)

        try:
            os.remove(self.journal_path)
        except FileNotFoundError:
            pass

    def enqueue_save(self, cell: Cell) -> None:
        payload = self._serialize_cell(cell)
        with self._lock:
            self._pending[payload["name"]] = payload
            self._append_journal(payload)
            self._append_index(payload)
            self._upsert_index(payload)
        self._queue.put(payload)

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self.flush()
        self._stop_event.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def flush(self) -> None:
        self._queue.join()

    def has(self, name: str) -> bool:
        return os.path.exists(self._path_for_name(name))

    def save(self, cell: Cell) -> None:
        self.enqueue_save(cell)

    def load(self, name: str, lookup: Dict[str, Cell]) -> Optional[Cell]:
        with self._lock:
            pending = self._pending.get(name)
        if pending is not None:
            return self._deserialize_cell(pending, lookup)

        path = self._path_for_name(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        return self._deserialize_cell(payload, lookup)

    def load_nearest(
        self,
        query_embedding: object,
        lookup: Dict[str, Cell],
        min_similarity: float = 0.85,
        exclude_names: Optional[Sequence[str]] = None,
    ) -> Optional[Cell]:
        query = self._to_numpy(query_embedding)
        if query.size == 0:
            return None

        exclude = set(exclude_names or [])
        best_payload: Optional[Dict[str, object]] = None
        best_score = float("-inf")

        with self._lock:
            pending_items = list(self._pending.values())

        candidates: List[Dict[str, object]] = []
        candidates.extend(pending_items)
        candidates.extend(self._index.values())

        for payload in candidates:
            name = str(payload.get("name", ""))
            if name in exclude:
                continue

            emb = payload.get("embedding")
            if emb is None:
                continue
            candidate = np.asarray(emb, dtype=float)
            if candidate.shape != query.shape:
                continue

            score = self._cosine_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_payload = payload

        if best_payload is None or best_score < min_similarity:
            return None
        return self._deserialize_cell(best_payload, lookup)

    def iter_index_payloads(self) -> List[Dict[str, object]]:
        with self._lock:
            payloads = list(self._index.values())
            payloads.extend(self._pending.values())
        return payloads

    def load_from_payload(self, payload: Dict[str, object], lookup: Dict[str, Cell]) -> Cell:
        return self._deserialize_cell(payload, lookup)

    def select_evictions(self, patterns: Sequence[Cell], weights: Sequence[float]) -> List[int]:
        if len(patterns) <= self.max_active_patterns:
            return []

        excess = len(patterns) - self.max_active_patterns
        spill_count = max(excess, int(len(patterns) * self.spill_fraction))
        spill_count = max(1, min(spill_count, len(patterns)))
        weighted = sorted(
            enumerate(weights),
            key=lambda item: float(item[1]),
        )
        return [idx for idx, _ in weighted[:spill_count]]
