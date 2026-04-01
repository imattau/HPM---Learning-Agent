"""Async controller layer for HFN.

This module keeps the HFN core synchronous and single-writer while providing
async orchestration around it:
- ingest and replay observations through one serialized worker
- prefetch nodes before future observations
- run gap queries outside the core learning loop
- export compact state snapshots for downstream systems
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from hfn import Observer
from hfn.forest import Forest

GapQueryFn = Callable[[np.ndarray, Any], list[str] | Awaitable[list[str]]]


@dataclass(frozen=True)
class PrefetchResult:
    requested_ids: tuple[str, ...]
    found_ids: tuple[str, ...]
    hot_count: int | None
    cold_count: int | None


@dataclass(frozen=True)
class GapQueryResult:
    gap_mu_shape: tuple[int, ...]
    context: Any
    suggestions: tuple[str, ...]
    encoded_count: int
    source: str


@dataclass(frozen=True)
class ControllerSnapshot:
    forest_id: str
    queue_size: int
    total_observations: int
    replayed_observations: int
    prefetched_nodes: int
    gap_queries: int
    last_event: str | None
    last_observed_ids: tuple[str, ...]
    n_nodes: int
    hot_count: int | None
    cold_count: int | None


@dataclass
class _Request:
    kind: str
    payload: dict[str, Any]
    future: asyncio.Future[Any]


@dataclass
class _Stats:
    total_observations: int = 0
    replayed_observations: int = 0
    prefetched_nodes: int = 0
    gap_queries: int = 0
    last_event: str | None = None
    last_observed_ids: tuple[str, ...] = field(default_factory=tuple)


class AsyncHFNController:
    """Single-writer async controller for HFN."""

    def __init__(
        self,
        forest: Forest,
        observer: Observer,
        *,
        gap_query_fn: GapQueryFn | None = None,
        converter=None,
    ) -> None:
        self.forest = forest
        self.observer = observer
        self.gap_query_fn = gap_query_fn
        self.converter = converter
        self._queue: asyncio.Queue[_Request | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._stats = _Stats()
        self._closed = False

    async def __aenter__(self) -> "AsyncHFNController":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="hfn-controller-worker")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.start()
        await self._submit("stop", {})
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None

    async def ingest(self, x: np.ndarray, *, label: str | None = None, metadata: Any = None):
        """Submit one observation to the serialized HFN worker."""
        return await self._submit(
            "observe",
            {"x": np.asarray(x, dtype=np.float64), "label": label, "metadata": metadata, "replay": False},
        )

    async def replay(self, observations: list[np.ndarray], *, label: str = "replay") -> list[Any]:
        """Replay a batch of observations through the same worker."""
        results: list[Any] = []
        for x in observations:
            results.append(await self._submit(
                "observe",
                {"x": np.asarray(x, dtype=np.float64), "label": label, "metadata": None, "replay": True},
            ))
        return results

    async def prefetch(self, node_ids: list[str]) -> PrefetchResult:
        """Promote likely-needed nodes into the hot tier by id."""
        return await self._submit("prefetch", {"node_ids": tuple(node_ids)})

    async def request_gap_query(self, gap_mu: np.ndarray, context: Any = None) -> GapQueryResult:
        """Run the external gap query outside the HFN mutation path."""
        if self.gap_query_fn is None:
            return GapQueryResult(tuple(np.asarray(gap_mu).shape), context, tuple(), 0, "disabled")

        if inspect.iscoroutinefunction(self.gap_query_fn):
            suggestions = await self.gap_query_fn(np.asarray(gap_mu, dtype=np.float64), context)
        else:
            suggestions = await asyncio.to_thread(self.gap_query_fn, np.asarray(gap_mu, dtype=np.float64), context)

        raw = tuple(str(s) for s in suggestions)
        encoded_count = 0
        if self.converter is not None and raw:
            encoded: list[np.ndarray] = []
            for item in raw:
                encoded.extend(self.converter.encode(item, self.forest._D))
            encoded_count = len(encoded)

        self._stats.gap_queries += 1
        return GapQueryResult(tuple(np.asarray(gap_mu).shape), context, raw, encoded_count, "external")

    async def snapshot_state(self) -> ControllerSnapshot:
        """Export a compact summary for the downstream controller."""
        return await self._submit("snapshot", {})

    async def _submit(self, kind: str, payload: dict[str, Any]) -> Any:
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put(_Request(kind=kind, payload=payload, future=future))
        return await future

    async def _worker(self) -> None:
        while True:
            request = await self._queue.get()
            if request is None:
                return
            try:
                if request.kind == "stop":
                    if not request.future.done():
                        request.future.set_result(None)
                    return
                result = self._handle_request(request.kind, request.payload)
                if inspect.isawaitable(result):
                    result = await result
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as exc:  # pragma: no cover - surfaced in tests
                if not request.future.done():
                    request.future.set_exception(exc)

    def _handle_request(self, kind: str, payload: dict[str, Any]) -> Any:
        if kind == "observe":
            return self._handle_observe(payload["x"], payload.get("label"), payload.get("metadata"), payload.get("replay", False))
        if kind == "prefetch":
            return self._handle_prefetch(payload["node_ids"])
        if kind == "snapshot":
            return self._snapshot()
        raise ValueError(f"Unknown controller request kind: {kind}")

    def _handle_observe(self, x: np.ndarray, label: str | None, metadata: Any, replay: bool) -> dict[str, Any]:
        result = self.observer.observe(np.asarray(x, dtype=np.float64))
        on_observe = getattr(self.forest, "_on_observe", None)
        if callable(on_observe):
            on_observe()

        explained_ids = tuple(node.id for node in result.explanation_tree)
        self._stats.total_observations += 1
        if replay:
            self._stats.replayed_observations += 1
        self._stats.last_event = label or ("replay" if replay else "ingest")
        self._stats.last_observed_ids = explained_ids
        return {
            "label": label,
            "metadata": metadata,
            "replay": replay,
            "residual_surprise": result.residual_surprise,
            "explained_ids": explained_ids,
            "accuracy_scores": result.accuracy_scores,
        }

    def _handle_prefetch(self, node_ids: tuple[str, ...]) -> PrefetchResult:
        found: list[str] = []
        for node_id in node_ids:
            getter = getattr(self.forest, "get", None)
            node = getter(node_id) if callable(getter) else None
            if node is not None:
                found.append(node_id)
        self._stats.prefetched_nodes += len(found)
        return PrefetchResult(
            requested_ids=tuple(node_ids),
            found_ids=tuple(found),
            hot_count=self._maybe_count("hot_count"),
            cold_count=self._maybe_count("cold_count"),
        )

    def _snapshot(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            forest_id=self.forest.id,
            queue_size=self._queue.qsize(),
            total_observations=self._stats.total_observations,
            replayed_observations=self._stats.replayed_observations,
            prefetched_nodes=self._stats.prefetched_nodes,
            gap_queries=self._stats.gap_queries,
            last_event=self._stats.last_event,
            last_observed_ids=self._stats.last_observed_ids,
            n_nodes=len(self.forest),
            hot_count=self._maybe_count("hot_count"),
            cold_count=self._maybe_count("cold_count"),
        )

    def _maybe_count(self, method_name: str) -> int | None:
        method = getattr(self.forest, method_name, None)
        return int(method()) if callable(method) else None


__all__ = [
    "AsyncHFNController",
    "ControllerSnapshot",
    "GapQueryResult",
    "PrefetchResult",
]
