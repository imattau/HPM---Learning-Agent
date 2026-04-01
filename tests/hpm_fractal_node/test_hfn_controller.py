from __future__ import annotations

import asyncio

import numpy as np

from hfn import Forest, HFN, Observer, calibrate_tau
from hfn.hfn_controller import AsyncHFNController


def _make_controller():
    forest = Forest(D=2, forest_id="test_forest")
    prior = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2), id="prior")
    forest.register(prior)
    obs = Observer(forest, tau=100.0, protected_ids={"prior"})
    queries: list[tuple[np.ndarray, object]] = []

    def gap_query_fn(gap_mu: np.ndarray, context=None):
        queries.append((gap_mu.copy(), context))
        return ["gap one", "gap two"]

    controller = AsyncHFNController(forest, obs, gap_query_fn=gap_query_fn)
    return controller, forest, queries


async def _exercise_controller():
    controller, forest, queries = _make_controller()
    async with controller:
        ingest_result = await controller.ingest(np.array([0.0, 0.0]), label="seed")
        replay_results = await controller.replay([np.array([0.0, 0.0]), np.array([0.0, 0.0])])
        prefetch_result = await controller.prefetch(["prior", "missing"])
        gap_result = await controller.request_gap_query(np.array([1.0, 2.0]), context={"source": "test"})
        snapshot = await controller.snapshot_state()

    return ingest_result, replay_results, prefetch_result, gap_result, snapshot, forest, queries


def test_async_controller_routes_observations_and_exports_state():
    ingest_result, replay_results, prefetch_result, gap_result, snapshot, forest, queries = asyncio.run(_exercise_controller())

    assert ingest_result["replay"] is False
    assert len(ingest_result["explained_ids"]) == 1
    assert len(replay_results) == 2
    assert prefetch_result.requested_ids == ("prior", "missing")
    assert prefetch_result.found_ids == ("prior",)
    assert gap_result.suggestions == ("gap one", "gap two")
    assert gap_result.encoded_count == 0
    assert len(queries) == 1
    assert snapshot.total_observations == 3
    assert snapshot.replayed_observations == 2
    assert snapshot.prefetched_nodes == 1
    assert snapshot.gap_queries == 1
    assert snapshot.n_nodes == len(forest)
    assert snapshot.last_event == "replay"
