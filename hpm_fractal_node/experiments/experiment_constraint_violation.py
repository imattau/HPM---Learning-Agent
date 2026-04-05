"""
Experiment: Constraint Violation Learning

Tests HFN system's ability to detect and attempt repair of structural constraint
violations. Exercises ResolutionRequest pathway, topological scoring, and
cost-aware node creation.

Three trials:
  T1: Clean match (A→B→C) — should succeed
  T2: Unknown terminus (A→B→D) — should emit ResolutionRequest
  T3: Inconsistent terminus (A→B→E) — should emit ResolutionRequest (topo fail)
"""
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn.decoder import Decoder, ResolutionRequest
from hfn import calibrate_tau

D = 30


def make_node(name: str, pos: int, d: int = D, sigma: float = 1e-4) -> HFN:
    """Create a test node at position pos along first axis."""
    mu = np.zeros(d)
    mu[0] = float(pos)
    return HFN(mu=mu, sigma=np.ones(d) * sigma, id=name, use_diag=True)


def build_forest(tmp: Path) -> tuple[TieredForest, HFN, HFN, HFN, HFN, HFN]:
    """Build forest with known chain A→B→C and extra nodes E (inconsistent) and D (unknown)."""
    forest = TieredForest(D=D, cold_dir=tmp / "cold")
    A = make_node("A", 1)
    B = make_node("B", 2)
    C = make_node("C", 3)
    E = make_node("E", 3.1)   # near C but different chain
    D_node = make_node("D", 8.0)  # far from known chain

    # Build known chain A→B→C
    B.add_edge(B, A, "PART_OF")
    C.add_edge(C, B, "PART_OF")

    # E has its own chain (irrelevant to A→B)
    X = make_node("X", 5)
    E.add_edge(E, X, "PART_OF")

    for n in [A, B, C, E, X]:
        forest.register(n)
    # D is NOT registered — it's the unknown

    return forest, A, B, C, D_node, E


@dataclass
class TrialResult:
    """Result of a single constraint violation trial."""
    name: str
    decode_outcome: str
    resolution_mu: Optional[np.ndarray]
    topo_score: float
    residual_surprise: float
    surprising_leaves: int
    explanation_tree_size: int
    nodes_before: int
    nodes_after: int
    node_created: bool
    new_node_mu: Optional[np.ndarray]


def run_trial(name: str, query: HFN, obs: Observer, dec: Decoder, forest: TieredForest) -> TrialResult:
    """Run a single trial: observe, decode, measure results."""
    nodes_before = len(list(forest.active_nodes()))

    # Observation step
    obs_result = obs.observe(query.mu)
    nodes_after = len(list(forest.active_nodes()))

    # Decode step
    decode_result = dec.decode(query)
    if isinstance(decode_result, ResolutionRequest):
        outcome = "resolution_request"
        resolution_mu = decode_result.missing_mu.copy()
        topo_score = -999.0
    elif decode_result:
        outcome = "success"
        resolution_mu = None
        topo_score = dec._score_topological_fit(query, decode_result[0])
    else:
        outcome = "empty"
        resolution_mu = None
        topo_score = -999.0

    node_created = nodes_after > nodes_before
    new_node_mu = None
    if node_created:
        # Find the newly created node
        old_ids = {n.id for n in list(forest.active_nodes())[:nodes_before]}
        for n in forest.active_nodes():
            if n.id not in old_ids:
                new_node_mu = n.mu.copy()
                break

    return TrialResult(
        name=name,
        decode_outcome=outcome,
        resolution_mu=resolution_mu,
        topo_score=topo_score,
        residual_surprise=obs_result.residual_surprise,
        surprising_leaves=len(obs_result.surprising_leaves),
        explanation_tree_size=len(obs_result.explanation_tree),
        nodes_before=nodes_before,
        nodes_after=nodes_after,
        node_created=node_created,
        new_node_mu=new_node_mu,
    )


def print_result(r: TrialResult):
    """Pretty-print trial result."""
    print(f"\n{'='*60}")
    print(f"Trial: {r.name}")
    print(f"  Decode:           {r.decode_outcome}")
    print(f"  Topo score:       {r.topo_score:.3f}")
    print(f"  Residual surpr:   {r.residual_surprise:.3f}")
    print(f"  Surprising leaves:{r.surprising_leaves}")
    print(f"  Explanation tree: {r.explanation_tree_size}")
    print(f"  Nodes before/after: {r.nodes_before} → {r.nodes_after}")
    print(f"  Node created:     {r.node_created}")
    if r.new_node_mu is not None:
        print(f"  New node mu[0:4]: {r.new_node_mu[:4]}")
    if r.resolution_mu is not None:
        print(f"  Missing mu[0:4]:  {r.resolution_mu[:4]}")


def main():
    """Run the constraint violation learning experiment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        forest, A, B, C, D_node, E = build_forest(tmp)
        tau = calibrate_tau(D, sigma_scale=0.5, margin=2.0)
        obs = Observer(forest=forest, tau=tau, node_use_diag=True, budget=8)
        dec = Decoder(target_forest=forest)

        # T1: Clean match — A→B→C query
        q_clean = make_node("q_clean", 3, sigma=2.0)
        q_clean.add_edge(q_clean, B, "PART_OF")
        r1 = run_trial("T1: Clean (A→B→C)", q_clean, obs, dec, forest)

        # T2: Unknown terminus — query for D (not in forest)
        q_unknown = make_node("q_unknown", 8.0, sigma=2.0)
        q_unknown.add_edge(q_unknown, B, "PART_OF")
        q_unknown.add_edge(q_unknown, D_node, "PART_OF")
        r2 = run_trial("T2: Unknown (A→B→D)", q_unknown, obs, dec, forest)

        # T3: Inconsistent — near C but wrong chain
        q_incon = make_node("q_incon", 3.05, sigma=2.0)
        q_incon.add_edge(q_incon, B, "PART_OF")
        Z = make_node("Z", 9)  # Z not in forest
        q_incon.add_edge(q_incon, Z, "MUST_SATISFY")
        r3 = run_trial("T3: Inconsistent (topology mismatch)", q_incon, obs, dec, forest)

        for r in [r1, r2, r3]:
            print_result(r)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"  T1 decode: {r1.decode_outcome}  (expected: success)")
        print(f"  T2 decode: {r2.decode_outcome}  (expected: resolution_request)")
        print(f"  T3 decode: {r3.decode_outcome}  (expected: resolution_request)")
        print(f"  T1 node created: {r1.node_created}  (expected: False — already known)")
        print(f"  T2 node created: {r2.node_created}  (expected: True — novel position)")

        t1_pass = r1.decode_outcome == "success"
        t2_pass = r2.decode_outcome == "resolution_request"
        t3_pass = r3.decode_outcome == "resolution_request"

        print(f"\n  T1 PASS: {t1_pass}")
        print(f"  T2 PASS: {t2_pass}")
        print(f"  T3 PASS: {t3_pass}")

        overall = t1_pass and t2_pass and t3_pass
        print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
        return 0 if overall else 1


if __name__ == "__main__":
    exit(main())
