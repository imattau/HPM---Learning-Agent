"""
Experiment: Gradient of Difficulty (Stretch Curve)

Maps the learning boundary of the HFN system by creating a gradient from
identical patterns to structural breaks. Shows where the system fails:
- Geometry failure: Gaussian coverage boundary
- Topology failure: Decoder constraint limit
- Unknown failure: Structural gap detection

Six difficulty levels probe the exact frontier.
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


def make_node(name: str, pos: float, d: int = D, sigma: float = 1e-4) -> HFN:
    """Create a test node at position pos along first axis."""
    mu = np.zeros(d)
    mu[0] = float(pos)
    return HFN(mu=mu, sigma=np.ones(d) * sigma, id=name, use_diag=True)


def build_forest(tmp: Path) -> tuple[TieredForest, HFN, HFN, HFN, HFN]:
    """Build forest with known chain A→B→C and unknown D."""
    forest = TieredForest(D=D, cold_dir=tmp / "cold")
    A = make_node("A", 1.0)
    B = make_node("B", 2.0)
    C = make_node("C", 3.0)
    D_unknown = make_node("D", 8.0)  # far from known chain, not registered

    # Build known chain A→B→C
    B.add_edge(B, A, "PART_OF")
    C.add_edge(C, B, "PART_OF")

    for n in [A, B, C]:
        forest.register(n)

    return forest, A, B, C, D_unknown


@dataclass
class DifficultyTrialResult:
    """Result of a single difficulty level trial."""
    level: int
    name: str
    decode_outcome: str               # "success" | "resolution_request" | "empty"
    topo_score: float
    residual_surprise: float
    surprising_leaves: int
    explanation_tree_size: int
    node_created: bool
    nearest_node_dist: float
    failure_mode: str                 # "none" | "geometry" | "topology" | "unknown"


def classify_failure(result: DifficultyTrialResult) -> str:
    """Classify why a trial failed."""
    if result.decode_outcome == "success":
        return "none"
    elif result.decode_outcome == "resolution_request":
        # Edge to unknown target → structural gap
        if result.name.count("D_unknown") > 0 or result.name.count("unknown") > 0:
            return "unknown"
        # Topological mismatch → topology constraint issue
        if result.topo_score < 0.0:
            return "topology"
        # Fallback: likely geometry
        return "geometry"
    else:  # empty
        return "geometry"


def run_trial(name: str, query: HFN, obs: Observer, dec: Decoder, forest: TieredForest, level: int) -> DifficultyTrialResult:
    """Run a single trial at a given difficulty level."""
    nodes_before = len(list(forest.active_nodes()))

    # Observation step
    obs_result = obs.observe(query.mu)
    nodes_after = len(list(forest.active_nodes()))

    # Nearest node distance (geometric proximity)
    nodes_in_forest = list(forest.active_nodes())
    if nodes_in_forest:
        dists = [np.linalg.norm(n.mu - query.mu) for n in nodes_in_forest]
        nearest_dist = float(np.min(dists))
    else:
        nearest_dist = float('inf')

    # Decode step
    decode_result = dec.decode(query)
    if isinstance(decode_result, ResolutionRequest):
        outcome = "resolution_request"
        topo_score = -999.0
    elif decode_result:
        outcome = "success"
        topo_score = dec._score_topological_fit(query, decode_result[0])
    else:
        outcome = "empty"
        topo_score = -999.0

    node_created = nodes_after > nodes_before

    result = DifficultyTrialResult(
        level=level,
        name=name,
        decode_outcome=outcome,
        topo_score=topo_score,
        residual_surprise=obs_result.residual_surprise,
        surprising_leaves=len(obs_result.surprising_leaves),
        explanation_tree_size=len(obs_result.explanation_tree),
        node_created=node_created,
        nearest_node_dist=nearest_dist,
        failure_mode="",  # set by classify_failure
    )
    result.failure_mode = classify_failure(result)
    return result


def print_result(r: DifficultyTrialResult):
    """Print trial result in compact form."""
    print(
        f"L{r.level} {r.name:25s} | "
        f"outcome={r.decode_outcome:18s} | "
        f"topo={r.topo_score:7.2f} | "
        f"surpr={r.residual_surprise:7.2f} | "
        f"dist={r.nearest_node_dist:6.2f} | "
        f"fail={r.failure_mode:8s}"
    )


def main():
    """Run the gradient of difficulty experiment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        forest, A, B, C, D_unknown = build_forest(tmp)
        tau = calibrate_tau(D, sigma_scale=0.5, margin=2.0)
        obs = Observer(forest=forest, tau=tau, node_use_diag=True, budget=8)
        dec = Decoder(target_forest=forest)

        results = []

        # L0: Identical — exact match to C (position 3.0)
        q_l0 = make_node("q_l0", 3.0, sigma=2.0)
        q_l0.add_edge(q_l0, B, "PART_OF")
        r0 = run_trial("Identical (pos=3.0)", q_l0, obs, dec, forest, 0)
        results.append(r0)

        # L1: Small shift — within Gaussian coverage (position 3.2)
        q_l1 = make_node("q_l1", 3.2, sigma=2.0)
        q_l1.add_edge(q_l1, B, "PART_OF")
        r1 = run_trial("Small shift (pos=3.2)", q_l1, obs, dec, forest, 1)
        results.append(r1)

        # L2: Medium shift — probes coverage boundary (position 4.5)
        q_l2 = make_node("q_l2", 4.5, sigma=2.0)
        q_l2.add_edge(q_l2, B, "PART_OF")
        r2 = run_trial("Medium shift (pos=4.5)", q_l2, obs, dec, forest, 2)
        results.append(r2)

        # L3: Relation change — MUST_SATISFY instead of PART_OF
        q_l3 = make_node("q_l3", 3.0, sigma=2.0)
        q_l3.add_edge(q_l3, B, "MUST_SATISFY")
        r3 = run_trial("Relation change", q_l3, obs, dec, forest, 3)
        results.append(r3)

        # L4: Structural break — edge to unknown node D
        q_l4 = make_node("q_l4", 3.0, sigma=2.0)
        q_l4.add_edge(q_l4, B, "PART_OF")
        q_l4.add_edge(q_l4, D_unknown, "PART_OF")
        r4 = run_trial("Structural break (edge to D_unknown)", q_l4, obs, dec, forest, 4)
        results.append(r4)

        # L5: Full unknown — novel position + unknown edge
        q_l5 = make_node("q_l5", 9.0, sigma=2.0)
        q_l5.add_edge(q_l5, D_unknown, "PART_OF")
        r5 = run_trial("Full unknown (pos=9.0, edge to D)", q_l5, obs, dec, forest, 5)
        results.append(r5)

        # Print stretch curve
        print("\n" + "="*120)
        print("STRETCH CURVE: Learning Boundary")
        print("="*120)
        for r in results:
            print_result(r)

        # Validation
        print("\n" + "="*120)
        print("VALIDATION")
        print("="*120)

        l0_pass = r0.decode_outcome == "success"
        l1_pass = r1.decode_outcome == "success"
        l2_pass = r2.decode_outcome in ("success", "resolution_request")  # boundary zone
        l3_pass = r3.decode_outcome == "success"
        l4_pass = r4.decode_outcome == "resolution_request"
        l5_pass = r5.decode_outcome == "resolution_request"

        print(f"L0 (identical):         {r0.decode_outcome:20s} → {'PASS' if l0_pass else 'FAIL'}")
        print(f"L1 (small shift):       {r1.decode_outcome:20s} → {'PASS' if l1_pass else 'FAIL'}")
        print(f"L2 (medium shift):      {r2.decode_outcome:20s} → {'PASS' if l2_pass else 'FAIL'} (boundary zone)")
        print(f"L3 (relation change):   {r3.decode_outcome:20s} → {'PASS' if l3_pass else 'FAIL'}")
        print(f"L4 (structural break):  {r4.decode_outcome:20s} → {'PASS' if l4_pass else 'FAIL'}")
        print(f"L5 (full unknown):      {r5.decode_outcome:20s} → {'PASS' if l5_pass else 'FAIL'}")

        overall = l0_pass and l1_pass and l2_pass and l3_pass and l4_pass and l5_pass
        print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")

        return 0 if overall else 1


if __name__ == "__main__":
    exit(main())
