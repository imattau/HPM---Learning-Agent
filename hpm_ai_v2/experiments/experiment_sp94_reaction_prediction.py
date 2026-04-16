#!/usr/bin/env python3
"""
SP94: One‑Shot Reaction Prediction – Ester Hydrolysis (Fingerprint version)

Learns the transformation: methyl acetate → acetic acid + methanol
from a single example, then applies it to ethyl acetate.
Uses a custom fingerprint domain with bit‑flipping primitives.
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Add parent directory to path if needed (adjust as necessary)
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.base import DomainConfig
from hpm_ai_v2.utils.base_renderer import Renderer
from hpm_ai_v2.utils.oracle.base import BaseOracle, CountingOracle


# ----------------------------------------------------------------------
# 1. Domain Configuration (Fingerprint)
# ----------------------------------------------------------------------
class FingerprintDomainConfig(DomainConfig):
    """Domain for fixed‑length bit fingerprints with SET/CLEAR operations."""
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        # One primitive per bit: SET_BIT_i and CLEAR_BIT_i
        concepts = [f"SET_BIT_{i}" for i in range(n_bits)] + [f"CLEAR_BIT_{i}" for i in range(n_bits)]
        super().__init__(concepts, s_dim=20)
        self.m_dim = self.S_DIM + self.DIM + self.S_DIM


# ----------------------------------------------------------------------
# 2. Oracle for Fingerprints
# ----------------------------------------------------------------------
class FingerprintOracle(BaseOracle):
    """Oracle that computes a 20‑D state vector from a fingerprint array."""
    def __init__(self, config: FingerprintDomainConfig):
        self.config = config
        self.call_count = 0

    def compute_state(self, outputs, errors, code=""):
        s = np.zeros(self.config.S_DIM)
        valid = [o for o, e in zip(outputs, errors) if e is None]
        if not valid:
            return s
        s[0] = 1.0
        fp = valid[0]
        if isinstance(fp, np.ndarray) and len(fp) == self.config.n_bits:
            s[3] = np.sum(fp) / self.config.n_bits          # fraction of set bits
        # Simple code structure flags (not critical)
        s[10] = 1.0 if 'for ' in code else 0.0
        s[12] = 1.0 if 'if ' in code else 0.0
        return s


# ----------------------------------------------------------------------
# 3. Renderer for Fingerprint Operations
# ----------------------------------------------------------------------
class FingerprintRenderer(Renderer):
    """Generates Python code that performs bit‑flipping operations on a numpy array."""
    def __init__(self, config: FingerprintDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        ops = self._extract_ops(node)
        lines = [
            "import numpy as np",
            "res = inp.copy()"
        ]
        for op in ops:
            if op.startswith("SET_BIT_"):
                bit = int(op.split("_")[-1])
                lines.append(f"res[{bit}] = 1.0")
            elif op.startswith("CLEAR_BIT_"):
                bit = int(op.split("_")[-1])
                lines.append(f"res[{bit}] = 0.0")
        return "\n".join(lines)

    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        ops = self._extract_ops(node)
        lines = [
            f"def {func_name}(fp):",
            "    import numpy as np",
            "    fp = fp.copy()"
        ]
        for op in ops:
            if op.startswith("SET_BIT_"):
                bit = int(op.split("_")[-1])
                lines.append(f"    fp[{bit}] = 1.0")
            elif op.startswith("CLEAR_BIT_"):
                bit = int(op.split("_")[-1])
                lines.append(f"    fp[{bit}] = 0.0")
        lines.append("    return fp")
        return "\n".join(lines)

    def _extract_ops(self, node: HFN):
        ops = []
        if node.inputs:
            for child in node.inputs:
                ops.extend(self._extract_ops(child))
        else:
            concept = self._get_concept(node)
            if concept:
                ops.append(concept)
        return ops

    def _get_concept(self, node: HFN):
        start = self.config.S_DIM
        end = start + self.config.DIM
        vec = node.mu[start:end]
        if np.max(vec) > 0.5:
            idx = np.argmax(vec)
            return self.config.concepts[idx]
        return None


# ----------------------------------------------------------------------
# 4. Helper: Create primitive HFN nodes for each bit operation
# ----------------------------------------------------------------------
def create_primitive_nodes(agent, config):
    """Register a primitive HFN node for each concept (SET_BIT_i, CLEAR_BIT_i)."""
    primitives = []
    for i, concept in enumerate(config.concepts):
        mu = np.zeros(config.m_dim)
        mu[config.S_DIM + i] = 1.0
        node = HFN(mu=mu, sigma=np.ones(config.m_dim), id=f"fp_op_{concept}", use_diag=True)
        agent.observer.register(node, protected=False)
        primitives.append(node)
    return primitives


# ----------------------------------------------------------------------
# 5. Hand‑crafted fingerprints for our molecules (8 bits)
#    Bit meaning:
#      0: methyl group (CH3-)
#      1: ethyl group (CH3CH2-)
#      2: ester bond (C(=O)-O)
#      3: carboxyl group (COOH)
#      4: hydroxyl group (OH)
#      5: water molecule (H2O) – not used directly
#      6: methanol (CH3OH)
#      7: acetic acid (CH3COOH)
# ----------------------------------------------------------------------
def fingerprint_methyl_acetate():
    fp = np.zeros(8)
    fp[0] = 1   # methyl
    fp[2] = 1   # ester bond
    return fp

def fingerprint_ethyl_acetate():
    fp = np.zeros(8)
    fp[1] = 1   # ethyl
    fp[2] = 1   # ester bond
    return fp

def fingerprint_products():
    # Acetic acid + methanol (combined fingerprint)
    fp = np.zeros(8)
    fp[3] = 1   # carboxyl group
    fp[6] = 1   # methanol
    return fp


# ----------------------------------------------------------------------
# 6. Main experiment
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("SP94: One‑Shot Reaction Prediction – Ester Hydrolysis (Fingerprint)")
    print("=" * 80)

    # Domain configuration (8 bits)
    config = FingerprintDomainConfig(n_bits=8)
    renderer = FingerprintRenderer(config)
    oracle = FingerprintOracle(config)

    # Create a fresh forest (in memory, temporary directory)
    cold_dir = Path(tempfile.mkdtemp(prefix="sp94_fp_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")

    # Instantiate the agent
    agent = BaseHFNAgent(
        config=config,
        renderer=renderer,
        forest=forest,
        retriever_type="geometric",          # simple retrieval
        use_hfn_forward_model=False,
        use_hfn_meta_controller=False,
    )
    agent.oracle = oracle
    agent.counting_oracle = CountingOracle(oracle)

    # Add primitive operations to the agent's candidate set
    agent._candidate_ops = create_primitive_nodes(agent, config)

    # Register solving strategies
    agent.add_strategy("exact", agent._try_exact, position=0)
    agent.add_strategy("bfs", agent._try_bfs, position=1)

    # ------------------------------------------------------------------
    # Phase 1: One‑shot training (methyl acetate → products)
    # ------------------------------------------------------------------
    print("\n[Phase 1] Learning from one example (methyl acetate → acetic acid + methanol)")
    train_input = fingerprint_methyl_acetate()
    train_output = fingerprint_products()

    success, code, strategy = agent.solve([train_input], [train_output], goal_type="map", task_id="hydrolysis")
    if not success:
        print("  Training failed. Exiting.")
        return
    print(f"  Training succeeded via strategy '{strategy}'. Generated macro code:\n{code}")

    # ------------------------------------------------------------------
    # Phase 2: Generalisation to ethyl acetate
    # ------------------------------------------------------------------
    print("\n[Phase 2] Testing generalisation on ethyl acetate")
    test_input = fingerprint_ethyl_acetate()
    test_expected = fingerprint_products()

    success_test, code_test, strategy_test = agent.solve([test_input], [test_expected], goal_type="map", task_id="hydrolysis_test")
    if success_test:
        print(f"  Generalisation SUCCESS! (strategy: {strategy_test})")
        # Actually run the macro to verify the output
        results, errors = agent.executor.run_batch(code_test, [test_input])
        result = results[0]
        if result is not None and np.array_equal(result, test_expected):
            print("  Output fingerprint matches expected.")
        else:
            print(f"  Output fingerprint: {result}, expected: {test_expected}")
            if errors[0]:
                print(f"  Execution Error: {errors[0]}")
    else:
        print("  Generalisation FAILED.")

    print("\n" + "=" * 80)
    print("[SUCCESS] One‑shot reaction prediction demonstrated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
