import time
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict
from hfn.tiered_forest import TieredForest
from hfn.forest import Forest
from hfn.hfn import HFN
from hfn.observer import Observer
from hfn.evaluator import Evaluator
from hfn.density import PatternDensityTracker
from hfn.affective import AffectiveEvaluator, AffectiveState

# Import components from existing experiments
from hpm_fractal_node.experiments.experiment_unified_perception_action import (
    ASTRenderer, EmpiricalOracle, PythonExecutor,
    CONCEPTS, CONCEPT_IDX, S_DIM, DIM,
    extract_perceptual_ops,
)
from hpm_fractal_node.experiments.experiment_meta_strategy_controller import (
    MetaStrategyController, CountingOracle, SolveRecord,
    _register_macro_with_transitions,
)
from hpm_fractal_node.experiments.experiment_cross_domain_analogy import (
    DomainTransferBridge, LearnabilityProbe, _render_path
)
from hpm_fractal_node.experiments.experiment_cross_domain_analogy_enhanced import (
    EnhancedAnalogicalAgent, ASTMacroSubstitutor
)
from hpm_fractal_node.experiments.experiment_generative_forward_model import StateTransitionModel

class SocialForest(TieredForest):
    """
    Shared repository for HFN nodes across agents with fractal-consistent institutional scaffolding.
    """
    def __init__(self, D: int, cold_dir: Path | str, hot_cap: int = 100):
        super().__init__(D, cold_dir, hot_cap=hot_cap)
        # Social state stored as HFN nodes in dedicated tiered forests
        self._exchange_forest = TieredForest(D=4, cold_dir=Path(cold_dir) / "exchanges", hot_cap=100)
        self._blackboard_forest = TieredForest(D=4, cold_dir=Path(cold_dir) / "blackboard", hot_cap=100)

    def share_node(self, node: HFN, sender_id: str):
        """Register node in shared forest and log exchange as HFN node."""
        self.register(node)
        ts = time.time()
        mu = np.array([float(hash(sender_id) % 10000), ts, float(hash(node.id) % 10000), 0.0])
        exch_node = HFN(mu=mu, sigma=np.ones(4), id=f"exch_{ts}", use_diag=True)
        self._exchange_forest.register(exch_node)

    def post_failure(self, task_id: str, rule_id: str, reason: str, timestamp: float):
        """Post failure as HFN node: mu = [task_hash, rule_hash, 0.0, timestamp]."""
        mu = np.array([float(hash(task_id) % 10000), float(hash(rule_id) % 10000), 0.0, timestamp])
        fail_node = HFN(mu=mu, sigma=np.ones(4), id=f"fail_{timestamp}", use_diag=True)
        fail_node._reason = reason # metadata
        self._blackboard_forest.register(fail_node)

    def get_failures(self, task_id: str) -> List[Dict]:
        """Query blackboard for failure nodes matching task_id hash."""
        task_hash = float(hash(task_id) % 10000)
        failures = []
        for node in self._blackboard_forest.active_nodes():
            if np.isclose(node.mu[0], task_hash):
                failures.append({
                    "rule": str(int(node.mu[1])), 
                    "reason": getattr(node, '_reason', "Unknown"),
                    "time": node.mu[3]
                })
        return failures

class RecombinationEngine:
    """
    Explicit structural recombination (HPM Appendix E).
    Uses the core HFN Recombination tool for structural merging.
    """
    def __init__(self):
        from hfn.recombination import Recombination
        self._core = Recombination()

    def recombine(self, macro_a: HFN, macro_b: HFN, forest: Forest) -> HFN:
        """Create a new macro by concatenating constituents of two macros."""
        return self._core.recombine_structural(macro_a, macro_b, forest)

    def insight_score(self, new_macro: HFN, existing_macros: list[HFN], 
                      test_inputs: list, test_outputs: list, executor) -> float:
        # Novelty: Normalized min distance to existing macros
        distances = [np.linalg.norm(new_macro.mu - m.mu) for m in existing_macros if m.id != new_macro.id]
        min_dist = min(distances) if distances else 1.0
        novelty = min_dist / (1.0 + min_dist) 
        
        # Effectiveness: accuracy on test tasks
        correct = 0
        total = len(test_inputs)
        if total == 0:
            effectiveness = 0.0
        else:
            for inp, out in zip(test_inputs, test_outputs):
                try:
                    pred = executor(new_macro, inp)
                    if pred == out:
                        correct += 1
                except:
                    continue
            effectiveness = correct / total
            
        return 0.5 * novelty + 0.5 * effectiveness

class SocialAnalogicalAgent(EnhancedAnalogicalAgent):
    """
    Revised Social Agent with Full HPM Stack (L1-L5).
    """
    def __init__(self, agent_id: str, shared_forest: SocialForest, partner=None):
        self.agent_id = agent_id
        self.partner = partner
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = shared_forest
        self.experiment_id = f"social_{agent_id}"
        
        from hfn.retriever import HybridRetriever
        self.retriever = HybridRetriever(self.forest, geometric_weight=0.5, structural_weight=0.5)
        
        self.observer = Observer(
            forest=self.forest, 
            retriever=self.retriever, 
            tau=0.5, 
            node_use_diag=True, 
            compression_cooccurrence_threshold=2
        )
        self.renderer = ASTRenderer()
        self.oracle = CountingOracle()
        self.executor = PythonExecutor()
        self.meta = MetaStrategyController()
        self.solve_history: List[SolveRecord] = []
        
        self.forward_model = StateTransitionModel()
        self._oracle_calls_imagination = 0
        
        self.perceptual_ops, self.perceptual_sources = extract_perceptual_ops(self.m_dim)
        self._inject_blank_priors() # Ensure priors are injected into shared forest
        for op in self.perceptual_ops:
            if op.id not in self.forest:
                self.observer.register(op, protected=True, initial_weight=0.5)
        
        self.macro_nodes: dict[str, HFN] = {}
        self._state_dir = Path(tempfile.mkdtemp(prefix=f"hpm_state_{agent_id}_"))
        self.density_tracker = PatternDensityTracker(self.observer, cold_dir=self._state_dir / "density")
        self.affective = AffectiveEvaluator(cold_dir=self._state_dir / "affective")
        self.ast_sub = ASTMacroSubstitutor()
        
        self.domain_ops_b: List[Dict] = []
        self._domain_b_training_count = 0
        self.bridge = self._create_enhanced_bridge()
        self.recombiner = RecombinationEngine()

    def should_share(self, macro: HFN, probe_input: Any) -> bool:
        """L4 Mental Simulation: Only share if forward model accurately predicts behaviour."""
        if not macro.inputs: return False
        baseline_outs, baseline_errs = self.executor.run_batch("pass", [probe_input])
        start_state = self.oracle.compute_state(baseline_outs, baseline_errs, "pass")
        predicted_state = self.forward_model.predict_path(start_state, list(macro.inputs))
        code = self.renderer.render(macro)
        outs, errs = self.executor.run_batch(code, [probe_input])
        actual_state = self.oracle.compute_state(outs, errs, code)
        
        # Compare only outcome features (0-10) to avoid binary flag summation issues
        min_len = min(10, len(predicted_state), len(actual_state))
        diff = predicted_state[:min_len] - actual_state[:min_len]
        error = np.linalg.norm(diff)
        return error < 0.1

    def exchange_patterns(self, probe_input: Any = None):
        """Exchange high-density, simulated-accurate macros with partner."""
        if not self.partner: return 0
        shared_count = 0
        for name, macro in list(self.macro_nodes.items()):
            dens = self.density_tracker.get_density(macro.id)
            if dens and dens[3] > 0.5:
                if probe_input is not None and not self.should_share(macro, probe_input):
                    continue
                self.forest.share_node(macro, self.agent_id)
                self.partner.receive_pattern(macro, self.agent_id)
                shared_count += 1
        return shared_count

    def receive_pattern(self, node: HFN, sender_id: str):
        """Incorporate a pattern received from another agent."""
        if node.id not in self.macro_nodes:
            name = node.id.replace("macro_", "")
            self.macro_nodes[name] = node
            if node.id not in self.forest:
                self.observer.register(node, protected=False, initial_weight=0.5)
            if node.inputs:
                self._record_transitions(list(node.inputs), [[1,2,3]])

    def try_recombination(self, task_inputs: List, task_outputs: List) -> Optional[Tuple[str, int]]:
        macros = list(self.macro_nodes.values())
        best_code, best_insight = None, -1.0
        def test_executor(macro, inp):
            code = self.renderer.render(macro)
            outs, _ = self.executor.run_batch(code, [inp])
            return outs[0] if outs else None
        for i, ma in enumerate(macros):
            for mb in macros[i+1:]:
                recomb = self.recombiner.recombine(ma, mb, self.forest)
                score = self.recombiner.insight_score(recomb, macros, task_inputs, task_outputs, test_executor)
                if score > 0.3 and score > best_insight:
                    best_insight = score
                    best_code = self.renderer.render(recomb)
                    self.affective.update_from_outcome(recomb.id, success=True, surprise=0.8)
                    name = recomb.id.replace("recomb_", "")
                    self.macro_nodes[name] = recomb
                    self.observer.register(recomb, protected=False, initial_weight=1.0)
        return (best_code, 3) if best_code else None

    def solve_with_social(self, task_id: str, inputs: List, outputs: List):
        code, rec = self.solve_with_affective_monitoring(task_id, inputs, outputs)
        if not rec.success:
            result = self.try_recombination(inputs, outputs)
            if result:
                code, depth = result
                rec = SolveRecord(task_id=task_id, goal_type="recombination", n_macros=len(self.macro_nodes), strategy="recombination", depth=depth, oracle_calls=3, success=True, wall_ms=0.0)
                self.meta.record(rec); self.solve_history.append(rec)
        if not rec.success:
            self.forest.post_failure(task_id, "failed", "No solution found", time.time())
        return code, rec

    def discover_meta_schema(self) -> Optional[HFN]:
        from hpm_fractal_node.experiments.experiment_induced_schema_library import InducedSchemaAgent
        temp_agent = InducedSchemaAgent()
        temp_agent.macro_nodes = self.macro_nodes
        temp_agent.forest = self.forest
        temp_agent.observer = self.observer
        temp_agent.perceptual_ops = self.perceptual_ops
        return temp_agent.discover_meta_schema()
