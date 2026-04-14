"""
BaseHFNAgent — foundation for all hpm_ai_v2 agents.

Provides:
- TieredForest + Observer initialisation
- PythonExecutor + EmpiricalOracle/CountingOracle
- Generic pattern storage (self.patterns dict)
- BFS search (_try_bfs, _try_exact)
- Strategy dispatch (add_strategy, solve)
- MetaStrategyController integration
- Persistence (save_state, load_state)
"""
from __future__ import annotations

import os
import pickle
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import numpy as np

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever, HybridRetriever, StructuralRetriever, GeometricRetriever

from hpm_ai_v2.utils.executor import PythonExecutor, _eval_path_worker
from hpm_ai_v2.utils.oracle import EmpiricalOracle, CountingOracle
from hpm_ai_v2.utils.renderer import ASTRenderer
from hpm_ai_v2.utils.base_renderer import Renderer
from hpm_ai_v2.utils.meta_controller import MetaStrategyController, SolveRecord

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig


class BaseHFNAgent:
    """
    Base class for all HPM agents.

    Subclasses (and mixins) extend this with additional capabilities at each
    HPM abstraction level (L2 macro composition, L3 relational schemas,
    L4 forward models, L5 meta-strategy, social sharing).
    """

    def __init__(
        self,
        config: "DomainConfig",
        cold_dir: str = "data/knowledge_base/hpm_ai_v2",
        forest_class: Type = TieredForest,
        hot_cap: int = 10_000,
        tau: float = 0.5,
        compression_cooccurrence_threshold: int = 2,
        use_density_tracker: bool = False,
        use_affective_evaluator: bool = False,
        n_workers: Optional[int] = None,
        renderer: Optional[Renderer] = None,
        retriever_type: str = "goal_conditioned",
        auto_observe_frequency: int = 0,
        replay_buffer_size: int = 100,
        auto_save_frequency: int = 0,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.s_dim = config.S_DIM
        self.dim = config.DIM
        self.m_dim = config.m_dim
        self.cold_dir = Path(cold_dir)

        # Pattern substrate: TieredForest
        self.forest: TieredForest = forest_class(
            D=self.m_dim,
            cold_dir=self.cold_dir,
            hot_cap=hot_cap,
        )

        # Retriever
        target_slice = slice(self.s_dim + self.dim, self.m_dim)
        if retriever_type == "hybrid":
            self.retriever = HybridRetriever(self.forest)
        elif retriever_type == "structural":
            self.retriever = StructuralRetriever(self.forest)
        elif retriever_type == "geometric":
            self.retriever = GeometricRetriever(self.forest)
        else:
            # Default to goal-conditioned for backward compatibility
            self.retriever = GoalConditionedRetriever(
                self.forest,
                target_slice=target_slice,
                target_weight=50.0,
            )

        # Pattern dynamics: Observer
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            tau=tau,
            node_use_diag=True,
            compression_cooccurrence_threshold=compression_cooccurrence_threshold,
            use_density_tracker=use_density_tracker,
            use_affective_evaluator=use_affective_evaluator,
        )
        
        # Link retriever to observer's weight provider if applicable
        if hasattr(self.retriever, 'weight_provider'):
            self.retriever.weight_provider = lambda nid: self.observer.get_weight(nid)

        # Parallelism
        self.n_workers: int = n_workers if n_workers is not None else os.cpu_count() or 1

        # Utils
        self.renderer = renderer if renderer is not None else ASTRenderer(config)
        self.oracle = EmpiricalOracle(config)
        self.counting_oracle = CountingOracle(config)
        self.executor = PythonExecutor()

        # Pattern evaluator: MetaStrategyController (L5)
        self.meta = MetaStrategyController()

        # Generic pattern storage: pattern_id -> HFN node
        self.patterns: Dict[str, HFN] = {}

        # Strategy registry: name -> callable(inputs, outputs) -> Optional[List[HFN]]
        self._strategies: Dict[str, Callable] = {}
        self._strategy_order: List[str] = []

        # Lifecycle and persistence
        self.auto_observe_frequency = auto_observe_frequency
        self._observe_counter = 0
        self._replay_buffer: List[np.ndarray] = []
        self._replay_buffer_size = replay_buffer_size
        self.auto_save_frequency = auto_save_frequency
        self._solve_counter = 0

        # Inject priors if forest is empty
        if len(self.forest) == 0:
            self._inject_blank_priors()

    # ------------------------------------------------------------------
    # Prior injection
    # ------------------------------------------------------------------

    def _inject_blank_priors(self) -> None:
        """Inject one prior node per concept into the forest."""
        list_concepts = {"LIST_INIT", "FOR_LOOP", "ITEM_ACCESS", "LIST_APPEND"}
        delta_offset = self.s_dim + self.dim
        for i, c in enumerate(self.config.concepts):
            mu = np.zeros(self.m_dim)
            mu[self.s_dim + i] = 5.0
            if c in list_concepts:
                mu[delta_offset + 1] = 1.0
            node = HFN(
                mu=mu,
                sigma=np.ones(self.m_dim) * 5.0,
                id=f"prior_rule_{c}",
                use_diag=True,
            )
            self.observer.register(node, protected=True, initial_weight=0.5)

    # ------------------------------------------------------------------
    # Strategy registry
    # ------------------------------------------------------------------

    def add_strategy(
        self,
        name: str,
        fn: Callable[[List[Any], List[Any]], Optional[List[HFN]]],
        position: Optional[int] = None,
    ) -> None:
        """Register a solve strategy callable."""
        self._strategies[name] = fn
        if name not in self._strategy_order:
            if position is not None:
                self._strategy_order.insert(position, name)
            else:
                self._strategy_order.append(name)

    def _ordered_strategies(
        self,
        goal_type: str = "scalar",
    ) -> List[Tuple[str, Callable]]:
        """Return strategies in meta-controller ranked order."""
        n_macros = sum(
            1 for n in self.patterns.values()
            if n.relation_type == "macro"
        )
        ranked = self.meta.rank_strategies(goal_type, n_macros)
        order = [s for s in ranked if s in self._strategies]
        for s in self._strategy_order:
            if s not in order:
                order.append(s)
        return [(name, self._strategies[name]) for name in order if name in self._strategies]

    def select_next_task(
        self,
        task_pool: List[Tuple[str, str, List[Any], List[Any]]],
        learnability_dict: Dict[str, float],
    ) -> Tuple[str, str, List[Any], List[Any]]:
        """
        Active Learning: Select next task from pool using affective curiosity.
        Requires use_affective_evaluator=True.
        """
        if not hasattr(self.observer, 'evaluator') or not hasattr(self.observer.evaluator, 'curiosity_exploration_probability'):
            # Fallback to random if no affective evaluator
            idx = np.random.randint(len(task_pool))
            return task_pool[idx]

        probs = []
        for task_id, _, _, _ in task_pool:
            learnability = learnability_dict.get(task_id, 0.5)
            # Higher curiosity for tasks with mid-range learnability
            prob = self.observer.evaluator.curiosity_exploration_probability(
                learnability
            )
            probs.append(max(1e-6, float(prob)))

        probs = np.array(probs) / sum(probs)
        chosen_idx = np.random.choice(len(task_pool), p=probs)
        return task_pool[chosen_idx]

    # ------------------------------------------------------------------
    # Lifecycle and persistence
    # ------------------------------------------------------------------

    def register_pattern(self, task_id: str, path: List[HFN]) -> None:
        """Register a successful solution path as a reusable pattern."""
        if not path:
            return
        if len(path) == 1:
            self.patterns[task_id] = path[0]
            if path[0].id not in self.forest:
                self.forest.register(path[0])
        else:
            composed = self._compose_sequence(path)
            if composed:
                self.patterns[task_id] = composed
                if composed.id not in self.forest:
                    self.forest.register(composed)

    def _maybe_auto_observe(self, x: Optional[np.ndarray] = None) -> None:
        """Periodically call observer.observe() to drive compression/absorption."""
        if self.auto_observe_frequency <= 0:
            return
        self._observe_counter += 1
        if self._observe_counter >= self.auto_observe_frequency:
            self._observe_counter = 0
            if x is not None:
                self.observer.observe(x)
            elif self._replay_buffer:
                sample = random.choice(self._replay_buffer)
                self.observer.observe(sample)

    def observe_example(self, x: np.ndarray) -> None:
        """Explicitly observe an input vector (e.g., during training)."""
        self.observer.observe(x)
        self._replay_buffer.append(x)
        if len(self._replay_buffer) > self._replay_buffer_size:
            self._replay_buffer.pop(0)
        self._maybe_auto_observe()

    def _maybe_save_state(self) -> None:
        """Automatically save agent state periodically."""
        if self.auto_save_frequency <= 0:
            return
        self._solve_counter += 1
        if self._solve_counter >= self.auto_save_frequency:
            self._solve_counter = 0
            self.save_state()

    # ------------------------------------------------------------------
    # Core solve loop
    # ------------------------------------------------------------------

    def solve(
        self,
        inputs: List[Any],
        outputs: List[Any],
        goal_type: str = "scalar",
        task_id: str = "task",
    ) -> Tuple[bool, Optional[str], str]:
        """
        Attempt to solve task using registered strategies in ranked order.

        Returns (success, code_str, strategy_used).
        """
        t0 = time.time()
        n_macros = sum(1 for n in self.patterns.values() if n.relation_type == "macro")
        for strat_name, strat_fn in self._ordered_strategies(goal_type):
            oracle_calls_before = self.counting_oracle.call_count
            path = strat_fn(inputs, outputs)
            if path is not None:
                code = (
                    self.renderer.render(path[0])
                    if len(path) == 1
                    else self._render_path(path)
                )
                results, errors = self.executor.run_batch(code, inputs)
                state = self.oracle.compute_state(results, errors, code)
                success = bool(state[0] > 0.5 and self._check_outputs(results, outputs))
                oracle_calls = self.counting_oracle.call_count - oracle_calls_before + 1
                wall_ms = (time.time() - t0) * 1000
                rec = SolveRecord(
                    task_id=task_id,
                    goal_type=goal_type,
                    n_macros=n_macros,
                    strategy=strat_name,
                    depth=len(path),
                    oracle_calls=oracle_calls,
                    success=success,
                    wall_ms=wall_ms,
                )
                self.meta.record(rec)
                if success:
                    self.register_pattern(task_id, path)

                    # After successful solve, optionally observe the encoded input
                    if self.auto_observe_frequency > 0:
                        # Encode the first input as a vector (simplified flattening)
                        flat = np.array(inputs[0]).flatten()
                        if len(flat) < self.m_dim:
                            flat = np.pad(flat, (0, self.m_dim - len(flat)))
                        self._maybe_auto_observe(flat[:self.m_dim])

                    self._maybe_save_state()
                    return True, code, strat_name
        return False, None, "none"

    def _render_path(self, path: List[HFN]) -> str:
        """Render a multi-node path by composing into a sequence node."""
        composed = self._compose_sequence(path)
        if composed is None:
            return ""
        return self.renderer.render(composed)

    def _compose_sequence(self, nodes: List[HFN]) -> Optional[HFN]:
        """Compose a list of nodes into a single sequence macro node."""
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]
        mu = np.zeros(self.m_dim)
        first_state = nodes[0].mu[:self.s_dim]
        last_state = nodes[-1].mu[:self.s_dim]
        mu[:self.s_dim] = last_state
        mu[self.s_dim + self.dim:] = last_state - first_state
        action_sum = sum(n.mu[self.s_dim: self.s_dim + self.dim] for n in nodes)
        mu[self.s_dim: self.s_dim + self.dim] = action_sum / len(nodes)
        return HFN(
            mu=mu,
            sigma=np.ones(self.m_dim),
            inputs=nodes,
            relation_type="macro",
            use_diag=True,
        )

    # ------------------------------------------------------------------
    # Built-in strategies
    # ------------------------------------------------------------------

    def _try_exact(
        self,
        inputs: List[Any],
        outputs: List[Any],
    ) -> Optional[List[HFN]]:
        """Strategy: retrieve nearest node and test directly."""
        goal_state = self._outputs_to_goal_state(outputs)
        query = HFN(mu=goal_state, sigma=np.ones(self.m_dim), use_diag=True)
        candidates = self.retriever.retrieve(query, k=5)
        for node in candidates:
            code = self.renderer.render(node)
            results, errors = self.executor.run_batch(code, inputs)
            if self._check_outputs(results, outputs):
                return [node]
        return None

    def _try_bfs(
        self,
        inputs: List[Any],
        outputs: List[Any],
        max_depth: int = 4,
        beam_width: int = 10,
    ) -> Optional[List[HFN]]:
        """Strategy: beam BFS over pattern space, evaluating each depth level in parallel."""
        goal_state = self._outputs_to_goal_state(outputs)
        query = HFN(mu=goal_state, sigma=np.ones(self.m_dim), use_diag=True)

        if hasattr(self, "_candidate_ops") and self._candidate_ops:
            primitives = self._candidate_ops
        else:
            primitives = self.retriever.retrieve(query, k=beam_width)

        if not primitives:
            return None

        visited_ids: set = set()
        current_level: List[List[HFN]] = [[p] for p in primitives]

        for _depth in range(max_depth):
            # Deduplicate and render all paths at this depth level
            candidates: List[Tuple[str, List[HFN]]] = []
            for path in current_level:
                path_key = tuple(n.id for n in path)
                if path_key in visited_ids:
                    continue
                visited_ids.add(path_key)
                composed = self._compose_sequence(path)
                if composed is None:
                    continue
                code = self.renderer.render(composed)
                candidates.append((code, path))

            if not candidates:
                break

            # Evaluate all candidates at this level in parallel
            if self.n_workers > 1 and len(candidates) > 1:
                with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                    futures = {
                        pool.submit(_eval_path_worker, code, inputs, outputs): path
                        for code, path in candidates
                    }
                    for fut in as_completed(futures):
                        if fut.result():
                            # Cancel remaining futures (best-effort)
                            for f in futures:
                                f.cancel()
                            return futures[fut]
            else:
                for code, path in candidates:
                    results, _ = self.executor.run_batch(code, inputs)
                    if self._check_outputs(results, outputs):
                        return path

            if _depth + 1 >= max_depth:
                break

            # Expand next level
            next_nodes = self.retriever.retrieve(query, k=beam_width)
            next_level: List[List[HFN]] = []
            for path in current_level:
                for nxt in next_nodes:
                    new_path = path + [nxt]
                    new_key = tuple(n.id for n in new_path)
                    if new_key not in visited_ids:
                        next_level.append(new_path)
            current_level = next_level

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _outputs_to_goal_state(self, outputs: List[Any]) -> np.ndarray:
        """Build a goal state vector from expected outputs."""
        results_dummy = outputs
        errors_dummy = [None] * len(outputs)
        delta_state = self.oracle.compute_state(results_dummy, errors_dummy)
        goal = np.zeros(self.m_dim)
        goal[self.s_dim + self.dim:] = delta_state
        return goal

    def _check_outputs(
        self,
        results: List[Any],
        expected: List[Any],
    ) -> bool:
        """Check whether execution results match expected outputs."""
        if len(results) != len(expected):
            return False
        for r, e in zip(results, expected):
            if isinstance(r, np.ndarray) and isinstance(e, np.ndarray):
                # For arrays, use allclose to handle float precision
                if not np.allclose(r, e, atol=1e-4):
                    return False
            elif r != e:
                return False
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[str] = None) -> Path:
        """Serialise agent state (patterns + meta stats) to disk."""
        save_path = Path(path) if path else self.cold_dir / "agent_state.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "patterns": self.patterns,
            "meta_stats": dict(self.meta._stats),
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        return save_path

    def load_state(self, path: Optional[str] = None) -> None:
        """Load agent state from disk."""
        load_path = Path(path) if path else self.cold_dir / "agent_state.pkl"
        if not load_path.exists():
            return
        with open(load_path, "rb") as f:
            state = pickle.load(f)
        self.patterns = state.get("patterns", {})
        if "meta_stats" in state:
            self.meta._stats.update(state["meta_stats"])
