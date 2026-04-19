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
from hpm_ai_v2.utils.oracle import ListOracle, CountingOracle
from hpm_ai_v2.domains.list_renderer import ListRenderer
from hpm_ai_v2.utils.base_renderer import Renderer
from hpm_ai_v2.utils.meta_controller import MetaStrategyController, SolveRecord
from hpm_ai_v2.utils.hfn_meta_controller import HFNMetaStrategyController

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
        self.tolerance = kwargs.get("tolerance", 0.05)

        # Pattern substrate: TieredForest
        if "forest" in kwargs:
            self.forest = kwargs.pop("forest")
        else:
            self.forest: TieredForest = forest_class(
                D=self.m_dim,
                cold_dir=self.cold_dir,
                hot_cap=hot_cap,
            )

        # Retriever
        target_slice = slice(self.s_dim + self.dim, self.m_dim)
        base_retriever = None
        if retriever_type == "hybrid":
            base_retriever = HybridRetriever(self.forest)
        elif retriever_type == "structural":
            base_retriever = StructuralRetriever(self.forest)
        elif retriever_type == "geometric":
            base_retriever = GeometricRetriever(self.forest)
        elif retriever_type == "contextual":
            from hfn.retriever import ContextualRetriever
            base_retriever = ContextualRetriever(self.forest)
        else:
            # Default to goal-conditioned for backward compatibility
            base_retriever = GoalConditionedRetriever(
                self.forest,
                target_slice=target_slice,
                target_weight=50.0,
            )
        
        from hfn.retriever import MacroPrioritizingRetriever
        self.retriever = MacroPrioritizingRetriever(base_retriever)

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
        
        # Pattern dynamics: Decay/forgetting tracking (L2-L4)
        self._pattern_decay_times: Dict[str, float] = {}  # pattern_id -> last_decay_time
        self._decay_half_life: float = kwargs.get("decay_half_life", 1000.0)  # ms
        
        # Link retriever to observer's weight provider if applicable
        if hasattr(self.retriever, 'weight_provider'):
            self.retriever.weight_provider = lambda nid: self.observer.get_weight(nid)

        # Pattern evaluator: Usage/boredom tracking (L5 meta-cognition)
        self._pattern_usage_count: Dict[str, int] = {}  # pattern_id -> usage_count
        self._boredom_alpha: float = kwargs.get("boredom_alpha", 0.5)

        # Parallelism
        self.n_workers: int = n_workers if n_workers is not None else os.cpu_count() or 1

        # Utils
        self.renderer = renderer if renderer is not None else ListRenderer(config)
        self.oracle = ListOracle(config)
        self.counting_oracle = CountingOracle(self.oracle)
        self.executor = PythonExecutor()

        # Pattern evaluator: MetaStrategyController (L5)
        use_fractal_meta = kwargs.get("use_fractal_meta", kwargs.get("use_hfn_meta_controller", False))
        meta_cold_dir = kwargs.get("meta_cold_dir")
        
        if use_fractal_meta:
            self.meta = HFNMetaStrategyController(meta_cold_dir)
        else:
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
        self._step_counter = 0

        super().__init__(**kwargs)


        # Temporal pattern field: recency tracking (L3-L5)
        self._pattern_timestamps: Dict[str, float] = {}  # pattern_id -> last_used_time
        self._recency_half_life: float = kwargs.get("recency_half_life", 5000.0)  # ms
        self._recency_boost: float = kwargs.get("recency_boost", 1.5)  # multiplicative boost

        # Inject priors if forest is empty
        if len(self.forest) == 0:
            self._inject_blank_priors()

    # ------------------------------------------------------------------
    # Prior injection
    # ------------------------------------------------------------------

    def _inject_blank_priors(self) -> None:
        """Inject one prior node per concept into the forest."""
        list_concepts = {
            "LIST_INIT", "FOR_LOOP", "ITEM_ACCESS", "LIST_APPEND", 
            "MAP_START", "MAP_END", "COND_IS_EVEN", "COND_IS_POSITIVE",
            "FOR_EACH_FRAME", "FRAME_APPEND"
        }
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
            base_prob = self.observer.evaluator.curiosity_exploration_probability(
                learnability
            )
            
            # Apply boredom satiation: reduce probability for overused patterns
            usage_count = self._pattern_usage_count.get(task_id, 0)
            satiated_prob = float(base_prob) / (1.0 + self._boredom_alpha * usage_count)
            probs.append(max(1e-6, satiated_prob))

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
                self.observer.register(path[0], protected=False)
        else:
            composed = self._compose_sequence(path)
            if composed:
                composed.id = f"macro_{task_id}"
                self.patterns[task_id] = composed
                if composed.id not in self.forest:
                    self.observer.register(composed, protected=False)

    def _maybe_auto_observe(self, x: Optional[np.ndarray] = None) -> None:
        """Periodically call observer.observe() to drive compression/absorption."""
        if self.auto_observe_frequency <= 0:
            return
        self._observe_counter += 1
        if self._observe_counter % self.auto_observe_frequency == 0:
            obs_input = x
            if obs_input is None and self._replay_buffer:
                obs_input = random.choice(self._replay_buffer)
            
            if obs_input is not None:
                exp = self.observer.observe(obs_input)
                if exp and hasattr(self.retriever.base_retriever, "notify_active"):
                    self.retriever.base_retriever.notify_active([n.id for n in exp.explanation_tree])

    def observe_example(self, x: np.ndarray) -> None:
        """Explicitly observe an input vector (e.g., during training)."""
        exp = self.observer.observe(x)
        if exp and hasattr(self.retriever.base_retriever, "notify_active"):
            self.retriever.base_retriever.notify_active([n.id for n in exp.explanation_tree])
        
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
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str], str, Optional[List[HFN]]]:
        """
        Attempt to solve task using registered strategies in ranked order.

        Returns (success, code_str, strategy_used, path).
        """
        t0 = time.time()
        n_macros = sum(1 for n in self.patterns.values() if n.relation_type == "macro")
        for strat_name, strat_fn in self._ordered_strategies(goal_type):
            oracle_calls_before = self.counting_oracle.call_count
            
            # Use inspection to only pass kwargs that the strategy accepts
            import inspect
            sig = inspect.signature(strat_fn)
            strat_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            path = strat_fn(inputs, outputs, **strat_kwargs)
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
                
                pattern_used = path[0] if len(path) == 1 else self._compose_sequence(path)
                self.meta.record(rec, pattern_used)
                if success:
                    # Track pattern usage for boredom mechanism (L5)
                    self._pattern_usage_count[task_id] = self._pattern_usage_count.get(task_id, 0) + 1
                    
                    self.register_pattern(task_id, path)
                    
                    # Update temporal pattern field: record usage timestamps (L3-L5)
                    current_time = time.time() * 1000
                    for node in path:
                        self._pattern_timestamps[node.id] = current_time

                    # Notify retriever of active nodes
                    if hasattr(self.retriever.base_retriever, "notify_active"):
                        self.retriever.base_retriever.notify_active([n.id for n in path])
                    
                    # Record transitions for forward model (L4)
                    if hasattr(self, "_record_transitions"):
                        self._record_transitions(path, inputs)

                    # After successful solve, optionally observe the encoded input
                    if self.auto_observe_frequency > 0:
                        # Encode the first input as a vector (simplified flattening)
                        flat = np.array(inputs[0]).flatten()
                        if len(flat) < self.m_dim:
                            flat = np.pad(flat, (0, self.m_dim - len(flat)))
                        self._maybe_auto_observe(flat[:self.m_dim])

                    self._maybe_save_state()
                    return True, code, strat_name, path
            else:
                # Strategy failed to even find a path
                wall_ms = (time.time() - t0) * 1000
                rec = SolveRecord(
                    task_id=task_id,
                    goal_type=goal_type,
                    n_macros=n_macros,
                    strategy=strat_name,
                    depth=0,
                    oracle_calls=self.counting_oracle.call_count - oracle_calls_before,
                    success=False,
                    wall_ms=wall_ms,
                )
                self.meta.record(rec, None)
        return False, None, "none", None

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

    def _try_greedy_chain(
        self,
        inputs: List[Any],
        outputs: List[Any],
        max_depth: int = 6,
    ) -> Optional[List[HFN]]:
        """Strategy: zero-branching greedy walk driven by top-ranked retrieval."""
        path = []
        visited_ids = set()
        for _ in range(max_depth):
            # 1. Evaluate current progress
            if not path:
                current_results = [0.0] * len(inputs) 
                current_errors = [None] * len(inputs)
                current_code = ""
            else:
                composed = self._compose_sequence(path)
                current_code = self.renderer.render(composed)
                current_results, current_errors = self.executor.run_batch(current_code, inputs)
                if self._check_outputs(current_results, outputs):
                    return path
            
            # 2. Compute residual goal
            target_state = self.oracle.compute_state(outputs, [None]*len(outputs))
            current_state = self.oracle.compute_state(current_results, current_errors, current_code)
            delta_needed = target_state - current_state
            
            query_mu = np.zeros(self.m_dim)
            query_mu[:self.s_dim] = current_state
            query_mu[self.s_dim + self.dim:] = delta_needed
            query = HFN(mu=query_mu, sigma=np.ones(self.m_dim), use_diag=True)
            
            # 3. Retrieve from candidate ops if provided, else forest
            if hasattr(self, "_candidate_ops") and self._candidate_ops:
                # We still want to rank them by retrieval score relative to query
                candidates = self._rank_nodes(query, self._candidate_ops)
            else:
                candidates = self.retriever.retrieve(query, k=10)
            
            if not candidates:
                break
            
            # Pick first non-visited
            best_node = None
            for cand in candidates:
                if cand.id not in visited_ids:
                    best_node = cand
                    break
            
            if not best_node: break
            
            path.append(best_node)
            visited_ids.add(best_node.id)
            print(f"  [Greedy] Step {len(path)}: {best_node.id}")
        return None

    def _rank_nodes(self, query: HFN, nodes: List[HFN]) -> List[HFN]:
        """Rank a list of nodes by their distance/score relative to query."""
        # Simple geometric + recency if ContextualRetriever is used
        scored = []
        recent_set = set()
        if hasattr(self.retriever.base_retriever, "_recent"):
            recent_set = set(self.retriever.base_retriever._recent)
            boost = self.retriever.base_retriever._recency_boost
        else:
            boost = 0.0

        for node in nodes:
            dist = float(np.sum((node.mu - query.mu) ** 2))
            # Lower score is better
            score = dist - (boost if node.id in recent_set else 0.0)
            scored.append((score, node))
        
        scored.sort(key=lambda x: x[0])
        return [n for s, n in scored]

    def get_weight_with_decay(self, node_id: str) -> float:
        """Get pattern weight with exponential decay applied."""
        current_time = time.time() * 1000  # ms
        base_weight = self.observer.get_weight(node_id)

        if node_id not in self._pattern_decay_times:
            self._pattern_decay_times[node_id] = current_time
            return base_weight

        last_decay = self._pattern_decay_times[node_id]
        time_elapsed = current_time - last_decay
        decay_factor = np.exp(-time_elapsed / self._decay_half_life)
        # Update decay time to keep it moving (standard HPM decay)
        self._pattern_decay_times[node_id] = current_time

        return base_weight * decay_factor

    def _apply_recency_weight(self, node_id: str, base_weight: float) -> float:
        """Apply recency boost: recently-used patterns get higher weight."""
        current_time = time.time() * 1000  # ms
        if node_id not in self._pattern_timestamps:
            self._pattern_timestamps[node_id] = current_time
            return base_weight

        last_used = self._pattern_timestamps[node_id]
        time_since_use = current_time - last_used

        # Recency decay: weight decays toward base over time
        recency_factor = 1.0 + (self._recency_boost - 1.0) * np.exp(
            -time_since_use / self._recency_half_life
        )
        return base_weight * recency_factor

    def _try_bfs(
        self,
        inputs: List[Any],
        outputs: List[Any],
        max_depth: int = 6,
        beam_width: int = 20,
    ) -> Optional[List[HFN]]:
        """Strategy: beam BFS over pattern space, evaluating each depth level in parallel."""
        goal_state = self._outputs_to_goal_state(outputs, inputs)
        query = HFN(mu=goal_state, sigma=np.ones(self.m_dim), use_diag=True)

        if hasattr(self, "_candidate_ops") and self._candidate_ops:
            primitives = self._candidate_ops
        else:
            primitives = self.retriever.retrieve(query, k=beam_width)

        if not primitives:
            return None

        # Execute search
        def run_search(pool: Optional[ProcessPoolExecutor]):
            visited_ids: set = set()
            current_level: List[List[HFN]] = [[p] for p in primitives]
            
            for _depth in range(max_depth):
                # Deduplicate and render candidates for this level
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

                # Evaluate all candidates at this level
                level_successes: List[Tuple[float, float, List[HFN]]] = []
                if pool and len(candidates) > 1:
                    futures = {
                        pool.submit(_eval_path_worker, code, inputs, outputs, self.tolerance): (code, path)
                        for code, path in candidates
                    }
                    for fut in as_completed(futures):
                        is_ok = fut.result()
                        code_eval, path_eval = futures[fut]
                        if is_ok:
                            results, errors = self.executor.run_batch(code_eval, inputs)
                            state = self.oracle.compute_state(results, errors, code_eval, inputs=inputs)
                            # Mask out structural flags (10+) to avoid bias against correct ops
                            mask = np.ones(self.s_dim)
                            mask[10:] = 0.0
                            dist = float(np.sum(((state - goal_state[:self.s_dim]) * mask)**2))
                            # Coherence: Average weight of nodes in path
                            weight = sum(
                                self._apply_recency_weight(n.id, self.get_weight_with_decay(n.id)) 
                                for n in path_eval
                            ) / len(path_eval)
                            level_successes.append((dist, -weight, path_eval))
                else:
                    for code, path in candidates:
                        results, errors = self.executor.run_batch(code, inputs)
                        if self._check_outputs(results, outputs):
                            state = self.oracle.compute_state(results, errors, code, inputs=inputs)
                            mask = np.ones(self.s_dim)
                            mask[10:] = 0.0
                            dist = float(np.sum(((state - goal_state[:self.s_dim]) * mask)**2))
                            weight = sum(
                                self._apply_recency_weight(n.id, self.get_weight_with_decay(n.id)) 
                                for n in path
                            ) / len(path)
                            level_successes.append((dist, -weight, path))
                
                if level_successes:
                    # [UPGRADE] Unified Utility Selection (HPM §3.4)
                    # Utility = Accuracy (1000/(d+1)) + Coherence (weight)
                    # Complexity is constant at this depth level.
                    scored = []
                    for d, nw, p in level_successes:
                        weight = -nw
                        utility = (1000.0 / (d + 1.0)) + weight
                        scored.append((utility, d, weight, p))
                    
                    # Sort by utility (DESC)
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return scored[0][3]

                if _depth + 1 >= max_depth:
                    break

                # Expand next level
                next_level_raw: List[List[HFN]] = []
                for path in [c[1] for c in candidates]:
                    for p in primitives:
                        next_level_raw.append(path + [p])
                
                # Beam Pruning: keep top beam_width by goal-state proximity
                if len(next_level_raw) > beam_width:
                    scored = []
                    for path in next_level_raw:
                        composed = self._compose_sequence(path)
                        if composed:
                            # [UPGRADE] Use Oracle to get actual state for guided ranking
                            code = self.renderer.render(composed)
                            results, errors = self.executor.run_batch(code, inputs)
                            state = self.oracle.compute_state(results, errors, code, inputs)
                            
                            # Update composed mu with empirical delta state
                            composed.mu[self.s_dim + self.dim:] = state
                            
                            dist = float(np.sum((composed.mu - query.mu)**2))
                            scored.append((dist, path))
                    scored.sort(key=lambda x: x[0])
                    top_path_ids = [n.id for n in scored[0][1]]
                    print(f"  [BFS] Depth {_depth+1} Top Rank: {top_path_ids} (Dist: {scored[0][0]:.4f})")
                    current_level = [p for d, p in scored[:beam_width]]
                else:
                    current_level = next_level_raw
            return None

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                return run_search(pool)
        else:
            return run_search(None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _outputs_to_goal_state(self, outputs: List[Any], inputs: Optional[List[Any]] = None) -> np.ndarray:
        """Build a goal state vector from expected outputs."""
        results_dummy = outputs
        errors_dummy = [None] * len(outputs)
        delta_state = self.oracle.compute_state(results_dummy, errors_dummy, inputs=inputs)
        goal = np.zeros(self.m_dim)
        goal[self.s_dim + self.dim:] = delta_state
        return goal

    def _check_outputs(
        self,
        results: List[Any],
        expected: List[Any],
        tolerance: Optional[float] = None,
    ) -> bool:
        """Check whether execution results match expected outputs within tolerance."""
        if tolerance is None:
            tolerance = self.tolerance
            
        if len(results) != len(expected):
            return False
            
        def is_equal(r: Any, e: Any) -> bool:
            if isinstance(r, (int, float, np.float64, np.int64)) and isinstance(e, (int, float, np.float64, np.int64)):
                abs_diff = abs(float(r) - float(e))
                # Hybrid tolerance: abs_diff < tolerance * (1 + abs(e))
                # This allows for absolute noise at small values and relative at large.
                return abs_diff < tolerance * (1.0 + abs(float(e)))
            
            if isinstance(r, np.ndarray) and isinstance(e, np.ndarray):
                return np.allclose(r, e, atol=tolerance, rtol=tolerance)
            if isinstance(r, list) and isinstance(e, list):
                if len(r) != len(e): return False
                return all(is_equal(ri, ei) for ri, ei in zip(r, e))
            import networkx as nx
            if isinstance(r, nx.Graph) and isinstance(e, nx.Graph):
                return set(r.nodes()) == set(e.nodes()) and set(r.edges()) == set(e.edges())
            try:
                return bool(r == e)
            except ValueError:
                # Handle cases like nested arrays that we missed
                if isinstance(r, (list, tuple, np.ndarray)) and isinstance(e, (list, tuple, np.ndarray)):
                    return np.array_equal(r, e)
                return False

        n_matches = 0
        for r, e in zip(results, expected):
            if is_equal(r, e):
                n_matches += 1

        # Robust match: at least 80% of points must match

        return (n_matches / len(expected)) >= 0.80

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[str] = None) -> Path:
        """Serialise agent state (patterns + meta stats) to disk."""
        save_path = Path(path) if path else self.cold_dir / "agent_state.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Patterns
        state = {
            "patterns": self.patterns,
        }
        
        # 2. Meta Stats (L5)
        if hasattr(self.meta, "_stats"):
            state["meta_stats"] = dict(self.meta._stats)
        elif hasattr(self.meta, "save_state"):
            self.meta.save_state()
            
        # 3. Forward Model (L4) - if it has its own persistence
        if hasattr(self, "forward_model") and hasattr(self.forward_model, "save_state"):
            self.forward_model.save_state()
            
        # 4. Global Forest
        if hasattr(self.forest, "save_to_cold"):
            self.forest.save_to_cold()

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
        
        # Meta stats recovery for dict-based controller
        if "meta_stats" in state and hasattr(self.meta, "_stats"):
            self.meta._stats.update(state["meta_stats"])
