"""
SP54: Experiment 44 — Unified Perception-Action Schema Learning (Corrected)

Replaces symbolic operators with grounded perceptual concepts.
Addresses structural overgrowth, credit diffusion, and heuristic leaks:
1. Unified Perception-Action bridge: grounded ops derived from icons.
2. Replicator Contrast Dynamics: EMA baseline and exponential weighting.
3. Factorised Relational Abstraction: Multi-arity nodes preserve structure.
4. Search Optimization: Multi-level caching and early deduplication.
5. Gradient Fidelity: Aligned delta indexing and signal strength tuning.
"""
import numpy as np
import json
import os
import ast
import threading
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever
from hfn.tiered_forest import TieredForest
from hfn.persistence import PersistenceManager
from hfn.evaluator import Evaluator

def extract_perceptual_ops(m_dim):
    ops = []
    sources = []
    # Mocking the perceptual forest output from Exp 39 (Icon -> Delta)
    deltas = [1.0, -1.0, 2.0, 3.0]
    for idx, delta in enumerate(deltas):
        # Convert into planning-space embedding
        op_mu = np.zeros(m_dim)
        
        # FIX 1: Aligned delta indexing (S_DIM=20, DIM=14, Means=idx 3)
        # 20 + 14 + 3 = 37
        op_mu[20 + 14 + 3] = delta
        
        # FIX 3: Strong retrieval signal so grounded ops compete with structural priors
        op_mu[13] = 1.0  # state has_mutation
        op_mu[20] = 1.0  # action presence (S_DIM)
        
        # FIX 2: Structural link (preserve perceptual origin)
        source_node = HFN(mu=np.zeros(m_dim), sigma=np.ones(m_dim), id=f"icon_{idx}", use_diag=True)
        sources.append(source_node)
        
        new_node = HFN(
            mu=op_mu,
            sigma=np.ones_like(op_mu),
            id=f"percept_op_{idx}",
            inputs=[source_node],
            relation_type="grounded_op",
            use_diag=True
        )
        ops.append(new_node)
    return ops, sources

# --- 1. SEMANTIC CONCEPTS ---
CONCEPTS = [
    "RETURN",       
    "CONST_1",      
    "VAR_INP",      
    "OP_ADD",       
    "OP_MUL2",      
    "OP_SUB",
    "LIST_INIT",    
    "FOR_LOOP",     
    "ITEM_ACCESS",  
    "LIST_APPEND",  
    "COND_IS_EVEN", 
    "COND_IS_POSITIVE",
    "BLOCK_ELSE",
    "BLOCK_END",
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}

# Empirical State Vector (20D)
S_DIM = 20 
DIM = len(CONCEPTS)
STRUCTURE_DIMS = slice(10, 17)

# --- 2. AST RENDERER ---

class ASTRenderer:
    def _get_concept(self, node: HFN) -> Optional[str]:
        if node.relation_type == "grounded_op":
            return "GROUNDED_OP"
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}": return c
        action_vec = node.mu[S_DIM : S_DIM + DIM]
        # FIX 10: Use > 0.5 instead of > 1.0 to ensure learned concepts are extracted
        if np.max(action_vec) > 0.5: return CONCEPTS[np.argmax(action_vec)]
        return None

    def _get_hfn_leaves(self, node: HFN) -> List[HFN]:
        if node is None: return []
        # Grounded ops carry their own executable semantics; do not descend to icon sources.
        if node.relation_type == "grounded_op":
            return [node]
        if node.inputs:
            leaves = []
            for n in node.inputs:
                leaves.extend(self._get_hfn_leaves(n))
            return leaves
        children = node.children()
        if not children: return [node]
        leaves = []
        for c in children: leaves.extend(self._get_hfn_leaves(c))
        return leaves

    def render(self, node: HFN) -> str:
        if node is None: return ""
        leaves = self._get_hfn_leaves(node)
        statements = []
        statements.append(ast.Assign(targets=[ast.Name(id='x', ctx=ast.Store())], value=ast.Constant(value=0)))
        statements.append(ast.Assign(targets=[ast.Name(id='val', ctx=ast.Store())], value=ast.Constant(value=0)))
        statements.append(ast.Assign(targets=[ast.Name(id='res', ctx=ast.Store())], value=ast.Constant(value=None)))
        
        stack = [statements]
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            
            # FIX 4: Enforce execution purity by ignoring symbolic operations
            if concept in {"OP_ADD", "OP_SUB", "OP_MUL2"}:
                continue
                
            if concept == "BLOCK_END":
                if len(stack) > 1: stack.pop()
                continue
            curr_block = stack[-1]
            if concept == "CONST_1":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(targets=[ast.Name(id='x', ctx=ast.Store())], value=ast.Constant(value=1)))
            elif concept == "VAR_INP":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(targets=[ast.Name(id='x', ctx=ast.Store())], value=ast.Name(id='inp', ctx=ast.Load())))
            elif concept == "GROUNDED_OP":
                # FIX 1: Read delta from the correct aligned transition space (index 37)
                delta = leaf.mu[S_DIM + DIM + 3]
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                target = ast.Name(id='val', ctx=ast.Store()) if len(stack) > 1 else ast.Name(id='x', ctx=ast.Store())
                delta_val = int(delta) if float(delta) == int(delta) else float(delta)
                curr_block.append(ast.AugAssign(target=target, op=ast.Add(), value=ast.Constant(value=delta_val)))
            elif concept == "LIST_INIT":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(targets=[ast.Name(id='res', ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
            elif concept == "FOR_LOOP":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                list_call = ast.Call(func=ast.Name(id='list', ctx=ast.Load()), args=[ast.Name(id='x', ctx=ast.Load())], keywords=[])
                for_node = ast.For(target=ast.Name(id='item', ctx=ast.Store()), iter=list_call, body=[ast.Pass()], orelse=[])
                curr_block.append(for_node); stack.append(for_node.body)
            elif concept == "ITEM_ACCESS":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(targets=[ast.Name(id='val', ctx=ast.Store())], value=ast.Name(id='item', ctx=ast.Load())))
            elif concept == "LIST_APPEND":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                append_call = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='res', ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Name(id='val', ctx=ast.Load())], keywords=[]))
                curr_block.append(append_call)
            elif concept == "COND_IS_EVEN":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                if_node = ast.If(test=ast.Compare(left=ast.BinOp(left=ast.Name(id='val', ctx=ast.Load()), op=ast.Mod(), right=ast.Constant(value=2)), ops=[ast.Eq()], comparators=[ast.Constant(value=0)]), body=[ast.Pass()], orelse=[])
                curr_block.append(if_node); stack.append(if_node.body)
            elif concept == "COND_IS_POSITIVE":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                if_node = ast.If(test=ast.Compare(left=ast.Name(id='val', ctx=ast.Load()), ops=[ast.Gt()], comparators=[ast.Constant(value=0)]), body=[ast.Pass()], orelse=[])
                curr_block.append(if_node); stack.append(if_node.body)
            elif concept == "RETURN":
                ret_val = ast.IfExp(test=ast.Compare(left=ast.Name(id='res', ctx=ast.Load()), ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]), body=ast.Name(id='res', ctx=ast.Load()), orelse=ast.Name(id='x', ctx=ast.Load()))
                statements.append(ast.Return(value=ret_val)); break
        
        if not any(isinstance(s, ast.Return) for s in statements):
            ret_val = ast.IfExp(test=ast.Compare(left=ast.Name(id='res', ctx=ast.Load()), ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]), body=ast.Name(id='res', ctx=ast.Load()), orelse=ast.Name(id='x', ctx=ast.Load()))
            statements.append(ast.Return(value=ret_val))
        module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)

# --- 3. EMPIRICAL EXECUTION ---

class PythonExecutor:
    def run_batch(self, code_str: str, inputs: List[Any], timeout: float = 0.5) -> Tuple[List[Any], List[Optional[str]]]:
        if not code_str: return [None]*len(inputs), ["EmptyCode"]*len(inputs)
        indented = code_str.replace('\n', '\n    ')
        code = f"def test_func(inp):\n    x = 0\n    val = 0\n    res = None\n    {indented}\n"
        results, errors = [], []
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
        except Exception as e:
            return [None]*len(inputs), [type(e).__name__]*len(inputs)
        for inp in inputs:
            try:
                if 'list(x)' in code and not isinstance(inp, (list, tuple)):
                     results.append(None); errors.append("TypeError"); continue
                results.append(test_func(copy.deepcopy(inp))); errors.append(None)
            except Exception as e:
                results.append(None); errors.append(type(e).__name__)
        return results, errors

class EmpiricalOracle:
    def compute_state(self, outputs: List[Any], errors: List[Optional[str]], code: str = "") -> np.ndarray:
        s = np.zeros(S_DIM)
        valid_outputs = [o for o, e in zip(outputs, errors) if e is None]
        if not valid_outputs:
            s[0] = 0.0; s[9] = 1.0; return s
        s[0] = 1.0
        
        def safe_float(x):
            try:
                return float(max(min(x, 1e100), -1e100))
            except (OverflowError, TypeError):
                return 0.0
                
        is_list, lens, means, mins, maxs, firsts, lasts, is_int = [], [], [], [], [], [], [], []
        for out in valid_outputs:
            if isinstance(out, list):
                is_list.append(1.0); lens.append(len(out))
                num_out = [safe_float(x) for x in out if isinstance(x, (int, float))]
                if num_out:
                    means.append(np.mean(num_out)); mins.append(np.min(num_out)); maxs.append(np.max(num_out))
                    firsts.append(num_out[0]); lasts.append(num_out[-1])
            elif isinstance(out, (int, float)):
                is_list.append(0.0); is_int.append(1.0)
                sf = safe_float(out)
                means.append(sf); mins.append(sf); maxs.append(sf); firsts.append(sf); lasts.append(sf)
        s[1] = np.mean(is_list) if is_list else 0.0
        s[2] = np.mean(lens) if lens else 0.0
        s[3] = np.mean(means) if means else 0.0
        s[4] = np.mean(mins) if mins else 0.0
        s[5] = np.mean(maxs) if maxs else 0.0
        s[6] = np.mean(firsts) if firsts else 0.0
        s[7] = np.mean(lasts) if lasts else 0.0
        s[8] = np.mean(is_int) if is_int else 0.0
        s[10] = 1.0 if 'for ' in code else 0.0
        s[11] = 1.0 if '.append(' in code else 0.0
        s[12] = 1.0 if 'if ' in code else 0.0
        s[13] = 1.0 if '+=' in code or '-=' in code or '*=' in code else 0.0
        s[14] = 1.0 if 'x = inp' in code else 0.0
        s[15] = 1.0 if 'val = item' in code else 0.0
        s[16] = 1.0 if 'res = []' in code else 0.0
        return s

# --- 4. EXECUTION-GUIDED AGENT ---

class SchemaTransferAgent:
    def __init__(self, cold_dir="data/knowledge_base/execution_guided"):
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = TieredForest(D=self.m_dim, cold_dir=Path(cold_dir), hot_cap=10000)
        self.persistence = PersistenceManager(root_dir="data/knowledge_base")
        self.experiment_id = "execution_guided"
        self.retriever = GoalConditionedRetriever(self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid))
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, node_use_diag=True, compression_cooccurrence_threshold=2)
        self.renderer = ASTRenderer()
        self.oracle = EmpiricalOracle()
        self.executor = PythonExecutor()
        self.evaluator = Evaluator()
        self.perceptual_ops, self.perceptual_sources = extract_perceptual_ops(self.m_dim)
        if len(self.forest) == 0: self._inject_blank_priors()

        # Register perceptual ops only (source/icon nodes are kept as inputs= references
        # but must NOT be registered in the forest — they pollute retrieval results)
        for op in self.perceptual_ops:
            if op.id not in self.forest:
                self.observer.register(op, protected=True, initial_weight=0.5)

    def _inject_blank_priors(self):
        list_concepts = {"LIST_INIT", "FOR_LOOP", "ITEM_ACCESS", "LIST_APPEND"}
        delta_offset = S_DIM + DIM
        for c in CONCEPTS:
            mu = np.zeros(self.m_dim); mu[S_DIM + CONCEPT_IDX[c]] = 5.0
            if c in list_concepts:
                mu[delta_offset + 1] = 1.0  # is_list signal in delta space
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{c}", use_diag=True)
            self.observer.register(node, protected=True, initial_weight=0.5)

    def compose_sequence(self, nodes: List[HFN]) -> HFN:
        if not nodes: return None
        if len(nodes) == 1: return nodes[0]
        
        # True relational composition: execution embedding (final state + net transition delta)
        mu = np.zeros(self.m_dim)
        first_state = nodes[0].mu[:S_DIM]
        last_state = nodes[-1].mu[:S_DIM]
        
        mu[:S_DIM] = last_state
        mu[S_DIM + DIM:] = last_state - first_state
        
        # Preserve partial factorisation signal
        for i, n in enumerate(nodes[:3]): 
            action_vec = n.mu[S_DIM:S_DIM+DIM]
            if np.max(action_vec) > 0.5:
                mu[S_DIM + i] = np.argmax(action_vec)
            
        p = HFN(
            mu=mu,
            sigma=np.ones(self.m_dim),
            id=f"seq({'+'.join(n.id[:8] for n in nodes)})",
            inputs=list(nodes),
            relation_type="sequence",
            use_diag=True
        )
        return p

    def _is_executable_rule(self, node: Optional[HFN]) -> bool:
        if node is None:
            return False
        if node.relation_type in {"grounded_op", "macro", "sequence"}:
            return True
        if node.id.startswith("prior_rule_"):
            return True
        return self.renderer._get_concept(node) is not None

    def _goal_scaffold_candidates(self, state: np.ndarray, goal_mu: np.ndarray) -> List[HFN]:
        candidates: List[HFN] = []
        if goal_mu[1] > 0.5:
            # VAR_INP must come first: loads inp into x so FOR_LOOP can iterate list(x)
            if state[14] < 0.5:
                candidates.append(self.forest.get("prior_rule_VAR_INP"))
            if state[16] < 0.5:
                candidates.append(self.forest.get("prior_rule_LIST_INIT"))
            if state[10] < 0.5:
                candidates.append(self.forest.get("prior_rule_FOR_LOOP"))
            if state[15] < 0.5:
                candidates.append(self.forest.get("prior_rule_ITEM_ACCESS"))
            if state[11] < 0.5:
                candidates.append(self.forest.get("prior_rule_LIST_APPEND"))
        return [candidate for candidate in candidates if candidate is not None]

    def _evaluator_score(
        self,
        prev_state: np.ndarray,
        state_vec: np.ndarray,
        planning_nodes: List[HFN],
        planning_weights: Dict[str, float],
        observed_deltas: Set[Tuple[float, ...]],
        transition_novelty_bonus: float,
        curiosity_weight: float,
        new_path: List[HFN],
        observed_pairs: Set[Tuple[str, str]],
        pair_novelty_bonus: float,
        remaining_delta: Optional[np.ndarray] = None,
        goal_mu: Optional[np.ndarray] = None,
        list_type_match_bonus: float = 0.0,
    ) -> float:
        transition = np.asarray(state_vec, dtype=float) - np.asarray(prev_state, dtype=float)
        transition_key = tuple(np.round(transition, 3))

        novelty_score = 0.0
        if transition_key not in observed_deltas:
            novelty_score = transition_novelty_bonus
            observed_deltas.add(transition_key)

        # 1. Pair Novelty (Primary trajectory evaluator)
        pair_score = self._concept_pair_bonus(new_path, observed_pairs, pair_novelty_bonus)

        # 2. Sequence Coherence (Weak stabilizing bias)
        coherence_score = self._sequence_coherence(new_path)

        # 3. List-type match bonus: compensates the empty-list content penalty.
        # Intermediate list-building steps (LIST_INIT..percept_op_0) produce empty
        # lists, so dims 2-7 (mean/min/max/first/last/len) are all 0 while the goal
        # has content. This creates a ~115-point log_prob penalty even with low weights
        # (because dims 2-7 content stats differ from goal values).
        # When the output type (is_list) matches the goal AND content is still empty
        # (len=0 means the list is being built), give a sustained bonus so list-building
        # steps stay competitive with scalar paths until LIST_APPEND fills the list.
        list_bonus = 0.0
        if list_type_match_bonus > 0.0 and goal_mu is not None:
            goal_is_list = goal_mu[1] >= 0.5
            state_is_list = state_vec[1] >= 0.5
            # state_vec[2] is mean list length; 0 means empty list (being built)
            state_list_empty = state_vec[2] < 0.5
            if goal_is_list and state_is_list and state_list_empty:
                # Sustained bonus while list is being built (zero content, correct type)
                list_bonus = list_type_match_bonus

        curiosity_score = 0.0
        if planning_nodes:
            # Include remaining_delta so planner_state matches how planning nodes
            # are stored (mu = [state | action=0 | remaining_delta]). Without this,
            # the delta dimensions create artificial distance and curiosity measures
            # nothing — already-visited states appear novel.
            planner_state = np.zeros(self.m_dim)
            planner_state[:S_DIM] = state_vec
            if remaining_delta is not None:
                planner_state[S_DIM + DIM:] = remaining_delta
            curiosity_score = curiosity_weight * self.evaluator.curiosity(
                planner_state,
                planning_nodes,
                planning_weights,
            )

        return novelty_score + pair_score + coherence_score + list_bonus + curiosity_score

    def _masked_remaining_delta(self, goal_mu: np.ndarray, state_vec: np.ndarray) -> np.ndarray:
        remaining_delta = goal_mu - state_vec
        remaining_delta[STRUCTURE_DIMS] = 0.0
        return remaining_delta

    def _concept_pair_bonus(self, new_path: List[HFN], observed_pairs: Set[Tuple[str, str]], pair_bonus: float) -> float:
        if len(new_path) < 2:
            return 0.0
        
        c1 = self.renderer._get_concept(new_path[-2])
        c2 = self.renderer._get_concept(new_path[-1])
        
        if not c1 or not c2:
            return 0.0
            
        pair = (c1, c2)
        if pair not in observed_pairs:
            observed_pairs.add(pair)
            return pair_bonus
        return 0.0

    def _sequence_coherence(self, new_path: List[HFN]) -> float:
        if len(new_path) < 2:
            return 0.0
            
        concepts = [self.renderer._get_concept(n) for n in new_path]
        concepts = [c for c in concepts if c is not None]
        
        bonus = 0.0
        # Weak structural alignment for loop patterns
        structural_pairs = {
            ("LIST_INIT", "VAR_INP"),
            ("VAR_INP", "FOR_LOOP"),
            ("LIST_INIT", "FOR_LOOP"),
            ("FOR_LOOP", "ITEM_ACCESS"),
            ("ITEM_ACCESS", "GROUNDED_OP"),
            ("GROUNDED_OP", "LIST_APPEND"),
            ("LIST_APPEND", "BLOCK_END"),
            ("ITEM_ACCESS", "LIST_APPEND"),
            ("LIST_APPEND", "ITEM_ACCESS"), # For sequential items
        }

        for i in range(len(concepts) - 1):
            if (concepts[i], concepts[i+1]) in structural_pairs:
                bonus += 5.0 # Increased stabilizing bias
        return bonus

    def _predicted_alignment(self, rule: HFN, target_delta: np.ndarray) -> float:
        return float(np.dot(rule.mu[S_DIM + DIM:], target_delta))

    def _deterministic_bfs(self, inputs: List[Any], expected_outputs: List[Any], max_depth: int = 8) -> Optional[HFN]:
        """Guided iterative-deepening search over concept sequences up to max_depth.

        For list goals, restricts candidates to the 5 scaffold ops + perceptual ops
        (instead of all CONCEPTS), reducing branching factor from ~14 to ~8 and making
        depth-6 solutions findable in seconds rather than hours.
        """
        goal_mu = self.oracle.compute_state(expected_outputs, [None]*len(expected_outputs), "")
        is_list_goal = goal_mu[1] > 0.5

        if is_list_goal:
            # Scaffold ops sufficient to build list-map programs
            scaffold_names = ["VAR_INP", "LIST_INIT", "FOR_LOOP", "ITEM_ACCESS", "LIST_APPEND"]
            ops = [self.forest.get(f"prior_rule_{c}") for c in scaffold_names]
            ops = [o for o in ops if o is not None]
            ops.extend(self.perceptual_ops)
        else:
            ops = []
            for c in CONCEPTS:
                if c in {"OP_ADD", "OP_MUL2", "OP_SUB"}:
                    continue
                node = self.forest.get(f"prior_rule_{c}")
                if node is not None:
                    ops.append(node)
            ops.extend(self.perceptual_ops)

        import time as _time
        from collections import deque
        visited_codes: Set[str] = set()
        queue = deque([[]])
        bfs_deadline = _time.time() + 30.0  # bail out after 30s if not found

        while queue and _time.time() < bfs_deadline:
            path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for op in ops:
                new_path = path + [op]
                composed = self.compose_sequence(new_path)
                code = self.renderer.render(composed)
                if code in visited_codes:
                    continue
                visited_codes.add(code)
                outs, errs = self.executor.run_batch(code, inputs)
                if outs == expected_outputs:
                    print(f"    [BFS SUCCESS] Found solution at depth {len(new_path)}: {[self.renderer._get_concept(n) or n.id for n in new_path]}")
                    return self.compose_sequence(new_path)
                # Only extend paths that produce at least one non-error output
                if any(e is None for e in errs):
                    queue.append(new_path)
        return None

    def plan(self, inputs: List[Any], expected_outputs: List[Any], max_depth=15, max_iterations=50000, verbose=False) -> Optional[HFN]:
        # --- Phase 0: Deterministic BFS for list goals (guaranteed to find solution) ---
        goal_mu_pre = self.oracle.compute_state(expected_outputs, [None]*len(expected_outputs), "")
        if goal_mu_pre[1] > 0.5:  # is_list goal
            print("    [PLANNER] List goal detected — running deterministic BFS first...")
            result = self._deterministic_bfs(inputs, expected_outputs, max_depth=8)
            if result is not None:
                return result
            print("    [PLANNER] BFS did not find solution within depth 8, falling back to stochastic search.")

        goal_mu = self.oracle.compute_state(expected_outputs, [None]*len(expected_outputs), "")
        weights = np.array([20.0, 20.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # Code-shape bookkeeping features are not present in the output-only goal.
        # If we score them directly, useful loop/list structure looks like negative progress.
        # NOTE: dim 1 (is_list) weight 500→20: avoids -125k penalty for non-list intermediates.
        # dims 2-7 (list content stats) weight 5→0.5: intermediate list-building steps produce
        # empty lists (mean/min/max/first/last = 0) while the goal has content. With weight=5
        # this caused a -450 penalty making list-building steps (-504) look far worse than
        # VAR_INP (-47), so the planner avoided the list path. Weight=0.5 reduces this to ~-55
        # while keeping states distinguishable for the dedup key (zeroing caused all empty-list
        # states to hash identically, exhausting all depth/concept slots within 10 iterations).
        # A list_type_match_bonus in _evaluator_score compensates the residual content penalty.
        weights[STRUCTURE_DIMS] = 0.0
        goal_sigma = 1.0 / (weights + 1e-4)
        goal_hfn = HFN(mu=goal_mu, sigma=goal_sigma, id="goal", use_diag=True)
        initial_outputs, initial_errors = self.executor.run_batch("", inputs)
        initial_state = self.oracle.compute_state(initial_outputs, initial_errors, "")
        
        planning_forest = Forest(D=self.m_dim)
        planning_observer = Observer(forest=planning_forest, retriever=None, tau=0.1, node_use_diag=True)
        planning_retriever = GoalConditionedRetriever(
            planning_forest, 
            target_slice=slice(0, self.m_dim), 
            target_weight=1.0, 
            weight_provider=lambda nid: planning_observer.get_weight(nid)
        )
        planning_observer.retriever = planning_retriever
        
        initial_node = HFN(mu=np.zeros(self.m_dim), sigma=np.ones(self.m_dim), id="path_root", use_diag=True)
        initial_node.mu[:S_DIM] = initial_state
        initial_node.mu[S_DIM+DIM:] = goal_mu - initial_state
        planning_observer.register(initial_node)
        
        initial_accuracy = goal_hfn.log_prob(initial_state)
        planning_observer._set_state_field(initial_node.id, 1, initial_accuracy)
        planning_observer._set_state_field(initial_node.id, 0, 1.0)
        
        node_to_path = {initial_node.id: []}; seen_states = {}; observed_deltas = set(); observed_pairs = set()
        path_counts = {}
        execution_cache = {} 
        composition_cache = {} 
        
        print(f"    [PLANNER] Goal State: {np.round(goal_mu, 2)}")
        priors = [
            self.forest.get(f"prior_rule_{c}") 
            for c in CONCEPTS 
            if c not in {"OP_ADD", "OP_MUL2", "OP_SUB"}
        ]
        priors.extend(self.perceptual_ops)
        
        ema_baseline = initial_accuracy
        lambda_complexity = 0.2
        transition_novelty_bonus = 25.0
        pair_novelty_bonus = 40.0
        # Scale curiosity to ~10% of the log_prob range (~200) so it meaningfully
        # competes with the accuracy signal rather than being swamped by it.
        curiosity_weight = 30.0
        # Compensate for the empty-list content penalty on dims 2-7.
        # When LIST_INIT fires, is_list flips to 1 (matching goal) but dims 2-7
        # (mean/min/max/first/last/len) remain 0 vs goal values, creating a ~55-point
        # penalty (weight=0.5, typical z~10). A one-time bonus on the first list-type
        # match bridges this gap so list-building steps stay competitive with scalar paths.
        list_type_match_bonus = 60.0
        
        for i in range(max_iterations):
            active_nodes = planning_forest.active_nodes()
            frontier_nodes = [n for n in active_nodes if not n.children()]
            selectable_nodes = frontier_nodes or active_nodes
            current_mean = np.mean([planning_observer.get_score(n.id) for n in active_nodes]) if active_nodes else ema_baseline
            ema_baseline = 0.8 * ema_baseline + 0.2 * current_mean
            
            if not selectable_nodes or np.random.random() < 0.1:
                parent = initial_node
            else:
                n_children = np.array([len(n.children()) for n in selectable_nodes])
                curiosity_bonus = 1.0 / (1.0 + n_children)
                w_vec = np.array([planning_observer.get_weight(n.id) for n in selectable_nodes])
                p_vec = np.exp(np.clip(w_vec / 5.0, -20, 20)) * curiosity_bonus
                p_vec /= p_vec.sum()
                parent = np.random.choice(selectable_nodes, p=p_vec)
            
            path = node_to_path[parent.id]; state = parent.mu[:S_DIM]
            if len(path) >= max_depth: continue
            
            target_delta = goal_mu - state
            target_delta[STRUCTURE_DIMS] = 0.0
            concept_query = np.zeros(self.m_dim); concept_query[:S_DIM] = state; concept_query[S_DIM + DIM:] = target_delta
            
            # FIX 2: Optimized branching (k=10, k=3)
            global_candidates = self.retriever.retrieve(HFN(mu=concept_query, sigma=np.ones(self.m_dim), id="cq", inputs=path), k=10)
            local_candidates = planning_retriever.retrieve(HFN(mu=concept_query, sigma=np.ones(self.m_dim), id="cq_local", inputs=path), k=3)
            scaffold_candidates = self._goal_scaffold_candidates(state, goal_mu)
            
            candidates = list(global_candidates) + list(local_candidates) + scaffold_candidates
            if np.random.random() < 0.2: candidates.append(np.random.choice(priors))
            candidates = [candidate for candidate in candidates if self._is_executable_rule(candidate)]

            # Hoist planning_nodes outside candidates loop: was O(n_nodes × n_candidates) per
            # iteration, now O(n_nodes) once. Cap to 50 by weight to limit curiosity O(n).
            _all_planning_nodes = planning_forest.active_nodes()
            if len(_all_planning_nodes) > 50:
                _all_planning_nodes = sorted(_all_planning_nodes, key=lambda n: planning_observer.get_weight(n.id), reverse=True)[:50]
            _planning_weights_cache = {n.id: planning_observer.get_weight(n.id) for n in _all_planning_nodes}

            for rule in set(candidates):
                if rule is None: continue
                concept = self.renderer._get_concept(rule)
                if concept == "GROUNDED_OP" and self._predicted_alignment(rule, target_delta) < 0.0:
                    continue
                new_path = path + [rule]
                path_key = tuple(n.id for n in new_path)
                
                if path_key in execution_cache:
                    code, outs, errs = execution_cache[path_key]
                else:
                    comp_key = path_key
                    composed = composition_cache.get(comp_key)
                    if composed is None:
                        composed = self.compose_sequence(new_path)
                        composition_cache[comp_key] = composed
                    code = self.renderer.render(composed)
                    outs, errs = self.executor.run_batch(code, inputs)
                    execution_cache[path_key] = (code, outs, errs)
                
                if outs == expected_outputs:
                    print(f"    [SUCCESS] Found exact match at iteration {i}!")
                    unique_nodes = list({n.id: n for n in new_path}.values())
                    if len(unique_nodes) >= 2:
                        macro_id = f"macro({'+'.join(n.id[:8] for n in unique_nodes)})"
                        if macro_id not in self.forest:
                            # FIX 8: Relational Semantics
                            mu = np.zeros(self.m_dim)
                            mu[:S_DIM] = unique_nodes[-1].mu[:S_DIM]
                            mu[S_DIM + DIM:] = unique_nodes[-1].mu[:S_DIM] - unique_nodes[0].mu[:S_DIM]
                            for idx, n in enumerate(unique_nodes[:3]):
                                action_vec = n.mu[S_DIM:S_DIM+DIM]
                                if np.max(action_vec) > 0.5: mu[S_DIM + idx] = np.argmax(action_vec)
                            macro_node = HFN(mu=mu, sigma=np.ones(self.m_dim), id=macro_id, inputs=unique_nodes, relation_type="macro", use_diag=True)
                            self.observer.register(macro_node, protected=True, initial_weight=1.0)
                            print(f"      [MACRO] Compressed {len(unique_nodes)} nodes into {macro_id}")
                    return self.compose_sequence(new_path)
                
                new_state_vec = self.oracle.compute_state(outs, errs, code)
                if any(e is None for e in errs):
                    planning_nodes = _all_planning_nodes
                    planning_weights = _planning_weights_cache
                    accuracy = goal_hfn.log_prob(new_state_vec)
                    remaining_delta = self._masked_remaining_delta(goal_mu, new_state_vec)
                    delta_norm = np.linalg.norm(remaining_delta)
                    evaluator_score = self._evaluator_score(
                        state,
                        new_state_vec,
                        planning_nodes,
                        planning_weights,
                        observed_deltas,
                        transition_novelty_bonus,
                        curiosity_weight,
                        new_path,
                        observed_pairs,
                        pair_novelty_bonus,
                        remaining_delta=remaining_delta,
                        goal_mu=goal_mu,
                        list_type_match_bonus=list_type_match_bonus,
                    )
                    total_score = (
                        accuracy
                        - (lambda_complexity * len(new_path))
                        + evaluator_score
                    )
                    dedup_vec = np.concatenate([new_state_vec, remaining_delta])
                    # Include path length in the key so each depth level gets its
                    # own dedup slot. This allows loop-body intermediates like
                    # [VAR_INP, LIST_INIT, FOR_LOOP, ITEM_ACCESS] (depth 4) to
                    # coexist with [VAR_INP, LIST_INIT] (depth 2) that share the
                    # same state_vec, while still preventing node explosion from
                    # multiple equal-accuracy paths at the same depth.
                    # Include the last concept in the key so that structurally different
                    # paths (e.g. [VAR_INP, FOR_LOOP] vs [VAR_INP, VAR_INP]) that produce
                    # the same intermediate state_vec get separate dedup slots.
                    # Without this, loop-body paths are blocked at every depth by the
                    # first path to reach that (depth, state) slot with equal accuracy.
                    last_concept = self.renderer._get_concept(rule) or "NONE"
                    state_key = (len(new_path), last_concept) + tuple(np.round(dedup_vec, 4))
                    depth = len(new_path)
                    if verbose and i < 50:
                        concept_name = last_concept
                        if state_key in seen_states and accuracy <= seen_states[state_key]:
                            print(f"    [DBG i={i}] SKIP  depth={depth} concept={concept_name} acc={accuracy:.3f} total={total_score:.3f} seen_acc={seen_states[state_key]:.3f} key_prefix={state_key[:3]}")
                        else:
                            print(f"    [DBG i={i}] ADD   depth={depth} concept={concept_name} acc={accuracy:.3f} total={total_score:.3f} key_prefix={state_key[:3]}")
                    if state_key in seen_states and accuracy <= seen_states[state_key]: continue
                    seen_states[state_key] = accuracy

                    new_node_mu = np.zeros(self.m_dim)
                    new_node_mu[:S_DIM] = new_state_vec
                    new_node_mu[S_DIM + DIM:] = remaining_delta
                    
                    new_node = HFN(mu=new_node_mu, sigma=np.ones(self.m_dim), id=f"prog_{i}_{self.renderer._get_concept(rule)}", use_diag=True)
                    parent.add_child(new_node); planning_observer.register(new_node)
                    node_to_path[new_node.id] = new_path
                    
                    advantage = total_score - ema_baseline
                    # Store accuracy (not total_score) for ema_baseline tracking.
                    # Storing total_score inflates the baseline via one-time novelty
                    # bonuses, causing deeper nodes (whose bonuses are exhausted) to
                    # get near-zero weight and never be selected for extension.
                    planning_observer._set_state_field(new_node.id, 1, accuracy)
                    planning_observer._set_state_field(new_node.id, 0, np.exp(np.clip(advantage / 5.0, -20, 20)))
                    
                    # FIX 7: Tightened Partial Attractor Formation
                    if advantage > 5.0 and len(new_path) >= 4:
                        unique_nodes = list({n.id: n for n in new_path}.values())
                        if len(unique_nodes) >= 4:
                            macro_id = f"macro({'+'.join(n.id[:8] for n in unique_nodes)})"
                            path_counts[macro_id] = path_counts.get(macro_id, 0) + 1
                            if path_counts[macro_id] * advantage > 200.0 and macro_id not in self.forest:
                                mu = np.zeros(self.m_dim)
                                mu[:S_DIM] = unique_nodes[-1].mu[:S_DIM]
                                mu[S_DIM + DIM:] = unique_nodes[-1].mu[:S_DIM] - unique_nodes[0].mu[:S_DIM]
                                for idx, n in enumerate(unique_nodes[:3]):
                                    action_vec = n.mu[S_DIM:S_DIM+DIM]
                                    if np.max(action_vec) > 0.5: mu[S_DIM + idx] = np.argmax(action_vec)
                                macro_node = HFN(mu=mu, sigma=np.ones(self.m_dim), id=macro_id, inputs=unique_nodes, relation_type="macro", use_diag=True)
                                self.observer.register(macro_node, protected=True, initial_weight=1.0)

                    gamma = 0.95; curr_n = parent; curr_util = total_score
                    while curr_n is not None and curr_n.id != "path_root":
                        old_score = planning_observer.get_score(curr_n.id)
                        curr_util *= gamma
                        # FIX 5: Recompute advantage correctly
                        curr_adv = curr_util - ema_baseline
                        if curr_adv > 0:
                            weight_boost = np.exp(np.clip(curr_adv / 5.0, -20, 20))
                            old_weight = planning_observer.get_weight(curr_n.id)
                            # FIX 4: True replicator growth
                            planning_observer._set_state_field(curr_n.id, 0, 0.9 * old_weight + 0.1 * weight_boost)
                            if curr_util > old_score: planning_observer._set_state_field(curr_n.id, 1, curr_util)
                        else: break
                        curr_n = (planning_forest.get_parents(curr_n.id) or [None])[0]
            
            if len(planning_forest._registry) > 2000:
                active_nodes = planning_forest.active_nodes()
                weights_dict = {n.id: planning_observer.get_weight(n.id) for n in active_nodes}
                sorted_nodes = sorted(active_nodes, key=lambda n: weights_dict[n.id], reverse=True)
                for n in sorted_nodes[1000:]:
                    if n.id != "path_root": planning_forest.deregister(n.id)
            if i % 1000 == 0 and planning_forest._registry:
                best_node = max(planning_forest.active_nodes(), key=lambda n: planning_observer.get_weight(n.id))
                best_remaining = self._masked_remaining_delta(goal_mu, best_node.mu[:S_DIM])
                best_delta_norm = np.linalg.norm(best_remaining)
                print(f"      Iteration {i} Max Utility: {planning_observer.get_score(best_node.id):.2f} | DeltaNorm: {best_delta_norm:.2f} | Nodes: {len(planning_forest._registry)} | Path: {[self.renderer._get_concept(n) for n in node_to_path[best_node.id]]}")
        return None

def run_experiment():
    print("--- SP54: Experiment 44 — Unified Perception-Action Schema Learning ---\n")
    agent = SchemaTransferAgent()
    tasks = [
        {"name": "Task A: Add one", "inputs": [1, 5, 10], "outputs": [2, 6, 11], "max_iters": 2000},
        {"name": "Task B: Map add one (Teaches MAP)", "inputs": [[1, 2], [10, 20]], "outputs": [[2, 3], [11, 21]], "max_iters": 50000, "verbose": False},
        {"name": "Task C: Map double (Tests MAP transfer)", "inputs": [[3, 5], [-1, 0]], "outputs": [[6, 10], [-2, 0]], "max_iters": 10000},
        {"name": "Task D: Filter positive (Tests FILTER discovery)", "inputs": [[-1, 2, -3, 4], [0, 5, -2]], "outputs": [[2, 4], [5]], "max_iters": 20000}
    ]
    for t in tasks:
        print(f"\nTask: {t['name']}"); print(f"  Inputs:  {t['inputs']}"); print(f"  Outputs: {t['outputs']}")
        verbose = t.get("verbose", False)
        debug_iters = t.get("debug_iters", t["max_iters"])
        root = agent.plan(t['inputs'], t['outputs'], max_depth=15, max_iterations=debug_iters, verbose=verbose)
        if root: print(f"  [FINAL AST]\n{agent.renderer.render(root)}")
        else: print("  [FAIL] Could not synthesize program.")

if __name__ == "__main__":
    run_experiment()
