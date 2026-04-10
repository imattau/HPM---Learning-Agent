"""
SP54: Experiment 36 — Replicator Contrast Dynamics & Schema Transfer

Addresses structural overgrowth and credit diffusion:
1. Replaces absolute utility with Replicator Contrast Dynamics (baseline subtraction).
2. Weights are updated based on advantage (Utility - Baseline) to restore selection pressure.
3. Tests Schema Transfer across tasks by capturing compositional co-occurrences (e.g., MAP schema).
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

# --- 2. AST RENDERER ---

class ASTRenderer:
    def _get_concept(self, node: HFN) -> Optional[str]:
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}": return c
        action_vec = node.mu[S_DIM : S_DIM + DIM]
        if np.max(action_vec) > 1.0: return CONCEPTS[np.argmax(action_vec)]
        return None

    def _get_hfn_leaves(self, node: HFN) -> List[HFN]:
        if node is None: return []
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
            elif concept == "OP_ADD":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                target = ast.Name(id='val', ctx=ast.Store()) if len(stack) > 1 else ast.Name(id='x', ctx=ast.Store())
                curr_block.append(ast.AugAssign(target=target, op=ast.Add(), value=ast.Constant(value=1)))
            elif concept == "OP_SUB":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                target = ast.Name(id='val', ctx=ast.Store()) if len(stack) > 1 else ast.Name(id='x', ctx=ast.Store())
                curr_block.append(ast.AugAssign(target=target, op=ast.Sub(), value=ast.Constant(value=1)))
            elif concept == "OP_MUL2":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                target = ast.Name(id='val', ctx=ast.Store()) if len(stack) > 1 else ast.Name(id='x', ctx=ast.Store())
                curr_block.append(ast.AugAssign(target=target, op=ast.Mult(), value=ast.Constant(value=2)))
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
            elif concept == "BLOCK_ELSE":
                pass
            elif concept == "RETURN":
                ret_val = ast.IfExp(test=ast.Compare(left=ast.Name(id='res', ctx=ast.Load()), ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]), body=ast.Name(id='res', ctx=ast.Load()), orelse=ast.Name(id='x', ctx=ast.Load()))
                curr_block.append(ast.Return(value=ret_val)); break
        
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
        code = f"def test_func(inp):\n    x = 0\n    val = 0\n    res = None\n    {code_str.replace('\n', '\n    ')}\n"
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
        is_list, lens, means, mins, maxs, firsts, lasts, is_int = [], [], [], [], [], [], [], []
        for out in valid_outputs:
            if isinstance(out, list):
                is_list.append(1.0); lens.append(len(out))
                num_out = [x for x in out if isinstance(x, (int, float))]
                if num_out:
                    means.append(np.mean(num_out)); mins.append(np.min(num_out)); maxs.append(np.max(num_out))
                    firsts.append(num_out[0]); lasts.append(num_out[-1])
            elif isinstance(out, (int, float)):
                is_list.append(0.0); is_int.append(1.0)
                means.append(out); mins.append(out); maxs.append(out); firsts.append(out); lasts.append(out)
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
        self.forest = TieredForest(D=self.m_dim, cold_dir=Path(cold_dir))
        self.persistence = PersistenceManager(root_dir="data/knowledge_base")
        self.experiment_id = "execution_guided"
        self.retriever = GoalConditionedRetriever(self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid))
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, node_use_diag=True, compression_cooccurrence_threshold=1)
        self.renderer = ASTRenderer()
        self.oracle = EmpiricalOracle()
        self.executor = PythonExecutor()
        self.evaluator = Evaluator()
        if not self.forest._registry: self._inject_blank_priors()

    def _inject_blank_priors(self):
        for c in CONCEPTS:
            mu = np.zeros(self.m_dim); mu[S_DIM + CONCEPT_IDX[c]] = 5.0
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{c}", use_diag=True)
            self.forest.register(node); self.observer.protected_ids.add(node.id)
            self.observer.meta_forest.register(HFN(mu=np.array([0.5, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True))

    def _fold_nodes(self, nodes: List[HFN], i: int) -> HFN:
        if not nodes: return None
        current = nodes[0]
        for idx, n in enumerate(nodes[1:]):
            p_mu = (current.mu + n.mu) / 2.0
            p = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=f"exec_compose({current.id}+{n.id})", use_diag=True)
            p.add_child(current); p.add_child(n); current = p
        return current

    def plan(self, inputs: List[Any], expected_outputs: List[Any], max_depth=15, max_iterations=20000) -> Optional[HFN]:
        goal_mu = self.oracle.compute_state(expected_outputs, [None]*len(expected_outputs), "")
        weights = np.array([20.0, 500.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        goal_sigma = 1.0 / (weights + 1e-4)
        goal_hfn = HFN(mu=goal_mu, sigma=goal_sigma, id="goal", use_diag=True)
        initial_outputs, initial_errors = self.executor.run_batch("", inputs)
        initial_state = self.oracle.compute_state(initial_outputs, initial_errors, "")
        
        planning_forest = Forest(D=self.m_dim)
        planning_retriever = GoalConditionedRetriever(planning_forest, target_slice=slice(S_DIM + DIM, self.m_dim), target_weight=1.0, weight_provider=lambda nid: planning_observer.get_weight(nid))
        planning_observer = Observer(forest=planning_forest, retriever=planning_retriever, tau=0.1, node_use_diag=True)
        
        initial_node = HFN(mu=np.zeros(self.m_dim), sigma=np.ones(self.m_dim), id="path_root", use_diag=True)
        initial_node.mu[:S_DIM] = initial_state
        planning_observer.register(initial_node)
        
        initial_accuracy = goal_hfn.log_prob(initial_state)
        planning_observer._set_state_field(initial_node.id, 1, initial_accuracy)
        planning_observer._set_state_field(initial_node.id, 0, 1.0 / (1.0 + np.exp(-initial_accuracy/20.0)))
        
        node_to_path = {initial_node.id: []}; seen_states = {}; observed_deltas = set()
        print(f"    [PLANNER] Goal State: {np.round(goal_mu, 2)}")
        priors = [self.forest.get(f"prior_rule_{c}") for c in CONCEPTS]
        
        for i in range(max_iterations):
            active_nodes = planning_forest.active_nodes()
            baseline_utility = np.mean([planning_observer.get_score(n.id) for n in active_nodes]) if active_nodes else 0.0
            
            if not active_nodes: parent = initial_node
            else:
                sorted_nodes = sorted(active_nodes, key=lambda n: planning_observer.get_weight(n.id), reverse=True)
                n_children = np.array([len(n.children()) for n in sorted_nodes])
                curiosity_bonus = 1.0 / (1.0 + n_children)
                ranks = np.arange(len(sorted_nodes))
                p_vec = np.exp(-ranks / 30.0) * curiosity_bonus; p_vec /= p_vec.sum()
                parent = np.random.choice(sorted_nodes, p=p_vec)
            
            path = node_to_path[parent.id]; state = parent.mu[:S_DIM]
            if len(path) >= max_depth: continue
            
            target_delta = (goal_mu - state) * 10.0
            concept_query = np.zeros(self.m_dim); concept_query[:S_DIM] = state; concept_query[S_DIM + DIM:] = target_delta
            candidates = self.retriever.retrieve(HFN(mu=concept_query, sigma=np.ones(self.m_dim), id="cq"), k=20)
            if np.random.random() < 0.3: candidates.append(np.random.choice(priors))
            
            for rule in set(candidates):
                if rule is None: continue
                new_path = path + [rule]
                code = self.renderer.render(self._fold_nodes(new_path, 0))
                outs, errs = self.executor.run_batch(code, inputs)
                if outs == expected_outputs:
                    print(f"    [SUCCESS] Found exact match at iteration {i}!")
                    if len(new_path) >= 3:
                        # Abstraction Reuse: compress successful sequences into new macro concepts
                        unique_nodes = list({n.id: n for n in new_path}.values())
                        if len(unique_nodes) >= 2:
                            self.observer._track_cooccurrence(unique_nodes)
                            self.observer._check_compression_candidates()
                            print(f"      [MACRO] Sent {len(unique_nodes)} unique concepts to Observer for potential macro-compression.")
                    return self._fold_nodes(new_path, 0)
                
                new_state_vec = self.oracle.compute_state(outs, errs, code)
                if errs[0] is None:
                    accuracy = goal_hfn.log_prob(new_state_vec)
                    complexity = 0.2 * len(new_path)
                    # Removed hand-tuned coherence weights! Relying entirely on backpropagated utility.
                    coherence = 0.0
                    utility = accuracy - complexity + coherence
                    
                    state_key = tuple(np.round(new_state_vec[:17], 4))
                    if state_key in seen_states and utility <= seen_states[state_key]: continue
                    seen_states[state_key] = utility
                    
                    new_node_mu = np.zeros(self.m_dim)
                    new_node_mu[:S_DIM] = new_state_vec
                    new_node_mu[S_DIM + DIM:] = goal_mu - new_state_vec # Store remaining delta for retriever alignment
                    
                    new_node = HFN(mu=new_node_mu, sigma=np.ones(self.m_dim), id=f"prog_{i}_{self.renderer._get_concept(rule)}", use_diag=True)
                    parent.add_child(new_node); planning_observer.register(new_node)
                    node_to_path[new_node.id] = new_path
                    
                    # Backpropagate utility up the planning tree (Temporal Credit Assignment)
                    gamma = 0.95
                    curr_n = new_node
                    curr_util = utility
                    while curr_n is not None and curr_n.id != "path_root":
                        old_score = planning_observer.get_score(curr_n.id)
                        if curr_util > old_score:
                            advantage = curr_util - baseline_utility
                            weight = 1.0 / (1.0 + np.exp(-advantage/10.0)) 
                            planning_observer._set_state_field(curr_n.id, 0, weight)
                            planning_observer._set_state_field(curr_n.id, 1, curr_util)
                        else:
                            break # Optimization: stop backprop if we don't improve the ancestor
                        curr_util *= gamma
                        parents = planning_forest.get_parents(curr_n.id)
                        curr_n = parents[0] if parents else None
                    
                    if new_state_vec[0] == 1.0:
                        delta = new_state_vec - state; vec = np.zeros(self.m_dim); vec[:S_DIM] = state
                        rule_concept = self.renderer._get_concept(rule)
                        delta_key = (rule_concept, tuple(np.round(delta, 4)))
                        if delta_key not in observed_deltas:
                            observed_deltas.add(delta_key)
                            if rule_concept: vec[S_DIM + CONCEPT_IDX[rule_concept]] = 5.0
                            vec[S_DIM + DIM:] = delta * 10.0; self.observer.observe(vec) 
            
            if len(planning_forest._registry) > 1000:
                active_nodes = planning_forest.active_nodes()
                weights_dict = {n.id: planning_observer.get_weight(n.id) for n in active_nodes}
                sorted_nodes = sorted(active_nodes, key=lambda n: weights_dict[n.id], reverse=True)
                for n in sorted_nodes[500:]:
                    if n.id != "path_root": planning_forest.deregister(n.id)
            if i % 1000 == 0 and planning_forest._registry:
                best_node = max(planning_forest.active_nodes(), key=lambda n: planning_observer.get_weight(n.id))
                best_utility = planning_observer.get_score(best_node.id)
                print(f"      Iteration {i} Max Utility: {best_utility:.2f} | Nodes: {len(planning_forest._registry)} | Best Path: {[self.renderer._get_concept(n) for n in node_to_path[best_node.id]]}")
        return None

def run_experiment():
    print("--- SP54: Experiment 36 — Replicator Contrast Dynamics & Schema Transfer ---\n")
    agent = SchemaTransferAgent()
    tasks = [
        {"name": "Task A: Add one", "inputs": [1, 5, 10], "outputs": [2, 6, 11]},
        {"name": "Task B: Map add one (Teaches MAP)", "inputs": [[1, 2], [10, 20]], "outputs": [[2, 3], [11, 21]]},
        {"name": "Task C: Map double (Tests MAP transfer)", "inputs": [[3, 5], [-1, 0]], "outputs": [[6, 10], [-2, 0]]},
        {"name": "Task D: Filter positive (Tests FILTER discovery)", "inputs": [[-1, 2, -3, 4], [0, 5, -2]], "outputs": [[2, 4], [5]]}
    ]
    for t in tasks:
        print(f"\nTask: {t['name']}"); print(f"  Inputs:  {t['inputs']}"); print(f"  Outputs: {t['outputs']}")
        root = agent.plan(t['inputs'], t['outputs'], max_depth=9)
        if root: print(f"  [FINAL AST]\n{agent.renderer.render(root)}")
        else: print("  [FAIL] Could not synthesize program.")

if __name__ == "__main__":
    run_experiment()
