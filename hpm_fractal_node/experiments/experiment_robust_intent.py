"""
SP53: Experiment 29 — Robust Intent Inference and AST Synthesis

Redesigns the Intent-Driven Reasoning pipeline to fix Oracle Leakage, 
lossy state representation, and brittle string-based rendering.
"""
import numpy as np
import json
import os
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever
from hfn.tiered_forest import TieredForest
from hfn.persistence import PersistenceManager

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
    "BLOCK_ELSE",
    "BLOCK_END",
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}

# 14D Extended State Vector
S_DIM = 14 
DIM = len(CONCEPTS)

# --- 2. AST RENDERER ---

class ASTRenderer:
    def _get_concept(self, node: HFN) -> Optional[str]:
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}": return c
        if node.id.startswith("prior_rule_") or node.id.startswith("call_") or node.id.startswith("template_") or node.id.startswith("discovery_"):
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
            elif concept == "OP_MUL2":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                target = ast.Name(id='val', ctx=ast.Store()) if len(stack) > 1 else ast.Name(id='x', ctx=ast.Store())
                curr_block.append(ast.AugAssign(target=target, op=ast.Mult(), value=ast.Constant(value=2)))
            elif concept == "LIST_INIT":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(targets=[ast.Name(id='res', ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
            elif concept == "FOR_LOOP":
                if isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                for_node = ast.For(target=ast.Name(id='item', ctx=ast.Store()), iter=ast.Name(id='x', ctx=ast.Load()), body=[ast.Pass()], orelse=[])
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
            elif concept == "RETURN":
                ret_val = ast.IfExp(test=ast.Compare(left=ast.Name(id='res', ctx=ast.Load()), ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]), body=ast.Name(id='res', ctx=ast.Load()), orelse=ast.Name(id='x', ctx=ast.Load()))
                curr_block.append(ast.Return(value=ret_val)); break
        
        if not any(isinstance(s, ast.Return) for s in statements):
            ret_val = ast.IfExp(test=ast.Compare(left=ast.Name(id='res', ctx=ast.Load()), ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]), body=ast.Name(id='res', ctx=ast.Load()), orelse=ast.Name(id='x', ctx=ast.Load()))
            statements.append(ast.Return(value=ret_val))
        module = ast.Module(body=statements, type_ignores=[]); ast.fix_missing_locations(module); return ast.unparse(module)

# --- 3. CONSTRAINT ORACLE ---

class ConstraintOracle:
    def __init__(self):
        self.knowledge = {
            "Double all numbers": {
                0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0, 6: 1.0, 12: 1.0, 10: 0.0
            },
            "Filter even numbers": {
                0: 1.0, 1: 1.0, 2: 1.0, 6: 1.0, 7: 1.0, 12: 1.0, 10: 0.0, 11: 0.0
            }
        }
    def get_constraints(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        s = np.zeros(S_DIM); mask = np.zeros(S_DIM, dtype=bool)
        for key, constraints in self.knowledge.items():
            if prompt.lower() in key.lower():
                for idx, val in constraints.items(): s[idx] = val; mask[idx] = True
                return s, mask
        s[0] = 1.0; mask[0] = True; return s, mask

# --- 4. ROBUST INTENT AGENT ---

class RobustIntentAgent:
    def __init__(self, cold_dir="data/knowledge_base/robust_intent"):
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = TieredForest(D=self.m_dim, cold_dir=Path(cold_dir))
        self.persistence = PersistenceManager(root_dir="data/knowledge_base")
        self.retriever = GoalConditionedRetriever(self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid))
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, node_use_diag=True)
        self.renderer = ASTRenderer(); self.oracle = ConstraintOracle()
        self.persistence.load(self.forest, self.observer, "curiosity")
        if "prior_rule_RETURN" not in self.forest: self._inject_priors()

    def _inject_priors(self):
        for c in CONCEPTS:
            mu = np.zeros(self.m_dim); mu[S_DIM + CONCEPT_IDX[c]] = 5.0
            if c == "RETURN": mu[S_DIM + DIM + 0] = 50.0; mu[1] = 1.0; mu[12] = 1.0; mu[6] = 1.0
            elif c == "LIST_INIT": mu[S_DIM + DIM + 1] = 50.0; mu[S_DIM + DIM + 12] = 50.0
            elif c == "FOR_LOOP": mu[S_DIM + DIM + 6] = 50.0; mu[S_DIM + DIM + 10] = 50.0; mu[12] = 1.0
            elif c == "COND_IS_EVEN": mu[S_DIM + DIM + 7] = 50.0; mu[S_DIM + DIM + 11] = 50.0; mu[10] = 1.0
            elif c == "OP_MUL2": mu[S_DIM + DIM + 3] = 50.0; mu[10] = 1.0
            elif c == "LIST_APPEND": mu[S_DIM + DIM + 2] = 50.0; mu[10] = 1.0
            elif c == "BLOCK_END": mu[S_DIM + DIM + 10] = -50.0; mu[S_DIM + DIM + 11] = -50.0; mu[10] = 1.0
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{c}", use_diag=True)
            self.forest.register(node)
            self.observer.protected_ids.add(node.id)
            self.observer.meta_forest.register(HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True))

    def plan(self, current_state, goal_state, mask, max_steps=15) -> Optional[HFN]:
        visited = set()
        def solve(state, path_nodes, steps_left):
            s_bytes = state.tobytes()
            if s_bytes in visited: return None
            visited.add(s_bytes)
            dist = np.linalg.norm(goal_state[mask] - state[mask]) if np.any(mask) else 0.0
            if dist < 0.1: return self._fold_nodes(path_nodes) if path_nodes else None
            if steps_left <= 0: return None
            query_mu = np.zeros(self.m_dim); query_mu[:S_DIM] = state; query_mu[S_DIM + DIM:] = (goal_state - state) * 50.0
            candidates = self.retriever.retrieve(HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="p"), k=15)
            def score(n): 
                w = self.observer.get_weight(n.id)
                d = np.linalg.norm(n.mu[:S_DIM] - state)
                return (d / (w + 1e-6)) + (0.5 * len(path_nodes))
            candidates.sort(key=score)
            for rule in candidates:
                if self.observer.get_weight(rule.id) < 0.05: continue
                concept = self.renderer._get_concept(rule)
                if concept == "RETURN" and np.linalg.norm(goal_state[mask & (np.arange(S_DIM)!=0)] - state[mask & (np.arange(S_DIM)!=0)]) > 0.1: continue
                next_state = state.copy()
                if concept == "RETURN": next_state[0] = 1.0
                elif concept == "LIST_INIT": next_state[12] = 1.0; next_state[1] = 1.0
                elif concept == "FOR_LOOP": next_state[6] += 1.0; next_state[10] = 1.0
                elif concept == "COND_IS_EVEN": next_state[7] += 1.0; next_state[11] = 1.0
                elif concept == "BLOCK_END": 
                    if next_state[11] == 1.0: next_state[11] = 0.0
                    elif next_state[10] == 1.0: next_state[10] = 0.0
                elif concept == "OP_MUL2": next_state[3] += 1.0
                elif concept == "ITEM_ACCESS": next_state[10] = 1.0
                elif concept == "LIST_APPEND": next_state[2] += 1.0
                elif concept == "OP_ADD": next_state[3] += 0.5
                elif concept == "VAR_INP": next_state[3] = 1.0
                res = solve(next_state, path_nodes + [rule], steps_left - 1)
                if res is not None: return res
            return None
        return solve(current_state, [], max_steps)

    def _fold_nodes(self, nodes: List[HFN]) -> HFN:
        if not nodes: return None
        current = nodes[0]
        for n in nodes[1:]:
            p_mu = (current.mu + n.mu) / 2.0; p_mu[S_DIM + DIM:] = current.mu[S_DIM + DIM:] + n.mu[S_DIM + DIM:]
            p = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=f"robust_compose({current.id}+{n.id})", use_diag=True)
            p.add_child(current); p.add_child(n); current = p
        return current

    def execute_intent(self, prompt: str):
        print(f"\nIntent: '{prompt}'"); s_goal, mask = self.oracle.get_constraints(prompt)
        print(f"    [CONSTRAINTS] Goal Vector: {np.round(s_goal, 2)}")
        root_node = self.plan(np.zeros(S_DIM), s_goal, mask)
        if root_node:
            code = self.renderer.render(root_node); print(f"    [AST OUTPUT] Successfully rendered AST code:\n{code}")
            try: ast.parse(code); print("    [VALIDATION] AST is syntactically valid.")
            except Exception as e: print(f"    [FAIL] Rendered code is invalid: {e}")
        else: print("    [FAIL] Planner could not satisfy constraints.")

def run_experiment():
    print("--- SP53: Experiment 29 — Robust Intent Inference and AST Synthesis ---\n")
    agent = RobustIntentAgent()
    agent.execute_intent("Double all numbers")
    agent.execute_intent("Filter even numbers")

if __name__ == "__main__":
    run_experiment()
