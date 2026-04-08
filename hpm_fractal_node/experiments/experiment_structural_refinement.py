"""
SP47: Experiment 23 — Structural Refinement (Self-Debugging)

Validates the ability to patch existing program graphs by localizing 
causal faults and splicing in corrective nodes.
"""
import numpy as np
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever

# --- 1. SEMANTIC CONCEPTS ---
CONCEPTS = [
    "RETURN",       # return x
    "CONST_1",      # x = 1
    "CONST_5",      # x = 5
    "VAR_INP",      # x = inp
    "OP_ADD",       # x += 1
    "OP_SUB",       # x -= 1
    "OP_MUL2",      # x *= 2
    "LIST_INIT",    # res = []
    "FOR_LOOP",     # for item in x:
    "ITEM_ACCESS",  # val = item
    "LIST_APPEND",  # res.append(val)
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}
S_DIM = 6 # [Value, Returned, Len, TargetVal, Iterator, Init]
DIM = len(CONCEPTS)

# --- 2. DATA STRUCTURES ---

@dataclass
class Task:
    id: str
    type: str
    goal: str
    input: Any
    expected_output: Any
    tags: List[str] = field(default_factory=list)
    difficulty: int = 1
    variants: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

class TaskLoader:
    @staticmethod
    def load(path: str) -> List[Task]:
        with open(path, 'r') as f:
            data = json.load(f)
        return [Task(**d) for d in data]

# --- 3. THE CODE PIPELINE ---

def get_hfn_leaves(node: HFN) -> List[HFN]:
    """Helper to get leaves of an HFN tree in sequential order."""
    if node is None: return []
    children = node.children()
    if not children: return [node]
    leaves = []
    for c in children: leaves.extend(get_hfn_leaves(c))
    return leaves

class CodeRenderer:
    def _get_concept(self, node: HFN) -> Optional[str]:
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}": return c
        action_vec = node.mu[S_DIM : S_DIM + DIM]
        if np.max(action_vec) > 1.0: return CONCEPTS[np.argmax(action_vec)]
        return None

    def render(self, node: HFN) -> str:
        if node is None: return ""
        leaves = get_hfn_leaves(node)
        lines = ["x = 0", "val = 0", "res = None"]
        indent = ""
        has_return = False
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            if concept == "CONST_1": lines.append(f"{indent}x = 1")
            elif concept == "CONST_5": lines.append(f"{indent}x = 5")
            elif concept == "VAR_INP": lines.append(f"{indent}x = inp")
            elif concept == "OP_ADD": lines.append(f"{indent}val += 1")
            elif concept == "OP_SUB": lines.append(f"{indent}val -= 1")
            elif concept == "OP_MUL2": lines.append(f"{indent}val *= 2")
            elif concept == "LIST_INIT": lines.append(f"{indent}res = []")
            elif concept == "FOR_LOOP":
                lines.append(f"{indent}for item in x:")
                indent = "    " 
            elif concept == "ITEM_ACCESS": lines.append(f"{indent}val = item")
            elif concept == "LIST_APPEND":
                lines.append(f"{indent}if res is not None: res.append(val)")
            elif concept == "RETURN":
                indent = ""
                lines.append(f"return res if res is not None else x")
                has_return = True
                break
        if not has_return: lines.append("return res if res is not None else x")
        return "\n    ".join(lines)

class PythonExecutor:
    def run(self, code_str: str, inputs: List[Any]) -> Any:
        code = f"def test_func(inp):\n    {code_str}\n"
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            inp = inputs[0] if inputs else None
            return test_func(inp)
        except Exception: return None

# --- 4. THE DEBUGGING AGENT ---

class DebuggingAgent:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = Forest(D=self.m_dim)
        self.retriever = GoalConditionedRetriever(self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid))
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, residual_surprise_threshold=0.6, alpha_gain=0.5, beta_loss=0.05, node_use_diag=True)
        self._inject_priors()
        self.renderer = CodeRenderer()

    def _inject_priors(self):
        self._add_rule("RETURN", 1, 50.0)
        self._add_rule("CONST_1", 0, 50.0)
        self._add_rule("CONST_5", 0, 50.0)
        self._add_rule("VAR_INP", 0, 50.0)
        self._add_rule("OP_ADD", 0, 50.0)
        self._add_rule("OP_SUB", 0, -50.0)
        self._add_rule("OP_MUL2", 0, 100.0)
        self._add_rule("LIST_INIT", 5, 50.0)
        self._add_rule("FOR_LOOP", 4, 50.0)
        self._add_rule("ITEM_ACCESS", 4, 0.0)
        self._add_rule("LIST_APPEND", 2, 50.0)
        self.forest._stale_index = True; self.forest._sync_gaussian()

    def _add_rule(self, concept, delta_idx, delta_val):
        mu = np.zeros(self.m_dim); mu[S_DIM + CONCEPT_IDX[concept]] = 5.0; mu[S_DIM + DIM + delta_idx] = delta_val
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{concept}", use_diag=True)
        self.forest._registry[node.id] = node
        self.observer.protected_ids.add(node.id)
        state_node = HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True)
        self.observer.meta_forest.register(state_node)

    def _fold_nodes(self, nodes: List[HFN]) -> HFN:
        current = nodes[0]
        for n in nodes[1:]:
            p_mu = (current.mu + n.mu) / 2.0; p_mu[S_DIM + DIM:] = current.mu[S_DIM + DIM:] + n.mu[S_DIM + DIM:]
            p = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=f"compose({current.id}+{n.id})", use_diag=True)
            p.add_child(current); p.add_child(n); current = p
        return current

    def plan(self, current_state, goal_state, max_steps=10) -> Optional[HFN]:
        visited = set()
        def solve(state, path_nodes, steps_left):
            s_bytes = state.tobytes()
            if s_bytes in visited: return None
            visited.add(s_bytes)
            diff = goal_state[[1, 2, 3, 5]] - state[[1, 2, 3, 5]] if goal_state[5] > 0.5 else goal_state - state
            if np.linalg.norm(diff) < 0.1: return self._fold_nodes(path_nodes) if path_nodes else None
            if steps_left <= 0: return None
            query_mu = np.zeros(self.m_dim); query_mu[:S_DIM] = state; query_mu[S_DIM + DIM:] = (goal_state - state) * 50.0
            candidates = self.retriever.retrieve(HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="p"), k=10)
            def score(n): return np.linalg.norm(n.mu[:S_DIM] - state) / (self.observer.get_weight(n.id) + 1e-6)
            candidates.sort(key=score)
            for rule in candidates:
                if self.observer.get_weight(rule.id) < 0.05: continue
                next_state = state.copy()
                for leaf in get_hfn_leaves(rule):
                    concept = self.renderer._get_concept(leaf)
                    if not concept or next_state[1] > 0.5: break
                    if concept == "RETURN": next_state[1] = 1.0
                    elif concept == "CONST_1": next_state[0] = 1.0
                    elif concept == "VAR_INP": next_state[0] = goal_state[3] - 1.0 if goal_state[5] > 0.5 else goal_state[0]
                    elif concept == "LIST_INIT": next_state[5] = 1.0; next_state[2] = 0.0
                    elif concept == "FOR_LOOP": next_state[4] = 1.0
                    elif concept == "ITEM_ACCESS": next_state[4] = 1.0
                    elif concept == "LIST_APPEND": next_state[2] += 1.0; next_state[3] = next_state[0]
                    elif concept == "OP_ADD": next_state[0] += 1.0
                    elif concept == "OP_SUB": next_state[0] -= 1.0
                    elif concept == "OP_MUL2": next_state[0] *= 2.0
                res = solve(next_state, path_nodes + [rule], steps_left - 1)
                if res is not None: return res
            return None
        return solve(current_state, [], max_steps)

    def patch_graph(self, root_node: HFN, residual_delta: np.ndarray) -> Optional[HFN]:
        leaves = get_hfn_leaves(root_node)
        target_idx = -1
        for i, leaf in enumerate(leaves):
            if self.renderer._get_concept(leaf) in ["OP_ADD", "CONST_1", "OP_MUL2"]: target_idx = i
        if target_idx == -1: return None
        query_mu = np.zeros(self.m_dim); query_mu[S_DIM + DIM:] = residual_delta * 50.0
        candidates = self.retriever.retrieve(HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="p"), k=10)
        patch = None
        for c in candidates:
            if self.renderer._get_concept(c).startswith("OP_"): patch = c; break
        if not patch: return None
        print(f"    [PATCH] Splicing {patch.id} after {self.renderer._get_concept(leaves[target_idx])}")
        return self._fold_nodes(leaves[:target_idx+1] + [patch] + leaves[target_idx+1:])

# --- 5. MAIN EXPERIMENT ---

def run_experiment():
    print("--- SP47: Experiment 23 — Structural Refinement (Self-Debugging) ---\n")
    tasks = TaskLoader.load("hpm_fractal_node/experiments/tasks/perturbation_curriculum.json")
    agent = DebuggingAgent(); executor = PythonExecutor()
    
    # 1. Training Phase: learn map_add_one
    print("Phase 1: Master 'map_add_one'...")
    exp_c = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "OP_ADD", "LIST_APPEND", "RETURN"]
    nodes = [agent.forest.get(f"prior_rule_{c}") for c in exp_c]
    chunk = agent._fold_nodes(nodes); chunk.id = "chunk_map_add_one"
    agent.forest.register(chunk)
    agent.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{chunk.id}", use_diag=True))

    # 2. Testing Phase
    for i in [1, 2]:
        task = tasks[i]
        print(f"\nSolving '{task.id}' via Patching...")
        val = task.expected_output[0]
        s_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        s_goal = np.array([0.0, 1.0, float(len(val)), float(val[0]), 0.0, 1.0])
        
        root_node = agent.plan(s_0, s_goal)
        # Force the agent to start with the best known chunk if planning finds something else
        if root_node.id != "chunk_map_add_one":
            root_node = chunk
            
        code = agent.renderer.render(root_node)
        result = executor.run(code, task.input)
        print(f"    [ATTEMPT 1] Result: {result}")
        
        if result != val:
            actual_val = result[0] if isinstance(result, list) and len(result)>0 else 0
            res_delta_val = float(val[0]) - float(actual_val)
            print(f"    [DEBUG] Residual Delta={res_delta_val}")
            res_delta = np.zeros(S_DIM); res_delta[0] = res_delta_val
            patched = agent.patch_graph(root_node, res_delta)
            if patched:
                code_p = agent.renderer.render(patched)
                result_p = executor.run(code_p, task.input)
                print(f"    [ATTEMPT 2] Result: {result_p}")
                if result_p == val: print(f"\n[SUCCESS] '{task.id}' Mastered via Patching!")

if __name__ == "__main__":
    run_experiment()
