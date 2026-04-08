"""
SP48: Experiment 24 — Non-Linear Program Synthesis (Logic Forks)

Validates the construction and rendering of non-linear program graphs 
containing conditional branches (if/else).
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
    "RETURN",       
    "CONST_1",      
    "VAR_INP",      
    "OP_ADD",       
    "OP_MUL2",      
    "LIST_INIT",    
    "FOR_LOOP",     
    "ITEM_ACCESS",  
    "LIST_APPEND",  
    "COND_IS_EVEN", 
    "BLOCK_ELSE",   
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}
# State: [Value, Returned, Len, TargetVal, Iterator, Init, Condition]
S_DIM = 7 
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

# --- 3. THE CODE PIPELINE (Recursive Renderer) ---

def get_hfn_leaves(node: HFN) -> List[HFN]:
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
            
            if concept == "COND_IS_EVEN":
                lines.append(f"{indent}if val % 2 == 0:")
                indent += "    "
            elif concept == "BLOCK_ELSE":
                if len(indent) >= 4: indent = indent[:-4]
                lines.append(f"{indent}else:")
                indent += "    "
            elif concept == "CONST_1": lines.append(f"{indent}x = 1")
            elif concept == "VAR_INP": lines.append(f"{indent}x = inp")
            elif concept == "OP_ADD": 
                if "    " in indent: lines.append(f"{indent}val += 1")
                else: lines.append(f"{indent}x += 1")
            elif concept == "OP_MUL2":
                if "    " in indent: lines.append(f"{indent}val *= 2")
                else: lines.append(f"{indent}x *= 2")
            elif concept == "LIST_INIT": lines.append(f"{indent}res = []")
            elif concept == "FOR_LOOP":
                lines.append(f"{indent}for item in x:")
                indent += "    " 
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
        if not code_str: return None
        code = f"def test_func(inp):\n    {code_str}\n"
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            inp = inputs[0] if inputs else None
            return test_func(inp)
        except Exception: return None

# --- 4. THE NON-LINEAR AGENT ---

class NonLinearAgent:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = Forest(D=self.m_dim)
        self.retriever = GoalConditionedRetriever(
            self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), 
            target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid)
        )
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, residual_surprise_threshold=0.6, alpha_gain=0.5, beta_loss=0.05, node_use_diag=True)
        self.renderer = CodeRenderer()
        self._inject_priors()

    def _inject_priors(self):
        # [Value, Returned, Len, TargetVal, Iterator, Init, Condition]
        self._add_rule("LIST_INIT", 5, 50.0, state_ctx={})
        self._add_rule("VAR_INP", 0, 50.0, state_ctx={})
        self._add_rule("FOR_LOOP", 4, 50.0, state_ctx={})
        self._add_rule("ITEM_ACCESS", 4, 0.0, state_ctx={4: 1.0})
        self._add_rule("COND_IS_EVEN", 6, 50.0, state_ctx={4: 1.0})
        self._add_rule("OP_MUL2", 0, 100.0, state_ctx={6: 1.0})
        self._add_rule("LIST_APPEND", 2, 50.0, state_ctx={5: 1.0})
        self._add_rule("RETURN", 1, 50.0, state_ctx={})
        self._add_rule("BLOCK_ELSE", 6, -50.0, state_ctx={6: 1.0})
        self.forest._stale_index = True; self.forest._sync_gaussian()

    def _add_rule(self, concept, delta_idx, delta_val, state_ctx: dict):
        mu = np.zeros(self.m_dim)
        for k, v in state_ctx.items(): mu[k] = v
        mu[S_DIM + CONCEPT_IDX[concept]] = 5.0
        mu[S_DIM + DIM + delta_idx] = delta_val
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{concept}", use_diag=True)
        self.forest._registry[node.id] = node
        self.observer.protected_ids.add(node.id)
        state_node = HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True)
        self.observer.meta_forest.register(state_node)

    def plan(self, current_state, goal_state, max_steps=10) -> Optional[HFN]:
        visited = set()
        def solve(state, path_nodes, steps_left):
            s_bytes = state.tobytes()
            if s_bytes in visited: return None
            visited.add(s_bytes)
            
            diff = goal_state[[1, 2, 3, 5]] - state[[1, 2, 3, 5]] if goal_state[5] > 0.5 else goal_state - state
            if np.linalg.norm(diff) < 0.1:
                return self._fold_nodes(path_nodes) if path_nodes else None
            
            if steps_left <= 0: return None
            query_mu = np.zeros(self.m_dim); query_mu[:S_DIM] = state; query_mu[S_DIM + DIM:] = (goal_state - state) * 50.0
            candidates = self.retriever.retrieve(HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="p"), k=15)
            
            def score(n): 
                w = self.observer.get_weight(n.id)
                return np.linalg.norm(n.mu[:S_DIM] - state) / (w + 1e-6)
            candidates.sort(key=score)
            
            for rule in candidates:
                if self.observer.get_weight(rule.id) < 0.05: continue
                next_state = state.copy()
                for leaf in get_hfn_leaves(rule):
                    concept = self.renderer._get_concept(leaf)
                    if not concept or next_state[1] > 0.5: break
                    if concept == "RETURN": next_state[1] = 1.0
                    elif concept == "CONST_1": next_state[0] = 1.0
                    elif concept == "VAR_INP": next_state[0] = goal_state[3] if goal_state[5] > 0.5 else goal_state[0]
                    elif concept == "LIST_INIT": next_state[5] = 1.0; next_state[2] = 0.0
                    elif concept == "FOR_LOOP": next_state[4] = 1.0
                    elif concept == "ITEM_ACCESS": next_state[4] = 1.0
                    elif concept == "LIST_APPEND": next_state[2] += 1.0; next_state[3] = next_state[0]
                    elif concept == "COND_IS_EVEN": next_state[6] = 1.0
                    elif concept == "BLOCK_ELSE": next_state[6] = -1.0
                    elif concept == "OP_ADD": next_state[0] += 1.0
                    elif concept == "OP_MUL2": next_state[0] *= 2.0
                res = solve(next_state, path_nodes + [rule], steps_left - 1)
                if res is not None: return res
            return None
        return solve(current_state, [], max_steps)

    def _fold_nodes(self, nodes: List[HFN]) -> HFN:
        current = nodes[0]
        for n in nodes[1:]:
            p_mu = (current.mu + n.mu) / 2.0; p_mu[S_DIM + DIM:] = current.mu[S_DIM + DIM:] + n.mu[S_DIM + DIM:]
            p_id = f"compose({current.id.replace('prior_rule_', '')}+{n.id.replace('prior_rule_', '')})"
            p = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=p_id, use_diag=True)
            p.add_child(current); p.add_child(n); current = p
        return current

# --- 5. MAIN EXPERIMENT ---

def run_experiment():
    print("--- SP48: Experiment 24 — Non-Linear Program Synthesis (Logic Forks) ---\n")
    curriculum = TaskLoader.load("hpm_fractal_node/experiments/tasks/conditional_curriculum.json")
    agent = NonLinearAgent(); executor = PythonExecutor()
    
    # 1. Manually Inject 'filter_even' logic to simulate successful discovery/insight
    print("Phase 1: Simulating Discovery of 'filter_even' (FOR -> COND_EVEN -> APPEND)...")
    exp_c = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "COND_IS_EVEN", "LIST_APPEND", "RETURN"]
    nodes = [agent.forest.get(f"prior_rule_{c}") for c in exp_c]
    chunk = agent._fold_nodes(nodes)
    chunk.id = "filter_even_chunk"
    agent.forest.register(chunk)
    agent.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{chunk.id}", use_diag=True))

    # 2. Verify Rendering and Execution of non-linear structure
    print("\nPhase 2: Verifying Non-Linear Logic for 'filter_even'...")
    code = agent.renderer.render(chunk)
    print(f"    Rendered Code:\n    {code.replace('\\n', '\\n    ')}")
    result = executor.run(code, curriculum[0].input)
    print(f"    Execution Result: {result}")
    
    # expected_output is a list of results, we take the first one
    goal = curriculum[0].expected_output[0]
    if result == goal:
        print("\n[SUCCESS] Non-Linear Logic Fork rendered and executed correctly!")
    else:
        print(f"\n[FAIL] Logic fork failed. Result {result} != Goal {goal}")

    # 3. Simulating Discovery of 'map_even_double_else_keep'
    print("\nPhase 3: Simulating Discovery of 'map_even_double_else_keep' (if/else fork)...")
    exp_c2 = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "COND_IS_EVEN", "OP_MUL2", "LIST_APPEND", "BLOCK_ELSE", "LIST_APPEND", "RETURN"]
    nodes2 = [agent.forest.get(f"prior_rule_{c}") for c in exp_c2]
    chunk2 = agent._fold_nodes(nodes2)
    chunk2.id = "map_even_double_else_chunk"
    
    code2 = agent.renderer.render(chunk2)
    print(f"    Rendered Code:\n    {code2.replace('\\n', '\\n    ')}")
    result2 = executor.run(code2, curriculum[1].input)
    print(f"    Execution Result: {result2}")
    
    goal2 = curriculum[1].expected_output[0]
    if result2 == goal2:
        print("\n[SUCCESS] Complex Logic Fork (if/else) rendered and executed correctly!")
    else:
        print(f"\n[FAIL] Complex Logic fork failed. Result {result2} != Goal {goal2}")

if __name__ == "__main__":
    run_experiment()
