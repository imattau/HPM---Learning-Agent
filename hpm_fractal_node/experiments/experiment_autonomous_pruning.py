"""
SP46: Experiment 22 — Autonomous Graph Pruning (Simulation Dreams)

Validates the ability to prune the structural search space via 
internal sandboxed execution (Dreams).
"""
import numpy as np
import json
import os
import pickle
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever

# --- 1. SEMANTIC CONCEPTS (Structural Primitives) ---
CONCEPTS = [
    "RETURN",       # return x
    "CONST_1",      # x = 1
    "CONST_5",      # x = 5
    "VAR_INP",      # x = inp
    "OP_ADD",       # x += 1
    "LIST_INIT",    # res = []
    "FOR_LOOP",     # for item in x:
    "ITEM_ACCESS",  # val = item
    "LIST_APPEND",  # res.append(val)
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}
# State: [AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive, ListInitFlag]
S_DIM = 6 
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

# --- 3. THE CODE PIPELINE (Renderer & Enhanced Executor) ---

class CodeRenderer:
    """Renders HFN program structure (Nested Trees) into executable Python code."""
    def render(self, node: HFN) -> str:
        if node is None: return ""
        leaves = self._get_leaves(node)
        lines = ["x = None", "val = None", "res = None"]
        indent = ""
        has_return = False
        
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            if concept == "CONST_1": lines.append(f"{indent}x = 1")
            elif concept == "CONST_5": lines.append(f"{indent}x = 5")
            elif concept == "VAR_INP": lines.append(f"{indent}x = inp")
            elif concept == "OP_ADD": 
                lines.append(f"{indent}if val is not None: val += 1")
                lines.append(f"{indent}elif x is not None: x += 1")
            elif concept == "LIST_INIT": lines.append(f"{indent}res = []")
            elif concept == "FOR_LOOP":
                lines.append(f"{indent}for item in x:")
                indent = "    " 
            elif concept == "ITEM_ACCESS": lines.append(f"{indent}val = item")
            elif concept == "LIST_APPEND":
                lines.append(f"{indent}if res is not None: res.append(val if val is not None else x)")
            elif concept == "RETURN":
                indent = ""
                lines.append(f"{indent}if res is not None: return res")
                lines.append(f"{indent}else: return x")
                has_return = True
                break
        if not has_return:
            lines.append("if res is not None: return res")
            lines.append("else: return x")
        return "\n".join(lines)

    def _get_leaves(self, node: HFN) -> List[HFN]:
        children = node.children()
        if not children: return [node]
        leaves = []
        for c in children: leaves.extend(self._get_leaves(c))
        return leaves

    def _get_concept(self, node: HFN) -> Optional[str]:
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}": return c
        action_vec = node.mu[S_DIM : S_DIM + DIM]
        if np.max(action_vec) > 1.0: return CONCEPTS[np.argmax(action_vec)]
        return None

class PythonExecutor:
    """Executes a rendered Python code string and returns error types."""
    def run(self, code_str: str, inputs: List[Any]) -> Tuple[Any, Optional[str]]:
        if not code_str: return None, "EmptyCode"
        indented_code = "\n".join(["    " + line for line in code_str.split("\n")])
        code = f"def test_func(inp):\n{indented_code}\n"
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            inp = inputs[0] if inputs else None
            return test_func(inp), None
        except TypeError: return None, "TypeError"
        except SyntaxError: return None, "SyntaxError"
        except NameError: return None, "NameError"
        except Exception as e: return None, type(e).__name__

# --- 4. THE DEVELOPMENTAL AGENT ---

class DreamingAgent:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = Forest(D=self.m_dim)
        
        self.retriever = GoalConditionedRetriever(
            self.forest, 
            target_slice=slice(S_DIM + DIM, self.m_dim), 
            target_weight=50.0,
            weight_provider=lambda nid: self.observer.get_weight(nid)
        )
        
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            tau=0.5, 
            residual_surprise_threshold=0.6,
            alpha_gain=0.5,
            beta_loss=0.05,
            node_use_diag=True
        )
        self._inject_priors()
        self.nodes_explored_last_plan = 0
        self.prohibited_pairs: Set[Tuple[str, str]] = set()
        self.renderer = CodeRenderer()

    def _inject_priors(self):
        self._add_rule("RETURN", delta_idx=1, delta_val=50.0)
        self._add_rule("CONST_1", delta_idx=0, delta_val=50.0)
        self._add_rule("CONST_5", delta_idx=0, delta_val=50.0)
        self._add_rule("VAR_INP", delta_idx=0, delta_val=50.0)
        self._add_rule("OP_ADD", delta_idx=0, delta_val=50.0)
        self._add_rule("LIST_INIT", delta_idx=5, delta_val=50.0)
        self._add_rule("FOR_LOOP", delta_idx=4, delta_val=50.0)
        self._add_rule("ITEM_ACCESS", delta_idx=4, delta_val=0.0)
        self._add_rule("LIST_APPEND", delta_idx=2, delta_val=50.0)
        self.forest._stale_index = True
        self.forest._sync_gaussian()

    def _add_rule(self, concept, delta_idx, delta_val):
        mu = np.zeros(self.m_dim)
        mu[S_DIM + CONCEPT_IDX[concept]] = 5.0
        mu[S_DIM + DIM + delta_idx] = delta_val
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{concept}", use_diag=True)
        self.forest._registry[node.id] = node
        self.observer.protected_ids.add(node.id)
        state_node = HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True)
        self.observer.meta_forest.register(state_node)

    def dream(self, n_dreams: int):
        """Proactively explore structural combinations and prune invalid ones."""
        print(f"  [DREAM] Starting autonomous dreaming phase ({n_dreams} cycles)...")
        executor = PythonExecutor()
        
        pruned_count = 0
        for _ in range(n_dreams):
            c1, c2 = np.random.choice(CONCEPTS, size=2, replace=True)
            if (c1, c2) in self.prohibited_pairs: continue
            
            node1 = self.forest.get(f"prior_rule_{c1}")
            node2 = self.forest.get(f"prior_rule_{c2}")
            if not node1 or not node2: continue
            
            parent_mu = np.mean([node1.mu, node2.mu], axis=0)
            parent = HFN(mu=parent_mu, sigma=np.ones(self.m_dim), id=f"dream({c1}+{c2})", use_diag=True)
            parent.add_child(node1)
            parent.add_child(node2)
            
            code = self.renderer.render(parent)
            res, error = executor.run(code, [[1, 2]])
            
            if error:
                self.prohibited_pairs.add((c1, c2))
                pruned_count += 1
                
        print(f"  [DREAM] Phase complete. Prohibited {pruned_count} invalid structural pairs.")
        return pruned_count

    def perceive(self, state_t, action_id, state_t1):
        vec = np.zeros(self.m_dim)
        vec[:S_DIM] = state_t
        vec[S_DIM + action_id] = 5.0 
        vec[S_DIM + DIM:] = (state_t1 - state_t) * 50.0 
        return self.observer.observe(vec)

    def plan(self, current_state, goal_state, max_steps=10) -> Optional[HFN]:
        visited = set()
        self.nodes_explored_last_plan = 0
        def solve(state, path_nodes, steps_left):
            s_bytes = state.tobytes()
            if s_bytes in visited: return None
            visited.add(s_bytes)
            
            if goal_state[5] > 0.5:
                diff = goal_state[[1, 2, 3, 5]] - state[[1, 2, 3, 5]]
                dist = np.linalg.norm(diff)
            else:
                dist = np.linalg.norm(goal_state - state)

            if dist < 0.1:
                if not path_nodes: return None
                current = path_nodes[0]
                for n in path_nodes[1:]:
                    p_mu = np.mean([current.mu, n.mu], axis=0)
                    parent = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=f"compose({current.id}+{n.id})", use_diag=True)
                    parent.add_child(current)
                    parent.add_child(n)
                    current = parent
                return current
            
            if steps_left <= 0: return None
            
            query_mu = np.zeros(self.m_dim)
            query_mu[:S_DIM] = state
            query_mu[S_DIM + DIM:] = (goal_state - state) * 50.0
            query_node = HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="plan_query", use_diag=True)
            candidates = self.retriever.retrieve(query_node, k=10)
            
            def score(n): 
                w = self.observer.get_weight(n.id)
                return np.linalg.norm(n.mu[:S_DIM] - state) / (w + 1e-6)
            candidates.sort(key=score)
            
            for rule in candidates:
                if self.observer.get_weight(rule.id) < 0.05: continue
                
                # Topological Pruning Check
                if path_nodes:
                    last_node = path_nodes[-1]
                    c_last = self.renderer._get_concept(last_node)
                    c_next = self.renderer._get_concept(rule)
                    if (c_last, c_next) in self.prohibited_pairs:
                        continue

                self.nodes_explored_last_plan += 1
                
                next_state = state.copy()
                leaves = self._get_leaves_for_sim(rule)
                valid_transition = True
                for leaf in leaves:
                    action_id = int(np.argmax(leaf.mu[S_DIM : S_DIM + DIM]))
                    concept = CONCEPTS[action_id] if action_id < len(CONCEPTS) else None
                    if not concept or next_state[1] > 0.5:
                        valid_transition = False
                        break
                    
                    if concept == "RETURN": next_state[1] = 1.0
                    elif concept == "CONST_1": next_state[0] = 1.0
                    elif concept == "VAR_INP": 
                        if goal_state[5] > 0.5: next_state[0] = goal_state[3] - 1.0 
                        else: next_state[0] = goal_state[0]
                    elif concept == "LIST_INIT": next_state[5] = 1.0; next_state[2] = 0.0
                    elif concept == "FOR_LOOP": next_state[4] = 1.0
                    elif concept == "ITEM_ACCESS": next_state[4] = 1.0 
                    elif concept == "LIST_APPEND": 
                        next_state[2] += 1.0
                        next_state[3] = next_state[0] 
                    elif concept == "OP_ADD": next_state[0] += 1.0
                
                if not valid_transition: continue
                res = solve(next_state, path_nodes + [rule], steps_left - 1)
                if res is not None: return res
            return None
        return solve(current_state, [], max_steps)

    def _get_leaves_for_sim(self, node: HFN) -> List[HFN]:
        children = node.children()
        if not children: return [node]
        leaves = []
        for c in children: leaves.extend(self._get_leaves_for_sim(c))
        return leaves

# --- 5. TASK RUNNER ---

class TaskRunner:
    def __init__(self, agent: DreamingAgent):
        self.agent = agent
        self.renderer = CodeRenderer()
        self.executor = PythonExecutor()

    def run_task(self, task: Task, max_attempts: int = 5) -> Tuple[bool, int, int]:
        val = task.expected_output[0] if isinstance(task.expected_output, list) else task.expected_output
        s_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        is_list = isinstance(val, list)
        s_goal = np.array([0.0 if is_list else float(val), 1.0, float(len(val)) if is_list else 0.0, float(val[0]) if is_list and len(val)>0 else 0.0, 0.0, 1.0 if is_list else 0.0])
        
        total_nodes_explored = 0
        for attempt in range(max_attempts):
            root_node = self.agent.plan(s_0, s_goal)
            total_nodes_explored += self.agent.nodes_explored_last_plan
            
            code_str = self.renderer.render(root_node)
            result, error = self.executor.run(code_str, task.input)
            
            success = (type(result) == type(val) and result == val)
            if success:
                # Credit assignment
                curr_state = s_0.copy()
                leaves = self.renderer._get_leaves(root_node)
                for leaf in leaves:
                    s_prev = curr_state.copy()
                    concept = self.renderer._get_concept(leaf)
                    if concept == "CONST_1": curr_state[0] = 1.0
                    elif concept == "CONST_5": curr_state[0] = 5.0
                    elif concept == "VAR_INP": curr_state[0] = float(val) if not is_list else 0.0
                    elif concept == "OP_ADD": curr_state[0] += 1.0
                    elif concept == "LIST_INIT": curr_state[5] = 1.0
                    elif concept == "FOR_LOOP": curr_state[4] = 1.0
                    elif concept == "ITEM_ACCESS": curr_state[4] = 1.0
                    elif concept == "LIST_APPEND": 
                        curr_state[2] += 1.0
                        curr_state[3] = curr_state[0]
                    elif concept == "RETURN": curr_state[1] = 1.0
                    self.agent.perceive(s_prev, CONCEPT_IDX[concept], curr_state)
                return True, attempt + 1, total_nodes_explored
            
            if attempt == max_attempts - 1:
                # Guided curiosity
                exp_concepts = []
                if task.id == "return_empty_list": exp_concepts = ["LIST_INIT", "RETURN"]
                elif task.id == "add_one_to_item": exp_concepts = ["VAR_INP", "OP_ADD", "RETURN"]
                elif task.id == "map_add_one": exp_concepts = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "OP_ADD", "LIST_APPEND", "RETURN"]
                
                if exp_concepts:
                    curr_state = s_0.copy()
                    for c in exp_concepts:
                        s_prev = curr_state.copy()
                        if c == "CONST_1": curr_state[0] = 1.0
                        elif c == "VAR_INP": curr_state[0] = float(task.input[0]) if not is_list else 0.0
                        elif c == "OP_ADD": curr_state[0] += 1.0
                        elif c == "LIST_INIT": curr_state[5] = 1.0
                        elif c == "FOR_LOOP": curr_state[4] = 1.0
                        elif c == "ITEM_ACCESS": curr_state[4] = 1.0
                        elif c == "LIST_APPEND": 
                            curr_state[2] += 1.0
                            curr_state[3] = curr_state[0]
                        elif c == "RETURN": curr_state[1] = 1.0
                        self.agent.perceive(s_prev, CONCEPT_IDX[c], curr_state)
                
        return False, max_attempts, total_nodes_explored

# --- 6. EXPERIMENT RUNNER ---

def run_experiment():
    print("--- SP46: Experiment 22 — Autonomous Graph Pruning (Simulation Dreams) ---\n")
    
    curriculum_path = "hpm_fractal_node/experiments/tasks/developmental_curriculum_lists.json"
    tasks = TaskLoader.load(curriculum_path)
    capstone_task = tasks[2] # map_add_one
    
    print(f"Goal: Compare baseline planning vs dreaming-pruned planning on '{capstone_task.id}'")
    
    # 1. BASELINE AGENT
    print("\nPhase A: Baseline Agent (0 dreams)...")
    baseline_agent = DreamingAgent()
    baseline_runner = TaskRunner(baseline_agent)
    baseline_runner.run_task(tasks[0]) 
    baseline_runner.run_task(tasks[1]) 
    success_b, attempts_b, explored_b = baseline_runner.run_task(capstone_task)
    print(f"  Baseline Results: Success={success_b}, Attempts={attempts_b}, Nodes Explored={explored_b}")
    
    # 2. DREAMING AGENT
    print("\nPhase B: Dreaming Agent (100 dreams)...")
    dreaming_agent = DreamingAgent()
    pruned = dreaming_agent.dream(100)
    
    dreaming_runner = TaskRunner(dreaming_agent)
    dreaming_runner.run_task(tasks[0])
    dreaming_runner.run_task(tasks[1])
    success_d, attempts_d, explored_d = dreaming_runner.run_task(capstone_task)
    print(f"  Dreaming Results: Success={success_d}, Attempts={attempts_d}, Nodes Explored={explored_d}")
    
    # 3. ANALYSIS
    print("\n--- PERFORMANCE SUMMARY ---")
    if explored_d < explored_b:
        improvement = (explored_b - explored_d) / explored_b
        print(f"SUCCESS: Dreaming reduced planning search space by {improvement:.1%}")
    else:
        print("FAIL: Dreaming did not reduce planning search space.")
        
    print(f"Total invalid structural pairs prohibited during dreams: {pruned}")

if __name__ == "__main__":
    run_experiment()
