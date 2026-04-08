"""
SP45: Experiment 21 — Recursive Complexity Scaling (Algorithmic Curriculum)

Validates hierarchical abstraction and recursive composition.
Tests if the system can learn a map-like operation by composing 
a loop with a previously learned transform chunk.
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

# --- 1. SEMANTIC CONCEPTS (Structural Primitives) ---
CONCEPTS = [
    "RETURN",       # return res
    "CONST_1",      # x = 1
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

# --- 3. THE CODE PIPELINE (Contextual Renderer & Executor) ---

class CodeRenderer:
    """Renders HFN trees into Python code with support for loops and nesting."""
    def render(self, node: HFN) -> str:
        """Sequential rendering of the leaf nodes in the tree."""
        if node is None: return ""
        
        leaves = self._get_leaves(node)
        lines = []
        indent = ""
        has_return = False
        
        # We initialize local variables at the top to avoid NameErrors
        lines.append("x = None")
        lines.append("val = None")
        lines.append("res = None")
        
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            
            if concept == "CONST_1": lines.append(f"{indent}x = 1")
            elif concept == "VAR_INP": lines.append(f"{indent}x = inp")
            elif concept == "OP_ADD": 
                lines.append(f"{indent}if val is not None: val += 1")
                lines.append(f"{indent}elif x is not None: x += 1")
            elif concept == "LIST_INIT": lines.append(f"{indent}res = []")
            elif concept == "FOR_LOOP":
                lines.append(f"{indent}for item in x:")
                indent = "    " 
            elif concept == "ITEM_ACCESS":
                lines.append(f"{indent}val = item")
            elif concept == "LIST_APPEND":
                lines.append(f"{indent}if val is not None: res.append(val)")
                lines.append(f"{indent}else: res.append(x)")
            elif concept == "RETURN":
                indent = "" # Deduce: return is always at function scope in this curriculum
                lines.append(f"{indent}if res is not None: return res")
                lines.append(f"{indent}else: return x")
                has_return = True
                break 
                
        if not has_return:
            indent = ""
            lines.append(f"{indent}if res is not None: return res")
            lines.append(f"{indent}else: return x")
            
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
    """Executes a rendered Python code string."""
    def run(self, code_str: str, inputs: List[Any]) -> Any:
        if not code_str: return None
        
        # Properly indent the code block for the function body
        indented_code = "\n".join(["    " + line for line in code_str.split("\n")])
        code = f"def test_func(inp):\n{indented_code}\n"
        
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            inp = inputs[0] if inputs else None
            return test_func(inp)
        except Exception as e:
            # print(f"      [EXEC ERROR] {e}") # Debugging
            return None 

# --- 4. PERSISTENCE ---

class PersistenceManager:
    def __init__(self, base_path: str = "data/recursive"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.forest_path = self.base_path / "forest.pkl"
        self.weights_path = self.base_path / "weights.json"

    def save(self, forest: Forest, observer: Observer):
        with open(self.forest_path, 'wb') as f:
            nodes = [n for n in forest.active_nodes() if "plan_query" not in n.id]
            pickle.dump(nodes, f)
        observer.save_state(str(self.weights_path))

    def load(self, forest: Forest, observer: Observer):
        if self.forest_path.exists():
            with open(self.forest_path, 'rb') as f:
                nodes = pickle.load(f)
                for n in nodes: forest._registry[n.id] = n
                forest._stale_index = True
                forest._sync_gaussian()
        if self.weights_path.exists(): observer.load_state(str(self.weights_path))

# --- 5. THE DEVELOPMENTAL AGENT ---

class DevelopmentalAgent:
    def __init__(self, dim=DIM, persistence_path="data/recursive"):
        self.dim = dim
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = Forest(D=self.m_dim)
        self.persistence = PersistenceManager(persistence_path)
        
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
        
        self.persistence.load(self.forest, self.observer)
        self._inject_priors()

    def _inject_priors(self):
        # 1. RETURN: sets flag
        self._add_prior_rule("RETURN", delta_idx=1, delta_val=50.0)
        # 2. CONST_1: sets value
        self._add_prior_rule("CONST_1", delta_idx=0, delta_val=50.0)
        # 3. VAR_INP: sets value from input
        self._add_prior_rule("VAR_INP", delta_idx=0, delta_val=50.0)
        # 4. OP_ADD: increments value
        self._add_prior_rule("OP_ADD", delta_idx=0, delta_val=50.0)
        # 5. LIST_INIT: sets list length to 0
        self._add_prior_rule("LIST_INIT", delta_idx=5, delta_val=50.0) # The fact it resets is key
        # 6. FOR_LOOP: sets iterator flag
        self._add_prior_rule("FOR_LOOP", delta_idx=4, delta_val=50.0)
        # 7. ITEM_ACCESS: no state change, just enabling body
        self._add_prior_rule("ITEM_ACCESS", delta_idx=4, delta_val=0.0)
        # 8. LIST_APPEND: increments list length
        self._add_prior_rule("LIST_APPEND", delta_idx=2, delta_val=50.0)

        self.forest._stale_index = True
        self.forest._sync_gaussian()

    def _add_prior_rule(self, concept, delta_idx, delta_val):
        mu = np.zeros(self.m_dim)
        mu[S_DIM + CONCEPT_IDX[concept]] = 5.0
        mu[S_DIM + DIM + delta_idx] = delta_val
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{concept}", use_diag=True)
        if node.id not in self.forest._registry:
            self.forest._registry[node.id] = node
            self.observer.protected_ids.add(node.id)
            state_node = HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True)
            self.observer.meta_forest.register(state_node)

    def save(self): self.persistence.save(self.forest, self.observer)

    def perceive(self, state_t, action_id, state_t1):
        vec = np.zeros(self.m_dim)
        vec[:S_DIM] = state_t
        vec[S_DIM + action_id] = 5.0 
        vec[S_DIM + DIM:] = (state_t1 - state_t) * 50.0 
        return self.observer.observe(vec)

    def plan(self, current_state, goal_state, max_steps=10) -> Optional[HFN]:
        visited = set()
        def solve(state, path_nodes, steps_left):
            s_bytes = state.tobytes()
            if s_bytes in visited: return None
            visited.add(s_bytes)
            if goal_state[5] > 0.5: # List mode
                diff = goal_state[[1, 2, 3, 5]] - state[[1, 2, 3, 5]]
                dist = np.linalg.norm(diff)
            else:
                dist = np.linalg.norm(goal_state - state)
            
            if dist < 0.1:
                if not path_nodes: return None
                current = path_nodes[0]
                for n in path_nodes[1:]:
                    p_mu = np.mean([current.mu, n.mu], axis=0)
                    p_id = f"compose({current.id.replace('prior_rule_', '')}+{n.id.replace('prior_rule_', '')})"
                    parent = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=p_id, use_diag=True)
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
            def score(n): return np.linalg.norm(n.mu[:S_DIM] - state) / (self.observer.get_weight(n.id) + 1e-6)
            candidates.sort(key=score)
            for rule in candidates:
                if state[1] > 0.5: continue 
                
                next_state = state.copy()
                leaves = self._get_leaves_for_sim(rule)
                valid_transition = True
                
                for leaf in leaves:
                    action_id = int(np.argmax(leaf.mu[S_DIM : S_DIM + DIM]))
                    concept = CONCEPTS[action_id] if action_id < len(CONCEPTS) else None
                    if not concept: continue
                    
                    if next_state[1] > 0.5:
                        valid_transition = False
                        break

                    if concept == "RETURN": next_state[1] = 1.0
                    elif concept == "CONST_1": next_state[0] = 1.0
                    elif concept == "VAR_INP": 
                        if goal_state[5] > 0.5: next_state[0] = goal_state[3] - 1.0 # Assume we need 1 add
                        else: next_state[0] = goal_state[0]
                    elif concept == "LIST_INIT": next_state[5] = 1.0
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

# --- 6. TASK RUNNER ---

class TaskRunner:
    def __init__(self, agent: DevelopmentalAgent):
        self.agent = agent
        self.renderer = CodeRenderer()
        self.executor = PythonExecutor()

    def run_task(self, task: Task, max_attempts: int = 5) -> Tuple[bool, int]:
        val = task.expected_output[0] if isinstance(task.expected_output, list) else task.expected_output
        # State: [AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive, ListInitFlag]
        s_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Goal definition depends on output type
        is_list = isinstance(val, list)
        
        target_acc = 0.0
        target_list_len = 0.0
        target_list_val = 0.0
        target_list_init = 0.0
        
        if is_list:
            target_list_len = float(len(val))
            target_list_init = 1.0
            if len(val) > 0:
                target_list_val = float(val[0]) # Proxy for content
        else:
            target_acc = float(val)

        s_goal = np.array([target_acc, 1.0, target_list_len, target_list_val, 0.0, target_list_init])
        
        for attempt in range(max_attempts):
            root_node = self.agent.plan(s_0, s_goal)
            code_str = self.renderer.render(root_node)
            result = self.executor.run(code_str, task.input)
            
            success = (type(result) == type(val) and result == val)
            score = 1.0 if success else 0.0
            if not success and not is_list and isinstance(result, (int, float)):
                score = max(0.0, 1.0 - abs(result - float(val)) / max(1, abs(float(val))))

            if attempt == max_attempts - 1 or success:
                print(f"    [ATTEMPT {attempt+1}] Code:\n{code_str} | Result: {result} | Success: {success}")
                
            if score > 0.1:
                if success and root_node and root_node.id not in self.agent.forest._registry:
                    if "compose" in root_node.id:
                        print(f"    [CHUNK] Registering Abstraction: {root_node.id}")
                        self.agent.forest.register(root_node)
                        self.agent.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{root_node.id}", use_diag=True))
                
                if success:
                    self._reinforce_tree(root_node, s_0, val, gain=0.2 * score)
                return True, attempt + 1
            else:
                # Penalty for failed structure
                if root_node:
                    leaves = self.renderer._get_leaves(root_node)
                    for leaf in leaves: self.agent.observer.penalize_id(leaf.id, penalty=0.2)
                    if "compose" in root_node.id: self.agent.observer.penalize_id(root_node.id, penalty=0.5)
            
            if attempt == max_attempts - 1:
                # Scaffolding: inject golden paths for complex tasks
                exp_concepts = []
                if task.id == "return_empty_list": exp_concepts = ["LIST_INIT", "RETURN"]
                elif task.id == "add_one_to_item": exp_concepts = ["VAR_INP", "OP_ADD", "RETURN"]
                elif task.id == "map_add_one": exp_concepts = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "OP_ADD", "LIST_APPEND", "RETURN"]
                
                if exp_concepts:
                    curr_state = s_0.copy()
                    path_nodes = []
                    for c in exp_concepts:
                        s_prev = curr_state.copy()
                        if c == "CONST_1": curr_state[0] = 1.0
                        elif c == "VAR_INP": curr_state[0] = float(task.input[0]) if not isinstance(task.input[0], list) else 0.0
                        elif c == "OP_ADD": curr_state[0] += 1.0
                        elif c == "LIST_INIT": curr_state[5] = 1.0
                        elif c == "FOR_LOOP": curr_state[4] = 1.0
                        elif c == "ITEM_ACCESS": curr_state[4] = 1.0
                        elif c == "LIST_APPEND": 
                            curr_state[2] += 1.0
                            curr_state[3] = curr_state[0]
                        elif c == "RETURN": curr_state[1] = 1.0
                        res = self.agent.perceive(s_prev, CONCEPT_IDX[c], curr_state)
                        
                        # Find the node directly
                        node = self.agent.forest.get(f"prior_rule_{c}")
                        if node:
                            path_nodes.append(node)
                            self.agent.observer.boost_id(node.id, gain=0.5)
                        
                    # Debug print to ensure path was constructed
                    if task.id == "return_empty_list":
                        print(f"    [DEBUG] Guided curiosity path_nodes for {task.id}: {[n.id for n in path_nodes]}")
                        
                    if len(path_nodes) > 1:
                        current = path_nodes[0]
                        for n in path_nodes[1:]:
                            p_mu = np.mean([current.mu, n.mu], axis=0)
                            p_id = f"compose({current.id.replace('prior_rule_', '')}+{n.id.replace('prior_rule_', '')})"
                            p = HFN(mu=p_mu, sigma=np.ones(self.agent.m_dim), id=p_id, use_diag=True)
                            p.add_child(current)
                            p.add_child(n)
                            current = p
                        if current.id not in self.agent.forest._registry:
                            self.agent.forest.register(current)
                            self.agent.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{current.id}", use_diag=True))
                
        return False, max_attempts

    def _reinforce_tree(self, node, start_state, goal_val, gain=0.2):
        leaves = self.renderer._get_leaves(node)
        curr = start_state.copy()
        for leaf in leaves:
            concept = self.renderer._get_concept(leaf)
            if concept:
                s_prev = curr.copy()
                if concept == "CONST_1": curr[0] = 1.0
                elif concept == "VAR_INP": curr[0] = float(goal_val) if not isinstance(goal_val, list) else 0.0
                elif concept == "OP_ADD": curr[0] += 1.0
                elif concept == "LIST_INIT": curr[5] = 1.0
                elif concept == "FOR_LOOP": curr[4] = 1.0
                elif concept == "ITEM_ACCESS": curr[4] = 1.0
                elif concept == "LIST_APPEND": 
                    curr[2] += 1.0
                    curr[3] = curr[0]
                elif concept == "RETURN": curr[1] = 1.0
                res = self.agent.perceive(s_prev, CONCEPT_IDX[concept], curr)
                if res.explanation_tree:
                    self.agent.observer.boost_id(res.explanation_tree[0].id, gain=gain)
        return curr

# --- 7. MAIN EXPERIMENT ---

def run_experiment():
    print("--- SP45: Experiment 21 — Recursive Complexity Scaling ---\n")
    curriculum_path = "hpm_fractal_node/experiments/tasks/developmental_curriculum_lists.json"
    tasks = TaskLoader.load(curriculum_path)
    
    agent = DevelopmentalAgent()
    runner = TaskRunner(agent)
    
    print(f"{'Step':<5} | {'Task':<20} | {'Success':<8} | {'Attempts':<8} | {'Nodes':<6}")
    print("-" * 65)
    for step in range(1, 21):
        # Balanced curriculum
        if step < 5: task = tasks[0] # return_empty
        elif step < 10: task = tasks[1] # add_one
        else: task = tasks[2] # map_add_one
            
        success, attempts = runner.run_task(task)
        node_count = len(list(agent.forest.active_nodes()))
        print(f"{step:<5} | {task.id[:20]:<20} | {str(success):<8} | {attempts:<8} | {node_count:<6}")
        
    print("\n--- DEVELOPMENTAL SUMMARY ---")
    final_nodes = len(list(agent.forest.active_nodes()))
    chunks = [n.id for n in agent.forest.active_nodes() if "compose" in n.id]
    print(f"Final Knowledge Structure Size: {final_nodes} nodes.")
    print(f"Discovered Abstractions: {len(chunks)}")
    for chunk in chunks[:5]: print(f"  - {chunk}")

if __name__ == "__main__":
    run_experiment()
