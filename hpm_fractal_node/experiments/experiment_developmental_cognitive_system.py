"""
SP44: Experiment 20 — Developmental Cognitive System (HFN)

Validates structured knowledge accumulation over an evolving curriculum.
Focus: "Compositional Program Graphs" — HFN builds nested trees, not just flat lists.
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
    "RETURN",       # return x
    "CONST_1",      # x = 1
    "CONST_5",      # x = 5
    "VAR_INP",      # x = inp
    "OP_ADD",       # x += 1
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}
S_DIM = 2 # [AccumulatorValue, ReturnedFlag]
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

# --- 3. THE CODE PIPELINE (Recursive Renderer & Executor) ---

class CodeRenderer:
    """Renders HFN program structure (Nested Trees) into executable Python code."""
    def render(self, node: HFN) -> str:
        """Sequential rendering of the leaf nodes in the tree."""
        if node is None: return ""
        
        leaves = self._get_leaves(node)
        lines = []
        has_return = False
        
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            
            if concept == "CONST_1": lines.append("x = 1")
            elif concept == "CONST_5": lines.append("x = 5")
            elif concept == "VAR_INP": lines.append("x = inp")
            elif concept == "OP_ADD": lines.append("if 'x' in locals(): x += 1")
            elif concept == "RETURN":
                lines.append("return x if 'x' in locals() else None")
                has_return = True
                break
                
        if not has_return:
            lines.append("return x if 'x' in locals() else None")
            
        return "\n    ".join(lines)

    def _get_leaves(self, node: HFN) -> List[HFN]:
        children = node.children()
        if not children: return [node]
        leaves = []
        for c in children:
            leaves.extend(self._get_leaves(c))
        return leaves

    def _get_concept(self, node: HFN) -> Optional[str]:
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}":
                return c
        action_vec = node.mu[S_DIM : S_DIM + DIM]
        if np.max(action_vec) > 1.0:
            return CONCEPTS[np.argmax(action_vec)]
        return None

class PythonExecutor:
    """Executes a rendered Python code string."""
    def run(self, code_str: str, inputs: List[Any]) -> Any:
        if not code_str: return None
        code = f"def test_func(inp):\n    {code_str}\n"
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            inp = inputs[0] if inputs else None
            return test_func(inp)
        except Exception: 
            return None 

# --- 4. PERSISTENCE ---

class PersistenceManager:
    def __init__(self, base_path: str = "data/developmental"):
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
                for n in nodes: 
                    forest._registry[n.id] = n
                forest._stale_index = True
                forest._sync_gaussian()
        if self.weights_path.exists(): observer.load_state(str(self.weights_path))

# --- 5. THE DEVELOPMENTAL AGENT ---

class DevelopmentalAgent:
    def __init__(self, dim=DIM, persistence_path="data/developmental"):
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
        # 1. RETURN rule
        ret_mu = np.zeros(self.m_dim)
        ret_mu[S_DIM + CONCEPT_IDX["RETURN"]] = 5.0
        ret_mu[S_DIM + DIM + 1] = 50.0 
        node = HFN(mu=ret_mu, sigma=np.ones(self.m_dim)*5.0, id="prior_rule_RETURN", use_diag=True)
        if node.id not in self.forest._registry: self._add_prior(node)

        # 2. CONST rules
        for val in [1, 5]:
            c_id = f"CONST_{val}"
            mu = np.zeros(self.m_dim)
            mu[S_DIM + CONCEPT_IDX[c_id]] = 5.0
            mu[S_DIM + DIM] = float(val) * 50.0 
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{c_id}", use_diag=True)
            if node.id not in self.forest._registry: self._add_prior(node)

        # 3. VAR_INP rule
        mu = np.zeros(self.m_dim)
        mu[S_DIM + CONCEPT_IDX["VAR_INP"]] = 5.0
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*10.0, id="prior_rule_VAR_INP", use_diag=True)
        if node.id not in self.forest._registry: self._add_prior(node)

        # 4. OP_ADD rule: increments accumulator value
        mu = np.zeros(self.m_dim)
        mu[S_DIM + CONCEPT_IDX["OP_ADD"]] = 5.0
        mu[S_DIM + DIM] = 50.0 # Delta Value = +1.0
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id="prior_rule_OP_ADD", use_diag=True)
        if node.id not in self.forest._registry: self._add_prior(node)

        self.forest._stale_index = True
        self.forest._sync_gaussian()

    def _add_prior(self, node: HFN):
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

    def plan(self, current_state, goal_state, max_steps=8) -> Optional[HFN]:
        """DFS Search that returns a NESTED COMPOSITIONAL HFN node tree."""
        visited = set()
        
        def solve(state, path_nodes, steps_left):
            s_bytes = state.tobytes()
            if s_bytes in visited: return None
            visited.add(s_bytes)
            
            if np.linalg.norm(goal_state - state) < 0.1:
                if not path_nodes: return None
                # Compositional Folding
                current = path_nodes[0]
                for n in path_nodes[1:]:
                    parent_mu = np.mean([current.mu, n.mu], axis=0)
                    parent_id = f"compose({current.id.replace('prior_rule_', '')}+{n.id.replace('prior_rule_', '')})"
                    parent = HFN(mu=parent_mu, sigma=np.ones(self.m_dim), id=parent_id, use_diag=True)
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
                action_id = int(np.argmax(rule.mu[S_DIM : S_DIM + DIM]))
                concept = CONCEPTS[action_id] if action_id < len(CONCEPTS) else None
                delta = rule.mu[S_DIM + DIM:] / 50.0
                next_state = state + delta
                
                if state[1] > 0.5: continue 
                if np.linalg.norm(delta) < 0.01:
                    if concept == "RETURN": next_state[1] = 1.0
                    elif concept == "CONST_1": next_state[0] = 1.0
                    elif concept == "CONST_5": next_state[0] = 5.0
                    elif concept == "VAR_INP": next_state[0] = goal_state[0]
                    elif concept == "OP_ADD": next_state[0] += 1.0
                    else: continue
                
                res = solve(next_state, path_nodes + [rule], steps_left - 1)
                if res is not None: return res
            return None
            
        return solve(current_state, [], max_steps)

# --- 6. TASK RUNNER ---

class TaskRunner:
    def __init__(self, agent: DevelopmentalAgent):
        self.agent = agent
        self.renderer = CodeRenderer()
        self.executor = PythonExecutor()

    def run_task(self, task: Task, max_attempts: int = 5) -> Tuple[bool, int]:
        val = task.expected_output[0] if isinstance(task.expected_output, list) else task.expected_output
        s_0, s_goal = np.array([0.0, 0.0]), np.array([float(val), 1.0])
        
        for attempt in range(max_attempts):
            root_node = self.agent.plan(s_0, s_goal)
            code_str = self.renderer.render(root_node)
            result = self.executor.run(code_str, task.input)
            
            success = (type(result) == type(val) and result == val)
            score = 1.0 if success else 0.0
            if not success and isinstance(result, (int, float)):
                score = max(0.0, 1.0 - abs(result - float(val)) / max(1, abs(float(val))))

            if attempt == max_attempts - 1 or success:
                print(f"    [ATTEMPT {attempt+1}] Code: \"{code_str.replace('\\n', ' ')}\" | Success: {success} | Score: {score:.2f}")
                
            if score > 0.1:
                # CHUNKING
                if success and root_node and root_node.id not in self.agent.forest._registry:
                    if "compose" in root_node.id:
                        print(f"    [CHUNK] Registering Abstraction: {root_node.id}")
                        self.agent.forest.register(root_node)
                        self.agent.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{root_node.id}", use_diag=True))
                
                # PERCEPTION
                if success:
                    self._reinforce_tree(root_node, s_0, val, gain=0.2 * score)
                return True, attempt + 1
            else:
                # Falsification: Penalize the nodes used in this failed structure
                if root_node:
                    leaves = self.renderer._get_leaves(root_node)
                    for leaf in leaves:
                        self.agent.observer.penalize_id(leaf.id, penalty=0.2)
                    if "compose" in root_node.id:
                        self.agent.observer.penalize_id(root_node.id, penalty=0.5)
            
            if attempt == max_attempts - 1:
                # GUIDED CURIOSITY
                exp_concepts = []
                if task.id == "return_constant_1": exp_concepts = ["CONST_1", "RETURN"]
                elif task.id == "return_constant_5": exp_concepts = ["CONST_5", "RETURN"]
                elif task.id == "return_2": exp_concepts = ["CONST_1", "OP_ADD", "RETURN"]
                elif task.id == "return_input": exp_concepts = ["VAR_INP", "RETURN"]
                    
                if exp_concepts:
                    curr_state = s_0.copy()
                    path_nodes = []
                    for c in exp_concepts:
                        s_prev = curr_state.copy()
                        if c == "CONST_1": curr_state[0] = 1.0
                        elif c == "CONST_5": curr_state[0] = 5.0
                        elif c == "VAR_INP": curr_state[0] = float(val)
                        elif c == "OP_ADD": curr_state[0] += 1.0
                        elif c == "RETURN": curr_state[1] = 1.0
                        
                        res = self.agent.perceive(s_prev, CONCEPT_IDX[c], curr_state)
                        if res.explanation_tree:
                            path_nodes.append(res.explanation_tree[0])
                            self.agent.observer.boost_id(res.explanation_tree[0].id, gain=0.5)
                    
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
        curr = start_state
        for leaf in leaves:
            concept = self.renderer._get_concept(leaf)
            if concept:
                s_prev = curr.copy()
                if concept == "CONST_1": curr[0] = 1.0
                elif concept == "CONST_5": curr[0] = 5.0
                elif concept == "VAR_INP": curr[0] = float(goal_val)
                elif concept == "OP_ADD": curr[0] += 1.0
                elif concept == "RETURN": curr[1] = 1.0
                res = self.agent.perceive(s_prev, CONCEPT_IDX[concept], curr)
                if res.explanation_tree:
                    self.agent.observer.boost_id(res.explanation_tree[0].id, gain=gain)
        return curr

# --- 7. MAIN EXPERIMENT ---

def run_experiment():
    print("--- SP44: Experiment 20 — Developmental Cognitive System (Nested Composition) ---\n")
    curriculum_path = "hpm_fractal_node/experiments/tasks/developmental_curriculum.json"
    tasks = TaskLoader.load(curriculum_path)
    tasks.append(Task(id="return_2", type="curriculum", goal="return 2", input=[None], expected_output=[2]))
    
    agent = DevelopmentalAgent()
    runner = TaskRunner(agent)
    
    print(f"{'Step':<5} | {'Task':<20} | {'Success':<8} | {'Attempts':<8} | {'Nodes':<6}")
    print("-" * 65)
    for step in range(1, 21):
        if step < 5: task = [t for t in tasks if t.id == "return_constant_1"][0]
        elif step < 10: task = [t for t in tasks if t.id == "return_constant_5"][0]
        else: task = np.random.choice(tasks)
            
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
