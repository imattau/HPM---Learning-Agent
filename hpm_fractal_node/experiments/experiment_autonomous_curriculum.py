"""
SP51: Experiment 27 — Autonomous Curriculum Generation (Self-Play & Curiosity)

Refined implementation with smarter sampling and structural novelty detection.
"""
import numpy as np
import json
import os
import time
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import threading

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
S_DIM = 9 
DIM = len(CONCEPTS)

# --- 2. THE STATE ORACLE ---

class StateOracle:
    def compute_state(self, inp: Any, out: Any) -> np.ndarray:
        s = np.zeros(S_DIM)
        s[1] = 1.0 if out is not None else 0.0
        if isinstance(out, list):
            s[2] = float(len(out))
            if len(out) > 0:
                s[3] = float(out[0])
            s[5] = 1.0
        elif isinstance(out, (int, float)):
            s[0] = float(out)
        return s

# --- 3. THE CODE PIPELINE ---

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
        if node.id.startswith("prior_rule_"):
            action_vec = node.mu[S_DIM : S_DIM + DIM]
            if np.max(action_vec) > 1.0: return CONCEPTS[np.argmax(action_vec)]
        return None

    def render(self, node: HFN) -> str:
        leaves = get_hfn_leaves(node)
        lines = ["x = 0", "val = 0", "res = None"]
        indent = ""
        has_return = False
        
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            
            if concept == "BLOCK_END":
                if len(indent) >= 4: indent = indent[:-4]
                continue
            
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
                if indent != "": lines.append(f"{indent}val += 1")
                else: lines.append(f"{indent}x += 1")
            elif concept == "OP_MUL2":
                if indent != "": lines.append(f"{indent}val *= 2")
                else: lines.append(f"{indent}x *= 2")
            elif concept == "OP_SUB":
                if indent != "": lines.append(f"{indent}val -= 1")
                else: lines.append(f"{indent}x -= 1")
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
        
        if not has_return:
            lines.append("return res if res is not None else x")
            
        return "\n    ".join(lines)

class PythonExecutor:
    def run(self, code_str: str, inp: Any, timeout: float = 0.5) -> Tuple[Any, Optional[str]]:
        code = f"def test_func(inp):\n    {code_str}\n"
        result = [None]
        error = [None]
        
        def target():
            try:
                local_ns = {}
                exec(code, {}, local_ns)
                test_func = local_ns["test_func"]
                result[0] = test_func(inp)
            except Exception as e:
                error[0] = type(e).__name__

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            return None, "Timeout"
        return result[0], error[0]

# --- 4. THE SELF-PLAY AGENT ---

class SelfPlayAgent:
    def __init__(self, cold_dir="data/knowledge_base/curiosity"):
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = TieredForest(D=self.m_dim, cold_dir=Path(cold_dir))
        self.persistence = PersistenceManager(root_dir="data/knowledge_base")
        self.experiment_id = "curiosity"
        
        self.retriever = GoalConditionedRetriever(
            self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), 
            target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid)
        )
        
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, node_use_diag=True)
        self.renderer = CodeRenderer()
        self.oracle = StateOracle()
        self.executor = PythonExecutor()
        
        self.persistence.load(self.forest, self.observer, self.experiment_id)
        if "prior_rule_RETURN" not in self.forest:
            self._inject_priors()

    def _inject_priors(self):
        for c in CONCEPTS:
            mu = np.zeros(self.m_dim)
            mu[S_DIM + CONCEPT_IDX[c]] = 5.0
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{c}", use_diag=True)
            self.forest.register(node)
            self.observer.protected_ids.add(node.id)
            state_node = HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True)
            self.observer.meta_forest.register(state_node)

    def _fold_nodes(self, nodes: List[HFN], i: int) -> HFN:
        if not nodes: return None
        current = nodes[0]
        for idx, n in enumerate(nodes[1:]):
            p_mu = (current.mu + n.mu) / 2.0
            p = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=f"play_{i}_step_{idx}", use_diag=True)
            p.add_child(current); p.add_child(n); current = p
        return current

    def explore(self, n_cycles: int = 100):
        print(f"  [CURIOSITY] Starting {n_cycles} cycles of unsupervised play...")
        discovered_tasks = []
        known_deltas = set()
        
        priors = {c: self.forest.get(f"prior_rule_{c}") for c in CONCEPTS}
        
        for i in range(n_cycles):
            sequence = []
            
            # Start with random header
            header = np.random.choice(["LIST_INIT", "VAR_INP", "CONST_1"])
            sequence.append(priors[header])
            
            # Add loop/condition with higher probability
            if np.random.random() > 0.3:
                sequence.append(priors["FOR_LOOP"])
                sequence.append(priors["ITEM_ACCESS"])
                # Inner logic - longer
                for _ in range(np.random.randint(2, 5)):
                    sequence.append(priors[np.random.choice(["OP_ADD", "OP_MUL2", "OP_SUB", "COND_IS_EVEN", "BLOCK_ELSE"])])
                sequence.append(priors["LIST_APPEND"])
                sequence.append(priors["BLOCK_END"])
            
            # Additional logic after loop
            for _ in range(np.random.randint(1, 4)):
                sequence.append(priors[np.random.choice(["OP_ADD", "OP_MUL2", "OP_SUB", "VAR_INP", "CONST_1"])])
            
            sequence.append(priors["RETURN"])
            
            tree = self._fold_nodes(sequence, i)
            code = self.renderer.render(tree)
            test_inputs = [[1, 2, 3, 4], 5, [10, 20]]
            
            stable = True
            outputs = []
            for inp in test_inputs:
                out, error = self.executor.run(code, inp)
                if error: 
                    stable = False; break
                outputs.append(out)
            
            if stable:
                final_state = self.oracle.compute_state(test_inputs[0], outputs[0])
                delta_key = tuple(np.round(final_state, 2))
                
                is_novel = True
                if outputs[0] == test_inputs[0] or outputs[0] == 0 or outputs[0] == [] or outputs[0] == None: 
                    is_novel = False
                elif delta_key in known_deltas: 
                    is_novel = False
                
                if is_novel:
                    known_deltas.add(delta_key)
                    task_id = f"auto_task_{len(discovered_tasks)}"
                    discovered_tasks.append({
                        "id": task_id,
                        "code": code,
                        "example_input": test_inputs[0],
                        "example_output": outputs[0],
                        "state_delta": final_state.tolist()
                    })
                    tree.id = f"discovery_{task_id}"
                    self.forest.register(tree)
                    self.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{tree.id}", use_diag=True))
                    print(f"    [DISCOVERY] Cycle {i}: Found unique behavior! Output: {outputs[0]}")

        print(f"  [CURIOSITY] Discovery phase complete. Found {len(discovered_tasks)} novel programs.")
        return discovered_tasks

    def save(self):
        self.persistence.save(self.forest, self.observer, self.experiment_id)

# --- 5. EXPERIMENT RUNNER ---

def run_experiment():
    print("--- SP51: Experiment 27 — Autonomous Curriculum Generation ---\n")
    agent = SelfPlayAgent()
    discoveries = agent.explore(200) 
    output_path = Path("hpm_fractal_node/experiments/tasks/auto_curriculum.json")
    with open(output_path, "w") as f:
        json.dump(discoveries, f, indent=2)
    print(f"\n  [CURRICULUM] Auto-generated curriculum saved to {output_path}")
    agent.save()

if __name__ == "__main__":
    run_experiment()
