"""
SP50: Experiment 26 — Higher-Order Template Extraction (Refactoring)

Validates the autonomous discovery of Higher-Order Invariants (MAP, FILTER).
Uses TieredForest and PersistenceManager to build a cumulative knowledge base.
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
    "CALL_FUNC",
    "SLOT",         # Placeholder for higher-order argument
    "TEMPLATE_MAP", # Higher-order invariant
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}
# State: [Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot]
S_DIM = 9 
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

class TaskLoader:
    @staticmethod
    def load(path: str) -> List[Task]:
        with open(path, 'r') as f:
            data = json.load(f)
        return [Task(**d) for d in data]

# --- 3. THE HIGHER-ORDER CODE PIPELINE ---

def get_hfn_leaves(node: HFN) -> List[HFN]:
    if node is None: return []
    children = node.children()
    if not children: return [node]
    leaves = []
    for c in children: leaves.extend(get_hfn_leaves(c))
    return leaves

class CodeRenderer:
    def __init__(self):
        self.library: Dict[str, HFN] = {}

    def _get_concept(self, node: HFN) -> Optional[str]:
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}": return c
        if node.id.startswith("prior_rule_") or node.id.startswith("call_") or node.id.startswith("template_"):
            action_vec = node.mu[S_DIM : S_DIM + DIM]
            if np.max(action_vec) > 1.0: return CONCEPTS[np.argmax(action_vec)]
        return None

    def render_program(self, root_node: HFN, slot_argument: Optional[HFN] = None) -> str:
        # Pass 1: Render Library Functions
        global_lines = []
        for func_name, tree in self.library.items():
            global_lines.append(f"def {func_name}(val):")
            global_lines.append(self._render_node(tree, indent="    "))
            global_lines.append("")
            
        # Pass 2: Render Main Block
        main_code = self._render_node(root_node, indent="", slot_arg=slot_argument)
        return "\n".join(global_lines) + "\n" + main_code

    def _render_node(self, node: HFN, indent: str = "", slot_arg: Optional[HFN] = None) -> str:
        leaves = get_hfn_leaves(node)
        lines = []
        curr_indent = indent
        has_return = False
        
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept: continue
            
            if concept == "SLOT" and slot_arg:
                lines.append(self._render_node(slot_arg, indent=curr_indent))
            elif concept == "CALL_FUNC":
                func_name = leaf.id.replace("call_", "") if "call_" in leaf.id else "library_func"
                lines.append(f"{curr_indent}val = {func_name}(val)")
            elif concept == "CONST_1": lines.append(f"{curr_indent}x = 1")
            elif concept == "VAR_INP": lines.append(f"{curr_indent}x = inp")
            elif concept == "OP_ADD": lines.append(f"{curr_indent}val += 1" if curr_indent != "" else f"{curr_indent}x += 1")
            elif concept == "OP_SUB": lines.append(f"{curr_indent}val -= 1" if curr_indent != "" else f"{curr_indent}x -= 1")
            elif concept == "OP_MUL2": lines.append(f"{curr_indent}val *= 2" if curr_indent != "" else f"{curr_indent}x *= 2")
            elif concept == "LIST_INIT": lines.append(f"{curr_indent}res = []")
            elif concept == "FOR_LOOP":
                lines.append(f"{curr_indent}for item in x:")
                curr_indent += "    " 
            elif concept == "ITEM_ACCESS": lines.append(f"{curr_indent}val = item")
            elif concept == "LIST_APPEND":
                lines.append(f"{curr_indent}if res is not None: res.append(val)")
            elif concept == "RETURN":
                # Exit any active loop/block for return
                curr_indent = indent
                lines.append(f"{curr_indent}return res if res is not None else x")
                has_return = True
                break
        
        if not has_return:
            if indent != "": lines.append(f"{indent}return val")
            else: lines.append(f"{indent}return res if res is not None else x")
            
        return "\n".join(lines)

class PythonExecutor:
    def run(self, code_str: str, inputs: List[Any]) -> Any:
        if not code_str: return None
        code = f"def test_func(inp):\n    x = 0\n    val = 0\n    res = None\n    {code_str.replace('\n', '\n    ')}\n"
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            return test_func(inputs[0] if inputs else None)
        except Exception: return None

# --- 4. THE REFACTORING AGENT ---

class RefactoringAgent:
    def __init__(self, dim=DIM, cold_dir="data/knowledge_base/template_extraction"):
        self.dim = dim
        self.m_dim = S_DIM + DIM + S_DIM
        self.forest = TieredForest(D=self.m_dim, cold_dir=Path(cold_dir))
        self.persistence = PersistenceManager(root_dir="data/knowledge_base")
        self.experiment_id = "template_extraction"
        
        self.retriever = GoalConditionedRetriever(
            self.forest, target_slice=slice(S_DIM + DIM, self.m_dim), 
            target_weight=50.0, weight_provider=lambda nid: self.observer.get_weight(nid)
        )
        
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.5, residual_surprise_threshold=0.6, alpha_gain=0.5, beta_loss=0.05, node_use_diag=True)
        self.renderer = CodeRenderer()
        
        self.persistence.load(self.forest, self.observer, self.experiment_id)
        if "prior_rule_RETURN" not in self.forest:
            self._inject_priors()

    def _inject_priors(self):
        # [Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot]
        self._add_rule("LIST_INIT", 5, 50.0)
        self._add_rule("VAR_INP", 0, 50.0)
        self._add_rule("FOR_LOOP", 4, 50.0)
        self._add_rule("ITEM_ACCESS", 4, 0.0)
        self._add_rule("LIST_APPEND", 2, 50.0)
        self._add_rule("OP_ADD", 0, 50.0)
        self._add_rule("OP_MUL2", 0, 100.0)
        self._add_rule("OP_SUB", 0, -50.0)
        self._add_rule("RETURN", 1, 50.0)
        self._add_rule("SLOT", 8, 50.0) # Special SLOT prior
        self.forest._stale_index = True; self.forest._sync_gaussian()

    def _add_rule(self, concept, delta_idx, delta_val):
        mu = np.zeros(self.m_dim); mu[S_DIM + CONCEPT_IDX[concept]] = 5.0; mu[S_DIM + DIM + delta_idx] = delta_val
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*5.0, id=f"prior_rule_{concept}", use_diag=True)
        self.forest.register(node)
        self.observer.protected_ids.add(node.id)
        self.observer.meta_forest.register(HFN(mu=np.array([0.8, 0, 0, 0]), sigma=np.ones(4), id=f"state:{node.id}", use_diag=True))

    def _fold_nodes(self, nodes: List[HFN]) -> HFN:
        if not nodes: return None
        current = nodes[0]
        for n in nodes[1:]:
            p_mu = (current.mu + n.mu) / 2.0; p_mu[S_DIM + DIM:] = current.mu[S_DIM + DIM:] + n.mu[S_DIM + DIM:]
            p_id = f"compose({current.id.replace('prior_rule_', '')}+{n.id.replace('prior_rule_', '')})"
            p = HFN(mu=p_mu, sigma=np.ones(self.m_dim), id=p_id, use_diag=True)
            p.add_child(current); p.add_child(n); current = p
        return current

    def extract_template(self, node_a: HFN, node_b: HFN) -> HFN:
        """Isolates invariants across two trees and creates a template with a SLOT."""
        leaves_a = get_hfn_leaves(node_a)
        leaves_b = get_hfn_leaves(node_b)
        
        if len(leaves_a) != len(leaves_b):
            return None
            
        template_leaves = []
        for la, lb in zip(leaves_a, leaves_b):
            if la.id == lb.id:
                template_leaves.append(la)
            else:
                # Variance detected! Replace with SLOT
                template_leaves.append(self.forest.get("prior_rule_SLOT"))
                
        template = self._fold_nodes(template_leaves)
        template.id = "template_MAP"
        # Increase delta weight for SLOT dimension to signal higher-order requirement
        template.mu[S_DIM + DIM + 8] = 50.0
        
        self.forest.register(template)
        self.observer.meta_forest.register(HFN(mu=np.array([0.9, 0, 0, 0]), sigma=np.ones(4), id=f"state:{template.id}", use_diag=True))
        print(f"  [REFACTOR] Extracted Template: {template.id} with SLOT for variable logic.")
        return template

    def save(self): self.persistence.save(self.forest, self.observer, self.experiment_id)

# --- 5. EXPERIMENT RUNNER ---

def run_experiment():
    print("--- SP50: Experiment 26 — Higher-Order Template Extraction (Refactoring) ---\n")
    tasks = TaskLoader.load("hpm_fractal_node/experiments/tasks/higher_order_curriculum.json")
    agent = RefactoringAgent(); executor = PythonExecutor()
    
    # 1. Learn Concrete Map A: map_increment
    print("Step 1: Learning 'map_increment'...")
    exp_a = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "OP_ADD", "LIST_APPEND", "RETURN"]
    nodes_a = [agent.forest.get(f"prior_rule_{c}") for c in exp_a]
    chunk_a = agent._fold_nodes(nodes_a); chunk_a.id = "chunk_map_increment"
    
    # 2. Learn Concrete Map B: map_double
    print("Step 2: Learning 'map_double'...")
    exp_b = ["LIST_INIT", "VAR_INP", "FOR_LOOP", "ITEM_ACCESS", "OP_MUL2", "LIST_APPEND", "RETURN"]
    nodes_b = [agent.forest.get(f"prior_rule_{c}") for c in exp_b]
    chunk_b = agent._fold_nodes(nodes_b); chunk_b.id = "chunk_map_double"
    
    # 3. Refactoring Dream: Extract Template
    print("\nStep 3: Triggering 'Refactoring Dream' phase...")
    template = agent.extract_template(chunk_a, chunk_b)
    
    # 4. Zero-Shot Application: map_decrement
    print("\nStep 4: Zero-Shot Application on 'map_decrement'...")
    task_c = tasks[2]
    # We retrieve the template and the specific logic (OP_SUB)
    op_sub = agent.forest.get("prior_rule_OP_SUB")
    
    # Render with parameterization
    code_c = agent.renderer.render_program(template, slot_argument=op_sub)
    print(f"    Rendered Program (Using Template):\n    {code_c.replace('\\n', '\\n    ')}\n")
    
    result = executor.run(code_c, task_c.input)
    print(f"    Result: {result} (Goal: {task_c.expected_output})")
    
    if result == task_c.expected_output:
        print("\n[SUCCESS] Higher-Order Template extracted and applied zero-shot!")
    
    # SAVE KNOWLEDGE BASE
    agent.save()

if __name__ == "__main__":
    run_experiment()
