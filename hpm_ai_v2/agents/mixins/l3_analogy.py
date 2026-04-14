"""
L3AnalogyMixin — HPM Level 3: cross-domain structural analogy.

Provides:
- _try_l3_analogy(inputs, outputs): synthesize a macro by mapping an L3 schema from another domain.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Dict

from hfn.hfn import HFN

class L3AnalogyMixin:
    """
    Mixin adding cross-domain analogy via L3 meta-schemas.
    
    Relies on CONCEPT_MAPPINGS to translate between domains.
    """
    
    # Generic cross-domain concept correspondences
    # Key: Source domain concept name, Value: Target domain concept name
    ANALOGY_MAP: Dict[str, str] = {
        "FOR_LOOP": "FOR_EACH_NODE",
        "LIST_INIT": "COPY_GRAPH",
        "ITEM_ACCESS": None, # Skip ITEM_ACCESS, handle it by appending target op
        "VAR_INP": "VAR_INP",
        "LIST_APPEND": None,
        "BLOCK_END": "BLOCK_END",
        "RETURN": None
    }

    def _try_l3_analogy(
        self,
        inputs: List[Any],
        outputs: List[Any],
    ) -> Optional[List[HFN]]:
        """
        Strategy: Find an L3 meta-schema (e.g. from List domain) and map it 
        to the current domain (e.g. Graph) to synthesize a solution.
        """
        # 1. Find a meta-schema node in the forest (shared or local)
        schemas = [n for n in self.forest.active_nodes() if n.relation_type == "meta_schema"]
        if not schemas:
            return None
        
        # All primitives in CURRENT domain
        primitives = [n for n in self.forest.active_nodes() if n.relation_type != "macro" and "prior_rule_" in n.id]
        
        # Sort by support or length if available, otherwise just try them
        for schema in schemas:
            # 2. Extract constituent nodes from the schema
            if not schema.inputs:
                continue
                
            base_path: List[HFN] = []
            for src_node in schema.inputs:
                src_concept = self._infer_concept_from_node(src_node)
                if not src_concept:
                    continue
                
                target_concept = self.ANALOGY_MAP.get(src_concept)
                if target_concept:
                    target_node = self._get_primitive_node(target_concept)
                    if target_node:
                        base_path.append(target_node)
            
            if not base_path:
                continue
                
            # 3. For each primitive in target domain, try as the loop body
            for op in primitives:
                # Synthesize: [VAR_INP, COPY_GRAPH, FOR_EACH_NODE, OP, BLOCK_END]
                test_path = base_path + [op, self._get_primitive_node("BLOCK_END")]
                test_path = [n for n in test_path if n is not None]
                
                # 4. Verify the synthesized path
                code = self.renderer.render(self._compose_sequence(test_path))
                results, errors = self.executor.run_batch(code, inputs)
                
                if self._check_outputs(results, outputs):
                    # Success! Register the new macro
                    print(f"  [ANALOGY SUCCESS] Found mapping using schema {schema.id} + body {op.id}")
                    return test_path
                
        return None

    def _infer_concept_from_node(self, node: HFN) -> Optional[str]:
        """Attempt to guess the concept name of a node from its ID or mu vector."""
        # Check ID (e.g. prior_rule_FOR_LOOP)
        if "FOR_LOOP" in node.id: return "FOR_LOOP"
        if "LIST_INIT" in node.id: return "LIST_INIT"
        if "ITEM_ACCESS" in node.id: return "ITEM_ACCESS"
        if "VAR_INP" in node.id: return "VAR_INP"
        if "LIST_APPEND" in node.id: return "LIST_APPEND"
        if "RETURN" in node.id: return "RETURN"
        
        # Fallback to a brute-force check against a common set of concepts?
        # For SP77, ID-based inference is sufficient because we control the names.
        return None

    def _get_primitive_node(self, concept: str) -> Optional[HFN]:
        """Find the primitive node for a concept in the current domain's forest."""
        # Try finding by ID first
        for nid in [f"prior_rule_{concept}", f"graph_op_{concept}", f"list_op_{concept}"]:
            node = self.forest.get(nid)
            if node: return node
            
        # Fallback: search active nodes by mu one-hot
        for node in self.forest.active_nodes():
            if node.relation_type == "macro": continue
            # Check concept one-hot in middle slice
            start = self.config.S_DIM
            end = start + self.config.DIM
            action_vec = node.mu[start:end]
            if np.max(action_vec) > 0.5:
                idx = np.argmax(action_vec)
                if self.config.concepts[idx] == concept:
                    return node
        return None
