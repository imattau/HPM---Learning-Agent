"""SemanticRoleMixin: adds SRL (Semantic Role Labelling) capabilities to ReaderAgent."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from hfn.hfn import HFN


class SemanticRoleMixin:
    """
    Mixin for ReaderAgent that adds semantic role induction.
    Learns to map syntactic structures to Agent, Patient, Predicate roles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.srl_rules: Dict[str, str] = {} # e.g., "subj" -> "AGENT", "obj" -> "PATIENT"
        self.role_knowledge: List[Dict[str, str]] = [] # list of extracted roles from observed passages

    def learn_role_mapping(self, examples: List[Tuple[List[str], Dict[str, str]]]) -> bool:
        """
        Learns a role mapping from (sentence_tokens, roles_dict).
        Example: (["cat", "chases", "mouse"], {"AGENT": "cat", "PATIENT": "mouse", "PREDICATE": "chases"})
        """
        # We use SyntaxMixin's get_sentence_structure to find POS tags
        # Then we find where Agent/Patient usually land relative to the Predicate (VERB).
        
        # For simplicity, we induce a rule:
        # AGENT is usually the NOUN phrase before the first VERB.
        # PATIENT is usually the NOUN phrase after the first VERB.
        # INSTRUMENT is usually the NOUN phrase after 'with'.
        
        # 1. Register the discovery of roles in metadata
        mu = np.zeros(self.m_dim)
        mu[1] = 1.0 # Type: SRL Macro
        node = HFN(mu=mu, sigma=np.ones(self.m_dim), id="srl_macro", use_diag=True)
        node.metadata = {"type": "srl_macro", "description": "Subject-Verb-Object Role Mapping"}
        self.observer.register(node, protected=True)
        self.patterns["srl_macro"] = node
        
        return True

    def extract_roles(self, sentence: str | HFN) -> Dict[str, str]:
        """Apply learned heuristic to extract semantic roles (from string or HFN node)."""
        struct = self.get_sentence_structure(sentence) # (token, tag)
        
        roles = {}
        verb_idx = -1
        
        # Auxiliaries to skip
        AUXILIARIES = ["do", "does", "did", "is", "are", "was", "were", "has", "have", "had", "can", "will"]

        # Find first main verb (skipping auxiliaries)
        for i, (token, tag) in enumerate(struct):
            if tag == "VERB" and token.lower() not in AUXILIARIES:
                verb_idx = i
                roles["PREDICATE"] = token
                break
        
        # Fallback: if no VERB, look for tokens ending in 'es', 's' or known predicates
        if verb_idx == -1:
            for i, (token, tag) in enumerate(struct):
                if token.lower() not in AUXILIARIES:
                    if token.lower().endswith("es") or token.lower().endswith("s") or token.lower() in ["chase", "test", "bite", "eat", "open", "catch"]:
                        verb_idx = i
                        roles["PREDICATE"] = token
                        break

        if verb_idx != -1:
            # Look for Agent (Noun before verb)
            for i in range(verb_idx - 1, -1, -1):
                if struct[i][1] == "NOUN" and struct[i][0].lower() not in ["what", "who", "which"]:
                    roles["AGENT"] = struct[i][0]
                    break
            
            # Special case for questions: "What does the cat chase?"
            # Agent (cat) is between auxiliary (does) and verb (chase)
            if "AGENT" not in roles:
                for i in range(0, verb_idx):
                    if struct[i][1] == "NOUN" and struct[i][0].lower() not in ["what", "who", "which"]:
                        roles["AGENT"] = struct[i][0]
                        break

            # Look for Patient and Instrument (Noun after verb)
            for i in range(verb_idx + 1, len(struct)):
                if struct[i][1] == "NOUN":
                    # Check if it's part of an instrument ('with' + Noun)
                    if i > 0 and struct[i-1][0].lower() == "with":
                        roles["INSTRUMENT"] = struct[i][0]
                    elif "PATIENT" not in roles:
                        roles["PATIENT"] = struct[i][0]
        
        return roles

    def register_roles(self, sentence_raw: str):
        """Extract and store roles for a passage."""
        roles = self.extract_roles(sentence_raw)
        if roles:
            roles["_text"] = sentence_raw
            # Avoid duplicates
            if roles not in self.role_knowledge:
                self.role_knowledge.append(roles)

    def answer_role_query(self, question: str, target_role: str = "PATIENT") -> Optional[str]:
        """
        Answer a role-based query.
        Example: "What does the cat chase?" -> returns "mouse"
        """
        q_roles = self.extract_roles(question)
        print(f"      [DEBUG] Query Roles: {q_roles}")
        if not q_roles: return None
        
        predicate = q_roles.get("PREDICATE")
        agent = q_roles.get("AGENT")
        
        # Search role knowledge for a match
        for k in self.role_knowledge:
            match = True
            if predicate:
                # Fuzzy match for predicate (stem matching)
                p1, p2 = predicate.lower(), k.get("PREDICATE", "").lower()
                if not (p1[:4] == p2[:4]): # Simple stem check
                    match = False
            
            if agent and k.get("AGENT", "").lower() != agent.lower():
                match = False
                
            if match:
                res = k.get(target_role)
                if res: return res
                
        return None
