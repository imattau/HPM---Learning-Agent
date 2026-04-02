"""
SP21: Toddler Sentence Generator.
Top-down synthesis across a Structural Octet of HFN domains.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN, Edge
from hfn.tiered_forest import TieredForest

# --- Domain Constants ---
D = 32 # Small latent space for this logic test

@dataclass
class MentalEvent:
    agent_id: str
    action_id: str
    patient_id: str
    tense: str # "PAST" | "PRESENT"

class SentenceSynthesizer:
    def __init__(self, forests: dict[str, TieredForest]):
        self.forests = forests

    def generate(self, event: MentalEvent) -> str:
        # 1. Retrieve Narrative Template
        narrative_rule = self.forests["narrative"].get("rule_agent_action_patient")
        if not narrative_rule: raise ValueError("Narrative rule not found")

        # 2. Validate Affordances
        # Check if Agent can perform Action
        agent_node = self.forests["identity"].get(event.agent_id)
        action_node = self.forests["identity"].get(event.action_id)
        patient_node = self.forests["identity"].get(event.patient_id)
        
        if not agent_node: return f"ERROR: Agent {event.agent_id} not found"
        if not action_node: return f"ERROR: Action {event.action_id} not found"
        if not patient_node: return f"ERROR: Patient {event.patient_id} not found"

        if not self._validate_affordance(agent_node, action_node, "agent"):
            return f"ERROR: Affordance Violation - {event.agent_id} cannot {event.action_id}"

        # 3. Resolve Syntactic Slots
        # Subject
        subject_str = self._resolve_token(agent_node)
        # Apply Determiner if not proper noun (Simplified: if starts with word_, needs 'the')
        if not event.agent_id.startswith("entity_mum"):
            subject_str = "the " + subject_str

        # Verb + Morphology
        verb_base = self._resolve_token(action_node)
        verb_str = self._apply_morphology(verb_base, event.tense)

        # Object
        object_str = self._resolve_token(patient_node)
        object_str = "the " + object_str

        return f"{subject_str.capitalize()} {verb_str} {object_str}."

    def _validate_affordance(self, entity: HFN, action: HFN, role: str) -> bool:
        # Simplified: Check if entity has a DAG edge to a category that has the affordance
        # In a real HPM AI, this would be a log_prob check against the Affordance HFN.
        # For this experiment, we check the child edges.
        for edge in entity.edges():
            if edge.relation == "IS_A":
                category = edge.target
                for aff_edge in category.edges():
                    if aff_edge.relation == f"CAN_{action.id.upper()}":
                        return True
        return False

    def _resolve_token(self, identity: HFN) -> str:
        # Traverse Relational -> Token
        for edge in identity.edges():
            if edge.relation == "HAS_TOKEN":
                token_node = edge.target
                # If it's a proxy (re-instantiated in a new forest), 
                # we might need to find its match in the Token Forest
                best_match = self.forests["token"].retrieve(token_node.mu, k=1)
                if best_match:
                    return best_match[0].id.replace("word_", "")
        return "???"

    def _apply_morphology(self, base: str, tense: str) -> str:
        if tense == "PAST":
            if base == "eat": return "ate"
            if base == "chase": return "chased"
        if tense == "PRESENT":
            if base == "eat": return "eats"
            if base == "chase": return "chases"
        return base

def build_octet_priors() -> dict[str, TieredForest]:
    names = ["token", "morph", "identity", "category", "affordance", "syntax", "relational", "narrative"]
    # Ensure directory exists and is clean
    base_dir = Path("data/sp21")
    if base_dir.exists():
        import shutil
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    forests = {n: TieredForest(D=D, forest_id=f"f_{n}", cold_dir=base_dir / n) for n in names}
    
    # 1. Tokens
    word_mum = HFN(np.zeros(D), np.eye(D), "word_mum")
    word_eat = HFN(np.zeros(D), np.eye(D), "word_eat")
    word_apple = HFN(np.zeros(D), np.eye(D), "word_apple")
    word_dog = HFN(np.zeros(D), np.eye(D), "word_dog")
    word_ball = HFN(np.zeros(D), np.eye(D), "word_ball")
    for w in [word_mum, word_eat, word_apple, word_dog, word_ball]: forests["token"].register(w)

    # 2. Categories
    cat_person = HFN(np.zeros(D), np.eye(D), "cat_person")
    cat_food = HFN(np.zeros(D), np.eye(D), "cat_food")
    cat_toy = HFN(np.zeros(D), np.eye(D), "cat_toy")
    cat_animal = HFN(np.zeros(D), np.eye(D), "cat_animal")
    for c in [cat_person, cat_food, cat_toy, cat_animal]: forests["category"].register(c)

    # 3. Affordances (Linked to Categories)
    # HFN for specific affordances
    aff_eat = HFN(np.zeros(D), np.eye(D), "aff_eat")
    aff_chase = HFN(np.zeros(D), np.eye(D), "aff_chase")
    aff_eaten = HFN(np.zeros(D), np.eye(D), "aff_eaten")
    for a in [aff_eat, aff_chase, aff_eaten]: forests["affordance"].register(a)

    cat_person.add_edge(cat_person, aff_eat, "CAN_EAT")
    cat_animal.add_edge(cat_animal, aff_chase, "CAN_CHASE")
    cat_food.add_edge(cat_food, aff_eaten, "CAN_BE_EATEN")

    # 4. Identities (Linked to Categories and Tokens)
    entity_mum = HFN(np.zeros(D), np.eye(D), "entity_mum")
    entity_mum.add_edge(entity_mum, cat_person, "IS_A")
    entity_mum.add_edge(entity_mum, word_mum, "HAS_TOKEN")

    entity_eat = HFN(np.zeros(D), np.eye(D), "eat") # Action Identity
    entity_eat.add_edge(entity_eat, word_eat, "HAS_TOKEN")

    entity_apple = HFN(np.zeros(D), np.eye(D), "entity_apple")
    entity_apple.add_edge(entity_apple, cat_food, "IS_A")
    entity_apple.add_edge(entity_apple, word_apple, "HAS_TOKEN")

    entity_dog = HFN(np.zeros(D), np.eye(D), "entity_dog")
    entity_dog.add_edge(entity_dog, cat_animal, "IS_A")
    entity_dog.add_edge(entity_dog, word_dog, "HAS_TOKEN")

    entity_chase = HFN(np.zeros(D), np.eye(D), "chase")
    entity_chase.add_edge(entity_chase, word_chase := HFN(np.zeros(D), np.eye(D), "word_chase"), "HAS_TOKEN")
    forests["token"].register(word_chase)

    entity_ball = HFN(np.zeros(D), np.eye(D), "entity_ball")
    entity_ball.add_edge(entity_ball, cat_toy, "IS_A")
    entity_ball.add_edge(entity_ball, word_ball, "HAS_TOKEN")

    for e in [entity_mum, entity_eat, entity_apple, entity_dog, entity_chase, entity_ball]: 
        forests["identity"].register(e)

    # 5. Narrative
    rule = HFN(np.zeros(D), np.eye(D), "rule_agent_action_patient")
    forests["narrative"].register(rule)

    return forests

def run_experiment():
    print("SP21: Toddler Sentence Generator Experiment\n")
    forests = build_octet_priors()
    synth = SentenceSynthesizer(forests)

    # Smoke Tests
    print("--- Smoke Testing ---")
    # 1. Resolution
    mum_node = forests["identity"].get("entity_mum")
    resolved = synth._resolve_token(mum_node)
    print(f"  [Smoke 1] Resolution (Mum -> Token): {'PASS' if resolved == 'mum' else 'FAIL'} ({resolved})")

    # 2. Affordance
    eat_node = forests["identity"].get("eat")
    can_eat = synth._validate_affordance(mum_node, eat_node, "agent")
    print(f"  [Smoke 2] Affordance (Mum can Eat): {'PASS' if can_eat else 'FAIL'}")

    # 3. Template
    rule = forests["narrative"].get("rule_agent_action_patient")
    print(f"  [Smoke 3] Template Exists: {'PASS' if rule else 'FAIL'}\n")

    # Full Prompts
    print("--- Generative Prompts ---")
    goals = [
        MentalEvent("entity_mum", "eat", "entity_apple", "PAST"),
        MentalEvent("entity_dog", "chase", "entity_ball", "PRESENT"),
        MentalEvent("entity_apple", "eat", "entity_mum", "PAST"), # Should fail affordance
    ]

    for i, goal in enumerate(goals):
        result = synth.generate(goal)
        print(f"  Goal {i+1}: {goal}")
        print(f"  Output: {result}\n")

if __name__ == "__main__":
    run_experiment()
