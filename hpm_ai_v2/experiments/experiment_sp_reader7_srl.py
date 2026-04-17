"""
SP-Reader 7: Semantic Role Induction (Predicate-Argument Structure).
Verifies learning of semantic roles and role-based query answering.
"""
import os
from pathlib import Path
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

# --- Training Data for POS (from SP-Reader 6) ---
TRAIN_POS_SENTENCES = [
    ["the", "cat", "chases", "the", "mouse"],
    ["a", "big", "dog", "runs", "fast"],
    ["she", "quickly", "eats", "an", "apple"],
    ["the", "fox", "jumps", "over", "the", "lazy", "dog"]
]
TRAIN_POS_TAGS = [
    ["DET", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "ADV"],
    ["NOUN", "ADV", "VERB", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]
]

# --- Phase 1 Training Data (Agent & Patient) ---
TRAIN_SRL = [
    (["cat", "chases", "mouse"], {"AGENT": "cat", "PATIENT": "mouse", "PREDICATE": "chases"}),
    (["dog", "bites", "bone"], {"AGENT": "dog", "PATIENT": "bone", "PREDICATE": "bites"}),
    (["girl", "eats", "apple"], {"AGENT": "girl", "PATIENT": "apple", "PREDICATE": "eats"})
]

KNOWLEDGE_BASE_DIR = "data/knowledge_base/reader_lifelong"

def run_experiment():
    print("================================================================================")
    print("SP-Reader 7: Semantic Role Induction & Shallow Semantics")
    print("================================================================================")

    # 1. Setup Persistent Reader Agent
    print("Phase 1: Initialising Persistent Reader Agent...")
    if Path(KNOWLEDGE_BASE_DIR).exists():
        print(f"  Loading existing knowledge from {KNOWLEDGE_BASE_DIR}...")
        agent = ReaderAgent.load_agent(KNOWLEDGE_BASE_DIR)
    else:
        print("  Creating new lifelong knowledge base...")
        config = TextDomainConfig.from_passages(["The initial knowledge base."], max_vocab=100)
        agent = ReaderAgent(config, cold_dir=KNOWLEDGE_BASE_DIR)

    # 2. Ensure POS knowledge is stable
    print("\nPhase 2: Stabilising Syntactic Knowledge...")
    agent.learn_pos_tagger(TRAIN_POS_SENTENCES, TRAIN_POS_TAGS)
    
    # 3. Phase 1: Semantic Role Induction
    print("\nPhase 3: Semantic Role Induction from minimal examples...")
    agent.learn_role_mapping(TRAIN_SRL)
    print("  SRL Macro induced.")

    # 4. Phase 2: Generalisation to New Verbs/Nouns
    print("\nPhase 4: Semantic Generalisation to new predicates...")
    test_sent = "The bird catches the worm."
    roles = agent.extract_roles(test_sent)
    print(f"  Sentence: '{test_sent}'")
    print(f"  Extracted Roles: {roles}")
    
    if roles.get("AGENT") == "bird" and roles.get("PATIENT") == "worm":
        print("  SUCCESS: Roles correctly generalized!")
    else:
        print("  PARTIAL: Roles mismatched.")

    # 5. Phase 3: Instrument Role Induction
    print("\nPhase 5: Instrument Role Induction...")
    test_instr = "The key opens the door with force."
    roles_instr = agent.extract_roles(test_instr)
    print(f"  Sentence: '{test_instr}'")
    print(f"  Extracted Roles: {roles_instr}")
    
    if roles_instr.get("INSTRUMENT") == "force":
        print("  SUCCESS: Instrument role correctly identified!")
    else:
        print("  PARTIAL: Instrument missing.")

    # 6. Phase 4: Role-Based Query Answering
    print("\nPhase 6: Role-Based Query Answering...")
    agent.register_roles("The cat chases the mouse.")
    agent.register_roles("The dog bites the bone.")
    
    query = "What does the cat chase?"
    print(f"  Query: '{query}'")
    ans = agent.answer_role_query(query, target_role="PATIENT")
    print(f"  Answer (Patient): {ans}")
    
    if ans == "mouse":
        print("  SUCCESS: Correct semantic answer retrieved!")
    else:
        print("  FAILURE: Answer incorrect.")

    # 7. Phase 5: Cross-Domain Role Transfer (Structural Analogy)
    print("\nPhase 7: Cross-Domain Role Transfer (via SP-Reader 5 Analogy)...")
    # Soft Domain: Code tests bugs
    agent.register_roles("The code tests the bugs.")
    
    # Analogy query: What does the code test?
    query_soft = "What does the code test?"
    print(f"  Query: '{query_soft}'")
    ans_soft = agent.answer_role_query(query_soft, target_role="PATIENT")
    print(f"  Answer (Patient): {ans_soft}")
    
    if ans_soft == "bugs":
        print("  SUCCESS: Role schema transferred to Software domain!")
    else:
        print("  PARTIAL: Soft domain answer incorrect.")

    # 8. Phase 6: Thematic Integration & Surprise
    print("\nPhase 8: Thematic Integration & Role-Reversal Surprise...")
    # Expected: Cat chases mouse
    # Surprising: Mouse chases cat
    
    # Simple predictive curiosity proxy: role violation
    def role_surprise(sentence: str) -> float:
        extracted = agent.extract_roles(sentence)
        pred = extracted.get("PREDICATE", "").lower()
        subj = extracted.get("AGENT", "").lower()
        obj = extracted.get("PATIENT", "").lower()
        
        # Check against knowledge
        surprise = 1.0
        for k in agent.role_knowledge:
            if k.get("PREDICATE", "").lower().startswith(pred[:4]):
                if k.get("AGENT", "").lower() == subj:
                    surprise = 0.1 # Consistent
                    break
        return surprise

    s1 = "The cat chases the mouse."
    s2 = "The mouse chases the cat."
    print(f"  Normal Event: '{s1}' -> Surprise: {role_surprise(s1)}")
    print(f"  Role-Reversed Event: '{s2}' -> Surprise: {role_surprise(s2)}")
    
    if role_surprise(s2) > role_surprise(s1):
        print("  SUCCESS: Semantic role violation detected as surprising!")
    else:
        print("  FAILURE: No surprise difference.")

    # 9. Persistence
    print("\nPhase 9: Saving expanded knowledge base...")
    agent.save_agent(KNOWLEDGE_BASE_DIR)
    print(f"  Knowledge base saved to {KNOWLEDGE_BASE_DIR}")

if __name__ == "__main__":
    run_experiment()
