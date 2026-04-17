"""
SP-Reader 5: Inter-Conceptual Analogy & Zero-Shot Transfer.
Verifies structural isomorphism detection between Biology and Software domains.
"""
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

# --- Source Domain: Biology (Circular Evolution) ---
BIO_TEXT = """
Mutation introduces genetic variation.

Selection filters for survival.

Inheritance passes on successful traits.

Evolution repeats this cyclical life cycle.
"""

# --- Target Domain: Software (Circular Development) ---
SOFT_TEXT = """
Coding creates new software features.

Testing filters for high quality.

Deployment ships the working software.

Iterative development repeats this cyclical release cycle.
"""

def run_experiment():
    print("================================================================================")
    print("SP-Reader 5: Structural Isomorphism & Zero-Shot Transfer")
    print("================================================================================")

    # 1. Setup Source Domain (Biology)
    print("Step 1: Training Source Domain (Biology)...")
    config_bio = TextDomainConfig.from_passages([BIO_TEXT], max_vocab=100)
    agent = ReaderAgent(config_bio, cold_dir="data/knowledge_base/sp_reader5_analogy")
    
    # Observe Biology document multiple times to stabilize transitions
    for _ in range(5):
        agent.observe_document(BIO_TEXT, min_length=1)
    
    agent.build_topic_clusters(n_clusters=4)
    agent.stabilize_universal_concepts(n_concepts=1)
    agent.learn_thematic_transitions(0)
    
    # 2. Observe Target Domain (Software)
    print("\nStep 2: Observing Target Domain (Software)...")
    agent.expand_vocabulary([SOFT_TEXT], max_new=100)
    for _ in range(5):
        agent.observe_document(SOFT_TEXT, min_length=1)
    
    # Rebuild topics for BOTH domains
    agent.build_topic_clusters(n_clusters=8) 
    agent.stabilize_universal_concepts(n_concepts=2) 
    
    # Re-learn transitions for BOTH domains
    print("  Re-learning thematic transitions for both domains...")
    agent.learn_thematic_transitions(0) # Bio
    agent.learn_thematic_transitions(5) # Soft (Doc 0-4 are Bio, Doc 5-9 are Soft)
    
    # Find which concept is which by checking summaries
    concepts = [k for k in agent.patterns if k.startswith("concept_")]
    source_concept_id = None
    target_concept_id = None
    
    for cid in concepts:
        summary = agent.summarize_node(cid).lower()
        if any(w in summary for w in ["mutation", "selection", "inheritance", "evolution", "genetic"]):
            source_concept_id = cid
        if any(w in summary for w in ["coding", "testing", "deployment", "software", "development"]):
            target_concept_id = cid
            
    print(f"  Source (Bio) Concept: {source_concept_id} - {agent.summarize_node(source_concept_id)}")
    print(f"  Target (Soft) Concept: {target_concept_id} - {agent.summarize_node(target_concept_id)}")

    if not source_concept_id or not target_concept_id or source_concept_id == target_concept_id:
        print("  FAILURE: Could not separate concepts clearly.")
        return

    # 3. Detect Structural Analogy
    print("\nStep 3: Finding Structural Analogy (Isomorphism)...")
    analogy_source_id = agent.find_structural_analogy(target_concept_id)
    print(f"  Analogy Found: {target_concept_id} is isomorphic to {analogy_source_id}")
    
    if analogy_source_id == source_concept_id:
        print(f"  SUCCESS: Isomorphism detected!")
    else:
        print("  FAILURE: Analogy detection failed.")

    # 4. Zero-Shot Strategic Transfer
    print("\nStep 4: Zero-Shot Strategic Transfer...")
    if analogy_source_id:
        mapping = agent.transfer_strategy(analogy_source_id, target_concept_id)
        print("  Functional Role Mapping:")
        for target_topic_id, source_topic_id in list(mapping.items())[:4]:
            t_sum = agent.summarize_node(target_topic_id, top_n=2)
            s_sum = agent.summarize_node(source_topic_id, top_n=2)
            print(f"    {t_sum} <---> {s_sum}")

    # 5. Cross-Domain Analogical Retrieval
    print("\nStep 5: Cross-Domain Analogical Retrieval:")
    query = "Testing filters for high quality and ensures working software."
    print(f"  Query: '{query}'")

    res = agent.query_hierarchical(query)
    print(f"  Combined Result:\n{res}")

    
    if "[Target]" in res and "[Analogy]" in res:
        print("\n  SUCCESS: Agent performed cross-domain retrieval via structural isomorphism!")
    else:
        print("\n  PARTIAL: Hierarchical retrieval worked.")

if __name__ == "__main__":
    run_experiment()
