"""
SP-Reader3 Verification: Narrative Themes (L4) and Universal Concepts (L5).
Verifies sequential mapping, thematic induction, and concept stabilization.
"""
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

def run_experiment():
    print("================================================================================")
    print("SP-Reader3: Thematic and Conceptual Understanding (L4/L5)")
    print("================================================================================")

    # 1. Setup Cross-Domain Corpus
    # Domain A: Chess
    doc_chess = "Chess is a ancient strategic board game.\n\nKnights move in L-shapes while bishops move diagonally.\n\nThe objective is to achieve checkmate against the king."
    
    # Domain B: General Strategy
    doc_strategy = "Strategy requires careful planning and foresight.\n\nTactical maneuvers allow units to outflank opponents.\n\nVictory is the ultimate goal of any competitive engagement."
    
    all_text = doc_chess + "\n\n" + doc_strategy
    passages = [p.strip() for p in all_text.split("\n\n") if p.strip()]
    
    config = TextDomainConfig.from_passages(passages, max_vocab=150)
    agent = ReaderAgent(config)

    # 2. Sequential Mapping & Thematic Induction (L4)
    print("Phase 1: Induction of Narrative Themes (L4)...")
    agent.observe_document(doc_chess, min_length=15)
    agent.observe_document(doc_strategy, min_length=15)
    agent.build_topic_clusters(n_clusters=6) # 3 per domain ideally
    
    agent.learn_thematic_transitions(0)
    agent.learn_thematic_transitions(1)
    
    # 3. Concept Stabilization (L5)
    print("Phase 2: Stabilization of Universal Concepts (L5)...")
    # We want to see if topics like "Game/Strategy" and "Checkmate/Victory" consolidate
    n_concepts = agent.stabilize_universal_concepts(n_concepts=2)
    print(f"  Stabilized {n_concepts} Universal Concepts.")

    # 4. Verify Concept Grouping
    print("\nPhase 3: Verifying Conceptual Abstraction...")
    # Find topic for "checkmate" and "victory"
    mu_checkmate = config.encode_passage("checkmate")
    mu_victory = config.encode_passage("victory")
    
    topic_keys = [k for k in agent.patterns if k.startswith("topic_")]
    
    def get_best_topic(mu):
        best_t = None
        best_sim = -1.0
        for k in topic_keys:
            sim = np.dot(mu, agent.patterns[k].mu) / (np.linalg.norm(mu) * np.linalg.norm(agent.patterns[k].mu) + 1e-9)
            if sim > best_sim:
                best_sim = sim
                best_t = agent.patterns[k]
        return best_t

    t_checkmate = get_best_topic(mu_checkmate)
    t_victory = get_best_topic(mu_victory)
    
    c_checkmate = t_checkmate.metadata.get("parent_concept")
    c_victory = t_victory.metadata.get("parent_concept")
    
    print(f"  'Checkmate' Topic belongs to Concept {c_checkmate}")
    print(f"  'Victory' Topic belongs to Concept {c_victory}")
    
    # 5. Concept-Aware Retrieval
    print("\nPhase 4: Verifying Concept-Aware Retrieval...")
    query = "winning the match"
    result = agent.query_via_concepts(query)
    print(f"  Query: '{query}'")
    print(f"  Result: '{result}'")

    if c_checkmate == c_victory and result is not None:
        print("\n  SUCCESS: Agent abstracted 'Checkmate' and 'Victory' into the same Concept!")
    elif result is not None:
        print("\n  PARTIAL SUCCESS: Retrieval worked, but concepts remained distinct.")
    else:
        print("\n  FAILURE: Conceptual understanding not achieved.")

if __name__ == "__main__":
    run_experiment()
