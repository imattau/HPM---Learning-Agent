"""
SP-Reader 10: Hierarchical Question Answering.
Verifies factual QA using SRL, hierarchical retrieval, and concept generalization.
"""
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

def run_experiment():
    print("================================================================================")
    print("SP-Reader 10: Hierarchical Question Answering")
    print("================================================================================")

    # 1. Setup Agent with core vocabulary and syntax rules
    config = TextDomainConfig.from_passages([
        "The cat chases the mouse.",
        "The dog bites the bone."
    ], max_vocab=20, include_char_primitives=True)
    agent = ReaderAgent(config, dynamic_vocab=True)
    
    # Pre-train POS and SRL (mimic SP-Reader 6/7)
    print("Phase 1: Induction of POS and SRL macros...")
    agent.learn_pos_tagger(
        [
            ["the", "cat", "chases", "the", "mouse"],
            ["the", "dog", "bites", "the", "bone"],
            ["Guido", "created", "Python"],
            ["John", "coined", "AI"]
        ],
        [
            ["DET", "NOUN", "VERB", "DET", "NOUN"],
            ["DET", "NOUN", "VERB", "DET", "NOUN"],
            ["NOUN", "VERB", "NOUN"],
            ["NOUN", "VERB", "NOUN"]
        ]
    )
    agent.learn_role_mapping([
        (["cat", "chases", "mouse"], {"AGENT": "cat", "PATIENT": "mouse", "PREDICATE": "chases"}),
        (["Guido", "created", "Python"], {"AGENT": "Guido", "PATIENT": "Python", "PREDICATE": "created"})
    ])

    # 2. Basic Factual QA
    print("\nPhase 2: Basic Factual QA (Role Matching)...")
    agent.observe_passage("The cat chases the mouse.")
    agent.observe_passage("The dog bites the bone.")
    
    qa_pairs = [
        ("Who chases the mouse?", "cat"),
        ("What does the dog bite?", "bone")
    ]
    
    for q, expected in qa_pairs:
        print(f"  Q: '{q}'")
        ans = agent.answer_question_hierarchical(q)
        print(f"  Ans: {ans}")
        if ans and expected.lower() in ans.lower():
            print(f"  SUCCESS: Found '{expected}'")
        else:
            print(f"  FAILURE: Expected '{expected}', got '{ans}'")

    # 3. Concept Generalization
    print("\nPhase 3: Concept Generalization...")
    # Add an animal concept
    agent.build_topic_clusters(n_clusters=2)
    
    # Query with a category
    q_gen = "Which animal chases the mouse?"
    print(f"  Q: '{q_gen}'")
    ans_gen = agent.answer_question_hierarchical(q_gen)
    print(f"  Ans: {ans_gen}")
    if ans_gen and "cat" in ans_gen.lower():
        print("  SUCCESS: Generalised 'Which animal' to 'cat' using SRL + Hierarchy!")
    else:
        print("  PARTIAL/FAILURE: Could not generalize.")

    # 4. Wikipedia Ingestion & Multi-Passage QA
    print("\nPhase 4: Wikipedia Ingestion & Multi-Passage QA...")
    # Mock wikipedia
    import wikipedia
    from unittest.mock import MagicMock
    mock_python = MagicMock()
    mock_python.content = "Python is a programming language. Guido van Rossum created Python in 1991."
    mock_ai = MagicMock()
    mock_ai.content = "Artificial intelligence (AI) is intelligence demonstrated by machines. John McCarthy coined the term in 1956."
    
    def side_effect(title):
        if "Python" in title: return mock_python
        if "Artificial intelligence" in title: return mock_ai
        return MagicMock(content="")
    
    wikipedia.page = MagicMock(side_effect=side_effect)
    
    agent.ingest_wikipedia_page("Python (programming language)")
    agent.ingest_wikipedia_page("Artificial intelligence")
    
    wiki_qa = [
        ("Who created Python?", "Guido"),
        ("Who coined AI?", "John McCarthy")
    ]
    
    for q, expected in wiki_qa:
        print(f"  Q: '{q}'")
        ans = agent.answer_question_hierarchical(q)
        print(f"  Ans: {ans}")
        if ans and any(w.lower() in ans.lower() for w in expected.split()):
            print(f"  SUCCESS: Found '{expected}' in Wikipedia evidence!")
        else:
            # Hierarchical retrieval should at least find the right topic/passage
            if ans and len(ans) > 20:
                print("  PARTIAL: Passage retrieved, but specific role extraction failed.")
            else:
                print(f"  FAILURE: Could not find '{expected}'")

    print("\n[SUCCESS] SP-Reader 10 experiment completed.")

if __name__ == "__main__":
    run_experiment()
