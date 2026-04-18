"""
SP-Reader 9: Open-Domain Wikipedia Ingestion.
Demonstrates autonomous ingestion of Wikipedia pages and dynamic vocabulary learning.
"""
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
import os
from pathlib import Path

KNOWLEDGE_BASE_DIR = "data/knowledge_base/reader_wikipedia"

def run_experiment():
    print("================================================================================")
    print("SP-Reader 9: Open-Domain Wikipedia Ingestion")
    print("================================================================================")

    # 1. Setup Reader Agent with character primitives and dynamic vocab
    print("Phase 1: Initialising Reader Agent...")
    if Path(KNOWLEDGE_BASE_DIR).exists():
        import shutil
        shutil.rmtree(KNOWLEDGE_BASE_DIR)
        
    config = TextDomainConfig.from_passages(["initial vocab"], max_vocab=50, include_char_primitives=True)
    agent = ReaderAgent(config, dynamic_vocab=True, cold_dir=KNOWLEDGE_BASE_DIR)

    # 2. Ingest Wikipedia Pages
    print("\nPhase 2: Ingesting Wikipedia Pages (Mocked)...")
    
    # Mock data
    MOCK_PAGES = {
        "Python (programming language)": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
        "Artificial intelligence": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals."
    }
    
    titles = list(MOCK_PAGES.keys())
    
    for title in titles:
        print(f"  Processing '{title}'...")
        text = MOCK_PAGES[title]
        sentences = agent.sentence_splitter.split(text)
        from hpm_ai_v2.utils.text_chunker import chunk_passages
        passages = chunk_passages(sentences, window_size=2, overlap=1)
        
        print(f"  Ingesting {len(passages)} passages...")
        for p in passages:
            agent.observe_passage(p)
            
        print(f"  Current vocab size: {len(agent.config.concepts)}")

    # 3. Build Hierarchy
    print("\nPhase 3: Building Knowledge Hierarchy (L3-L5)...")
    agent.build_topic_clusters(n_clusters=8)
    agent.stabilize_universal_concepts(n_concepts=4)
    print(f"  Hierarchy built: {len(agent.patterns)} patterns in total.")

    # 4. Hierarchical Querying
    print("\nPhase 4: Hierarchical Querying...")
    queries = [
        ("What is Python?", "programming language"),
        ("What does AI mean?", "Artificial intelligence"),
    ]

    for q, expected in queries:
        print(f"  Query: '{q}'")
        result = agent.query_hierarchical(q)
        if result:
            print(f"  Result: {result[:150]}...")
            if expected.lower() in result.lower():
                print(f"  SUCCESS: Found '{expected}' in retrieved passage!")
            else:
                print(f"  PARTIAL: Passage retrieved but '{expected}' not found.")
        else:
            print("  FAILURE: No passage retrieved.")

    # 5. Verify Dynamic Word Macros
    print("\nPhase 5: Verifying Dynamic Word Macros...")
    new_word = "programming"
    if new_word in agent.config.concepts:
        word_id = f"word_spelling_{new_word}"
        if word_id in agent.patterns:
            print(f"  SUCCESS: Dynamic word macro '{word_id}' exists!")
        else:
            print(f"  FAILURE: Macro for '{new_word}' missing.")
    else:
        print(f"  FAILURE: '{new_word}' not added to vocabulary.")

    # 6. Persistence
    print("\nPhase 6: Saving Knowledge Base...")
    agent.save_agent(KNOWLEDGE_BASE_DIR)
    print(f"  Knowledge base saved to {KNOWLEDGE_BASE_DIR}")

if __name__ == "__main__":
    run_experiment()
