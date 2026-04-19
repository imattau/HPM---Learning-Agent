"""
SP-Web6: Librarian Specialist Test.
Tests the division of labor between ReaderAgent (Perception) and LibrarianAgent (Concept Discovery).
"""
import os
import time
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.librarian_agent import LibrarianAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

def run_librarian_test():
    print("================================================================================")
    print("SP-Web6: Librarian Specialist & Conceptual Mapping")
    print("================================================================================")

    # Use a fresh test forest
    test_kb = "data/test_librarian"
    if not os.path.exists(test_kb):
        os.makedirs(test_kb)
        print(f"  [INIT] Created new knowledge base at {test_kb}")
    else:
        print(f"  [RESUME] Using existing knowledge base at {test_kb}")

    shared_forest = TieredForest(D=100, cold_dir=test_kb)
    orchestrator = AgentOrchestrator(shared_forest)
    
    # 1. Initialize Society
    text_config = TextDomainConfig.from_passages(["knowledge", "wisdom", "discovery", "science"])
    
    librarian = LibrarianAgent(text_config, forest=shared_forest)
    reader = ReaderAgent(text_config, forest=shared_forest, librarian_agent=librarian)
    
    orchestrator.register(librarian)
    orchestrator.register(reader)
    
    # 2. Ingest Multi-Domain Text
    texts = [
        ("The force of gravity attracts massive objects toward each other.", "Physics Basics"),
        ("Calculus is the mathematical study of continuous change.", "Math Basics"),
        ("Gravity in physics can be described using calculus equations.", "Synthesis")
    ]
    
    print("\nPhase 1: Ingesting Text via ReaderAgent...")
    for text, title in texts:
        print(f"  Ingesting: '{title}'")
        reader.ingest_text(text, title=title)
        
    # 3. Verify Librarian Discovery
    print("\nPhase 2: Verifying Librarian Concept Discovery...")
    topics = [n for n in shared_forest.active_nodes() if n.relation_type == "topic"]
    print(f"  Total Topics Discovered: {len(topics)}")
    for t in topics:
        print(f"    - Topic: {t.metadata.get('word')} (ID: {t.id})")
        
    # 4. Cross-Domain Search
    print("\nPhase 3: Cross-Domain Semantic Search...")
    query = "physical forces"
    results = librarian.search_knowledge(query)
    print(f"  Search Results for '{query}':")
    for res in results:
        label = res.metadata.get('word') or res.metadata.get('title') or res.id
        print(f"    - Found: {label} [{res.relation_type}]")
        
    # 5. Thematic Summary
    print("\nPhase 4: Thematic Synthesis...")
    gravity_topic = shared_forest.get("topic_gravity")
    if gravity_topic:
        summary = librarian.summarize_theme(gravity_topic)
        print(f"  Librarian Summary: {summary}")

    print("\n[SUCCESS] SP-Web6 Librarian Specialist test completed.")

if __name__ == "__main__":
    run_librarian_test()
