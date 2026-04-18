"""
SP-Live-Web: Real-World Knowledge Acquisition.
Uses WebAgent to fetch LIVE Wikipedia pages and Reader/Writer to process them.
"""
import os
import shutil
import time
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.domains.web_domain import WebDomainConfig
from hpm_ai_v2.agents.web_agent import WebAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent

def run_experiment():
    print("================================================================================")
    print("SP-Live-Web: Real-World Knowledge Acquisition (LIVE)")
    print("================================================================================")

    knowledge_base = "data/live_web_forest"
    if os.path.exists(knowledge_base):
        shutil.rmtree(knowledge_base)

    # 1. Setup Shared Environment
    print("Phase 1: Initialising Cooperative Agent Triad...")
    
    # Text config with large vocab for real-world text
    # We'll use a dynamic vocab to adapt to the live page
    text_config = TextDomainConfig.from_passages(["initial"], max_vocab=2000, include_char_primitives=True)
    m_dim = text_config.m_dim
    
    shared_forest = TieredForest(D=m_dim, cold_dir=knowledge_base, hot_cap=5000)

    # Web Agent
    web_config = WebDomainConfig(force_dim=text_config.DIM)
    web_agent = WebAgent(web_config, forest=shared_forest)

    # Reader Agent
    reader_agent = ReaderAgent(text_config, forest=shared_forest, web_agent=web_agent, dynamic_vocab=True)

    # Writer Agent
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, forest=shared_forest)

    # 2. Live Discovery
    topic = "Quantum computing"
    print(f"\nPhase 2: Live Acquisition of '{topic}'...")
    
    start_time = time.time()
    # We use a very small limit for the live experiment to keep it fast
    # but still enough to show hierarchy creation.
    success = writer_agent.request_knowledge(topic, max_passages=2)
    end_time = time.time()
    
    if not success:
        print("  [ERROR] Failed to fetch live content. Check your internet connection.")
        return

    print(f"  Acquisition completed in {end_time - start_time:.2f} seconds.")
    print(f"  Forest size after ingestion: {len(shared_forest._hot)} nodes")

    # 3. Structural Verification
    print("\nPhase 3: Verifying Structural Integrity...")
    doc_id = f"document_{topic.replace(' ', '_')}"
    if doc_id in reader_agent.patterns:
        doc_node = reader_agent.patterns[doc_id]
        print(f"  SUCCESS: Document node '{doc_id}' created.")
        print(f"  Hierarchy depth: L5 (Document) -> {len(doc_node.children())} Paragraphs")
        
        # Check derivation
        derived_edges = [e for e in doc_node._edges if e.relation == "derived_from"]
        if derived_edges:
            print(f"  SUCCESS: Linked to live source: {derived_edges[0].target.metadata.get('url')}")
    else:
        print(f"  [ERROR] Document node '{doc_id}' missing. Listing available nodes:")
        for k in list(reader_agent.patterns.keys())[:10]:
            print(f"    - {k}")

    # 4. Natural QA over Live Data
    print("\nPhase 4: Natural Language QA over Live Data...")
    questions = [
        f"What is {topic}?",
        "What is a qubit?",
        "Who founded quantum computing?"
    ]
    
    for q in questions:
        print(f"\n  Q: {q}")
        answer = writer_agent.answer_natural(q)
        print(f"  A: {answer}")

    # 5. Summary Generation
    print("\nPhase 5: Automated Summary Synthesis...")
    if doc_id in reader_agent.patterns:
        summary = writer_agent.generate_summary(reader_agent.patterns[doc_id], max_sentences=2)
        print(f"  Synthesized Summary:\n  {summary}")

    print(f"\nFinal Forest Stats:")
    print(f"  Total Hot Nodes: {len(shared_forest._hot)}")
    print(f"  Reader Patterns: {len(reader_agent.patterns)}")
    print(f"  Vocabulary Size: {len(reader_agent.config.concepts)}")

    print("\n[SUCCESS] SP-Live-Web experiment completed.")

if __name__ == "__main__":
    run_experiment()
