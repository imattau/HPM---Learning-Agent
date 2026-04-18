"""
SP-Writer 1: Circular Cooperative Intelligence Loop.
Demonstrates Reader, Writer, and WebAgent collaborating in a shared forest.
"""
import os
import shutil
from unittest.mock import MagicMock
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.domains.web_domain import WebDomainConfig
from hpm_ai_v2.agents.web_agent import WebAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
import hpm_ai_v2.agents.mixins.web_fetch as wf

def run_experiment():
    print("================================================================================")
    print("SP-Writer 1: Circular Cooperative Intelligence Loop")
    print("================================================================================")

    if os.path.exists("data/writer_forest"):
        shutil.rmtree("data/writer_forest")

    # 1. Setup Shared Environment
    print("Phase 1: Initialising Cooperative Agent Triad...")
    
    # Text config with large fixed vocab for stability
    text_config = TextDomainConfig.from_passages(["The cat chases the mouse."], max_vocab=1000, include_char_primitives=True)
    m_dim = text_config.m_dim
    
    shared_forest = TieredForest(D=m_dim, cold_dir="data/writer_forest", hot_cap=1000)

    # Web Agent
    web_config = WebDomainConfig(force_dim=text_config.DIM)
    web_agent = WebAgent(web_config, forest=shared_forest)

    # Reader Agent
    reader_agent = ReaderAgent(text_config, forest=shared_forest, web_agent=web_agent, dynamic_vocab=False)

    # Writer Agent
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, forest=shared_forest)

    # Mock Web
    MOCK_WEB = {
        "deep learning": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    }
    def mock_fetch(url):
        topic = url.split("/")[-1].lower().replace("_", " ")
        return MOCK_WEB.get(topic, f"General information about {topic}.")
    
    wf.fetch_url = MagicMock(side_effect=mock_fetch)

    # 2. Understanding & Generation
    print("\nPhase 2: Semantic Extraction and Synthesis...")
    seed = "The cat chases the mouse."
    print(f"  Reader observing: '{seed}'")
    reader_agent.observe_passage(seed)
    
    roles = reader_agent.extract_roles(seed)
    print(f"  Extracted roles: {roles}")
    
    new_sent = writer_agent.generate_sentence(roles["PREDICATE"], roles["AGENT"], roles["PATIENT"])
    print(f"  Writer generated new sentence: '{new_sent}'")

    # 3. Natural Answer
    print("\nPhase 3: Natural Language Question Answering...")
    q1 = "What does the cat chase?"
    print(f"  Q: {q1}")
    a1 = writer_agent.answer_natural(q1)
    print(f"  A: {a1}")

    # 4. Meta-Cognitive Discovery Loop
    print("\nPhase 4: Identifying Gaps and Requesting Knowledge...")
    q2 = "What is deep learning?"
    print(f"  Q: {q2}")
    a2_initial = writer_agent.answer_natural(q2)
    print(f"  Initial A: {a2_initial}")
    
    print("\n  Writer identifies knowledge gap, triggering request...")
    success = writer_agent.request_knowledge("Deep learning")
    
    if success:
        print("\n  Retrying answer after knowledge acquisition...")
        a2_final = writer_agent.answer_natural(q2)
        print(f"  Final A: {a2_final}")
    else:
        print("  FAILURE: Knowledge acquisition failed.")

    # 5. Forest Integrity
    print("\nPhase 5: Verifying Cross-Domain Linking...")
    # Check if we have a document for Deep learning and if it's linked
    docs = [n for k, n in reader_agent.patterns.items() if k.startswith("document_")]
    for d in docs:
        derived_edges = [e for e in d._edges if e.relation == "derived_from"]
        if derived_edges:
            print(f"  SUCCESS: Document '{d.id}' linked to source web resource.")
            break
    else:
        print("  FAILURE: No linked documents found in shared forest.")

    print(f"\n  Final Forest size: {len(shared_forest._hot)} nodes")
    print("\n[SUCCESS] SP-Writer 1 experiment completed.")

if __name__ == "__main__":
    run_experiment()
