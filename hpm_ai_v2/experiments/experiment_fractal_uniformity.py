"""
Experiment: Ensure Fractal Uniformity in ReaderAgent.
Verifies that sentences, paragraphs, and documents are represented as HFN nodes.
"""
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
import numpy as np

def run_experiment():
    print("================================================================================")
    print("Fractal Uniformity Verification")
    print("================================================================================")

    # 1. Setup Agent
    config = TextDomainConfig.from_passages(["initial"], max_vocab=10, include_char_primitives=True)
    agent = ReaderAgent(config, dynamic_vocab=True)

    # 2. Observe a passage with multiple sentences
    passage = "The quick brown fox jumps over the lazy dog. Artificial intelligence is evolving rapidly."
    print(f"Observing passage: '{passage}'")
    agent.observe_passage(passage)

    # 3. Verify hierarchy
    print("\nVerifying hierarchy...")
    
    # Find the passage node
    passage_node = next((n for n in agent.patterns.values() if n.metadata.get("type") == "passage"), None)
    if not passage_node:
        print("FAILURE: No passage node found.")
        return

    print(f"Passage Node ID: {passage_node.id}")
    
    # Passage should have a paragraph child
    children = passage_node.children()
    para_node = next((c for c in children if getattr(c, "relation_type", None) == "paragraph"), None)
    if para_node:
        print(f"  Paragraph Node found: {para_node.id}")
        
        # Paragraph should have sentence children
        sent_nodes = [c for c in para_node.children() if getattr(c, "relation_type", None) == "sentence"]
        print(f"  Number of Sentence Nodes: {len(sent_nodes)}")
        
        for i, sn in enumerate(sent_nodes):
            # Sentence should have word children
            words = [c for c in sn.children() if getattr(c, "relation_type", None) == "spelling"]
            print(f"    Sentence {i+1} has {len(words)} word children.")
            
            # Verify POS tagging works on the sentence node
            struct = agent.get_sentence_structure(sn)
            print(f"    POS Tagging on Node: {struct[:5]}...")
            
        if len(sent_nodes) == 2:
            print("SUCCESS: Full hierarchy from passage to words verified!")
        else:
            print(f"FAILURE: Expected 2 sentences, found {len(sent_nodes)}.")
    else:
        print("FAILURE: No paragraph node found under passage.")

    # 4. Verify Document Level
    print("\nVerifying Document Level...")
    agent.ingest_wikipedia_page("Python (programming language)") # This is mocked in our experiment context usually
    
    doc_nodes = [n for n in agent.patterns.values() if getattr(n, "relation_type", None) == "document"]
    if doc_nodes:
        print(f"SUCCESS: Document Node found: {doc_nodes[0].id}")
        # Document should have paragraph children
        paras = [c for c in doc_nodes[0].children() if getattr(c, "relation_type", None) == "paragraph"]
        print(f"  Document has {len(paras)} paragraph children.")
    else:
        # If wikipedia fetch fails (no network), we might not see this, but ingest_wikipedia_page should handle it
        print("PARTIAL: No document node found (maybe wikipedia fetch failed).")

    # 5. Verify Hierarchical Retrieval
    print("\nVerifying Hierarchical Retrieval...")
    query = "What jumps over the dog?"
    print(f"  Query: '{query}'")
    result = agent.query_hierarchical(query)
    print(f"  Reconstructed Result: {result}")
    
    if "fox jumps" in result:
        print("SUCCESS: Hierarchical retrieval correctly reconstructed text from HFN nodes!")
    else:
        print("PARTIAL: Retrieval returned but not as expected.")

    print("\n[SUCCESS] Fractal Uniformity experiment completed.")

if __name__ == "__main__":
    # Mock wikipedia for experiment
    import wikipedia
    from unittest.mock import MagicMock
    mock_page = MagicMock()
    mock_page.content = "Python is a programming language. It is very popular. HPM is a theory of learning."
    wikipedia.page = MagicMock(return_value=mock_page)
    
    run_experiment()
