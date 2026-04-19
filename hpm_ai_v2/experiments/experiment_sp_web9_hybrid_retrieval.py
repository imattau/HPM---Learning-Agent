"""
Experiment SP-Web9: Hybrid Retrieval (Knowledge Graph + FAISS)
Verifies dictionary POS grounding and structured factual reasoning.
"""
from __future__ import annotations
import os
import time
import numpy as np
from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.knowledge_graph_agent import KnowledgeGraphAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.knowledge_graph_domain import KnowledgeGraphDomainConfig

def run_hybrid_retrieval_experiment():
    print("================================================================================")
    print("Experiment SP-Web9: Hybrid Retrieval (Knowledge Graph + FAISS)")
    print("================================================================================")

    # RESUME from the existing scientific knowledge base
    knowledge_base = "data/scientific_curiosity_v2"
    if not os.path.exists(knowledge_base):
        print(f"Error: Knowledge base {knowledge_base} not found. Run SP-Web7 first.")
        # Fallback to local test dir if SP-Web7 hasn't run
        knowledge_base = "data/hybrid_retrieval_test"
        if not os.path.exists(knowledge_base):
            os.makedirs(knowledge_base)
    
    # 1. Initialize Society
    print(f"Phase 1: Initializing Society (Loading {knowledge_base})...")
    
    # Load D from forest_meta.json automatically
    shared_forest = TieredForest(D=None, cold_dir=knowledge_base)
    D = shared_forest._D
    s_dim = 4
    
    # Reconstruct text config concepts to match D
    concepts = ["ai", "logic", "tesla", "inventor"]
    needed = D - (2 * s_dim)
    if len(concepts) < needed:
        concepts += [f"pad_{i}" for i in range(needed - len(concepts))]
    else:
        concepts = concepts[:needed]
        
    text_config = TextDomainConfig(concepts=concepts, idf={}, s_dim=s_dim)
    orchestrator = AgentOrchestrator(shared_forest)
    
    reader = ReaderAgent(text_config, forest=shared_forest)
    dictionary = DictionaryAgent(text_config, reader_agent=reader, forest=shared_forest)
    reader.dictionary_agent = dictionary
    
    kg_config = KnowledgeGraphDomainConfig(s_dim=s_dim)
    kg_agent = KnowledgeGraphAgent(kg_config, forest=shared_forest)
    
    orchestrator.register(reader)
    orchestrator.register(dictionary)
    orchestrator.register(kg_agent)
    
    # 2. Verify Dictionary POS Grounding & Native Correction
    print("\nPhase 2: Verifying Dictionary POS Grounding & Native Correction...")
    
    # We test the new heuristic and correction in DictionaryAgent.
    # If the KB already has "noun" for "high", the agent should correct it.
    words_to_test = ["high", "led", "ai"]
    for word in words_to_test:
        def_node = dictionary.lookup(word)
        if def_node:
            pos = def_node.metadata.get("pos")
            print(f"  [DICT] '{word}' -> Current POS: {pos}")
        else:
            print(f"  [DICT] Could not find definition for '{word}'")

    # Demonstate Pruning
    print("\nPhase 2b: Structural Pruning...")
    # Add a low-weight 'junk' node
    junk_id = "pattern_noise_999"
    junk_node = HFN(mu=np.zeros(D), sigma=np.ones(D), id=junk_id)
    shared_forest.register(junk_node)
    dictionary.observer.penalize_id(junk_id, penalty=0.999) # Drop weight to near zero
    
    before_count = len(shared_forest)
    removed = dictionary.observer.prune(min_weight=1e-3)
    after_count = len(shared_forest)
    print(f"  [PRUNE] Removed {removed} low-utility nodes. Forest size: {before_count} -> {after_count}")

    # 3. Knowledge Graph Reasoning
    print("\nPhase 3: Knowledge Graph Reasoning (Wikidata)...")
    # We search for Nikola Tesla on Wikidata
    print("  [KG] Querying Wikidata for Nikola Tesla...")
    kg_nodes = kg_agent.find_facts("Nikola Tesla")
    
    if kg_nodes:
        print(f"  [KG] Successfully imported {len(kg_nodes)} nodes related to Nikola Tesla.")
        for node in kg_nodes[:3]:
            print(f"    - {node.id}: {node.metadata.get('label')}")
    else:
        print("  [KG] No nodes returned from Wikidata. (Likely network issue or API limit)")
        # Fallback to manual fact enrichment
        print("  [KG] Falling back to manual fact enrichment...")
        kg_agent.enrich_forest_with_fact("Nikola Tesla", "invented", "Alternating Current")
        kg_agent.enrich_forest_with_fact("Nikola Tesla", "born_in", "Smiljan")

    # 4. Hybrid Retrieval (FAISS + Graph)
    print("\nPhase 4: Hybrid Retrieval...")
    # Add a semantic pattern to the forest
    pattern_text = "Alternating current is a form of electric power transmission."
    reader.ingest_text(pattern_text, title="AC_Definition")
    
    # Search for something related
    query = "Who invented AC?"
    mu_query = text_config.encode_passage(query)
    
    # Accelerated Retrieval
    t_start = time.time()
    matches = shared_forest.retrieve(mu_query, k=5)
    t_end = time.time()
    
    print(f"  [RETRIEVAL] Found {len(matches)} matches in {t_end-t_start:.6f}s.")
    for m in matches[:3]:
        rel = m.relation_type if hasattr(m, "relation_type") else "unknown"
        print(f"    - {m.id} ({rel})")

    print("\n[SUCCESS] SP-Web9 Hybrid Retrieval experiment completed.")

if __name__ == "__main__":
    run_hybrid_retrieval_experiment()
