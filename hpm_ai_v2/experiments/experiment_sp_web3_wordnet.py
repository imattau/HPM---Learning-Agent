"""
SP-Web3: WordNet Integration Test.
Verifies that DictionaryAgent can fetch real definitions and hypernyms from NLTK.
"""
import os
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Web3: WordNet Integration & Lexical Grounding")
    print("================================================================================")

    knowledge_base = "data/wordnet_test"
    
    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Environment...")
    text_config = TextDomainConfig.from_passages(["lexical grounding"], max_vocab=500)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=1000)
    orchestrator = AgentOrchestrator(shared_forest)
    
    # Reader needs to be initialized first so it can be passed to Dictionary
    reader_agent = ReaderAgent(text_config, forest=shared_forest)
    # DictionaryAgent defaults to 'wordnet' now
    dict_agent = DictionaryAgent(text_config, reader_agent=reader_agent, forest=shared_forest)
    reader_agent.dictionary_agent = dict_agent
    
    orchestrator.register(reader_agent)
    orchestrator.register(dict_agent)

    # 2. Testing real word lookup (Not in mock)
    print("\nPhase 2: Looking up 'Gravity' (WordNet)...")
    def_node = dict_agent.lookup("gravity")
    
    if def_node:
        print(f"  Word: {def_node.metadata['word']}")
        print(f"  POS: {def_node.metadata['pos']}")
        print(f"  Definition: {def_node.metadata['definition']}")
        
        # Check for hypernyms (is_a links)
        is_a_links = [e.target.id for e in def_node.edges() if e.relation == "is_a"]
        print(f"  Hypernyms (is_a): {is_a_links}")
        
        if "noun" in def_node.metadata['pos'] and len(is_a_links) > 0:
            print("  [SUCCESS] Real WordNet data fetched and hypernyms extracted.")
        else:
            print("  [WARNING] POS or hypernyms missing.")
    else:
        print("  [ERROR] WordNet lookup failed.")

    # 3. Testing complex word
    print("\nPhase 3: Looking up 'Momentum'...")
    def_node_mom = dict_agent.lookup("momentum")
    if def_node_mom:
        print(f"  Definition: {def_node_mom.metadata['definition']}")
        is_a_links = [e.target.id for e in def_node_mom.edges() if e.relation == "is_a"]
        print(f"  Hypernyms: {is_a_links}")

    # 4. Persistence Check
    print("\nPhase 4: Persisting knowledge base...")
    shared_forest.save_to_cold()

    print("\n[SUCCESS] SP-Web3 WordNet test completed.")

if __name__ == "__main__":
    run_experiment()
