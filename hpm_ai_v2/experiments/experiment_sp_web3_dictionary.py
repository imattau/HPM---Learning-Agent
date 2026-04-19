"""
SP-Web 3: DictionaryAgent Verification Experiment.
Demonstrates deterministic external knowledge ingestion into the HFN forest
and semantic enrichment via cooperative agents.
"""
import os
import shutil
from typing import Dict, List
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Web 3: DictionaryAgent - Cooperative Semantic Grounding")
    print("================================================================================")

    knowledge_base = "data/research_marathon_forest"

    # 1. Initialize Agents
    print("Phase 1: Initializing Agents...")
    
    import pickle
    config_path = os.path.join(knowledge_base, "config.pkl")
    if os.path.exists(config_path):
        print("  Loading existing TextDomainConfig...")
        with open(config_path, "rb") as f:
            text_config = pickle.load(f)
    else:
        print("  Creating new TextDomainConfig...")
        text_config = TextDomainConfig.from_passages(["initial"], max_vocab=1000, include_char_primitives=True)
        
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=1000)

    # Note: dynamic_vocab=True will trigger proactive dictionary lookups
    reader_agent = ReaderAgent(text_config, forest=shared_forest, dynamic_vocab=True)
    dict_agent = DictionaryAgent(text_config, reader_agent, forest=shared_forest, dict_source="mock")
    writer_agent = WriterAgent(text_config, reader_agent, dictionary_agent=dict_agent, forest=shared_forest)

    # Link dictionary agent back to reader agent for proactive lookups
    reader_agent.dictionary_agent = dict_agent

    print("\nPhase 2: Seeding Knowledge & Proactive Grounding...")
    # The reader agent observes a sentence with a new word ("neural").
    # It will dynamically add it to the vocabulary and trigger dict_agent.lookup("neural").
    reader_agent.observe_passage("The system uses neural processing.")
    
    # Check if the definition was successfully ingested
    def_node_neural = shared_forest.get("definition_neural")
    if def_node_neural:
        print(f"  [SUCCESS] Proactive dictionary lookup created: {def_node_neural.id}")
    else:
        print("  [FAILURE] Definition for 'neural' was not created.")

    print("\nPhase 3: Answer Enrichment via WriterAgent...")
    # Seed the answer into the forest so the reader can find it
    reader_agent.observe_passage("Neural processing is advanced.")
    
    # The writer agent should extract "neural" from the answer and fetch the definition
    q1 = "What is neural processing?"
    print(f"  Q: {q1}")
    a1 = writer_agent.answer_natural(q1)
    print(f"  A: {a1}")
    
    if "nerve" in a1.lower():
        print("  [SUCCESS] WriterAgent successfully enriched answer with dictionary definition.")
    else:
        print("  [FAILURE] WriterAgent did not include dictionary definition.")

    print("\nPhase 4: Conceptual Analogy...")
    # Now we observe "nerve", which was introduced via the dictionary definition.
    reader_agent.observe_passage("The optic nerve transmits signals.")
    
    # Check graph connectivity
    print("\nPhase 5: Knowledge Graph Verification...")
    word_node = shared_forest.get("word_spelling_neural")
    if word_node:
        edges = [e for e in word_node._edges if e.relation == "defined_as"]
        if edges:
            print(f"  [SUCCESS] Found 'defined_as' edge from word macro to definition.")
            print(f"            {word_node.id} --defined_as--> {edges[0].target.id}")
        else:
            print("  [FAILURE] Missing 'defined_as' edge.")
    
    # Print node counts
    nodes = shared_forest.active_nodes()
    def_nodes = [n for n in nodes if n.relation_type == "definition"]
    print(f"\nFinal Statistics:")
    print(f"  Total Forest Nodes: {len(nodes)}")
    print(f"  Definition Nodes: {len(def_nodes)}")
    print(f"  Vocabulary Size: {len(text_config.concepts)}")

    print("\n[SUCCESS] SP-Web 3 DictionaryAgent experiment completed.")

if __name__ == "__main__":
    run_experiment()
