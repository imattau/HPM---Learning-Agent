"""
SP-Web 3: Fractal Configuration Verification Experiment.
Demonstrates HFN-native bootstrapping: Agents re-derive their manifold 
configuration (vocabulary, dimensions, IDF) purely from the HFN forest.
"""
import os
import shutil
import numpy as np
from typing import List
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Web 3: Fractal Configuration - HFN-Native Bootstrapping")
    print("================================================================================")

    knowledge_base = "data/fractal_config_test"

    # STAGE 1: Knowledge Acquisition & Fractal Saving
    print("\nPhase 1: Knowledge Acquisition & Fractal Saving...")
    
    # Initialize with a fresh config and marathon-style ingestion
    initial_passages = ["The fractal nature of intelligence is emergent."]
    text_config = TextDomainConfig.from_passages(initial_passages, max_vocab=500, include_char_primitives=True)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=1000)
    
    reader_agent = ReaderAgent(text_config, forest=shared_forest, dynamic_vocab=True)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, forest=shared_forest)
    
    # Ingest a complex sentence to expand vocab and increase forest complexity
    complex_text = "HPM agents learn by progressively discovering, stabilizing, and refining hierarchical patterns."
    reader_agent.ingest_text(complex_text, title="HPM Philosophy")
    
    print(f"  [STAGE 1] Forest built with {len(shared_forest)} nodes.")
    print(f"  [STAGE 1] Vocabulary size: {len(text_config.concepts)}")
    print(f"  [STAGE 1] Dimension D: {text_config.m_dim}")

    # Explicitly save state (this triggers config.save_to_forest)
    reader_agent.save_state()
    writer_agent.save_state()
    shared_forest.save_to_cold()
    
    # Delete the config.pkl if it was created (to prove we don't need it)
    config_pkl = os.path.join(knowledge_base, "config.pkl")
    if os.path.exists(config_pkl):
        os.remove(config_pkl)
        print("  [STAGE 1] Deleted config.pkl (proving HFN-native intent).")

    # STAGE 2: Fractal Bootstrapping ("The Re-Awakening")
    print("\nPhase 2: Fractal Bootstrapping ('The Re-Awakening')...")
    
    # Re-initialize agents providing ONLY the cold_dir.
    # BaseHFNAgent should now autodetect D and load the config from HFNs.
    
    print("  [STAGE 2] Re-initializing Reader and Writer with NO config...")
    # We pass config=None explicitly
    new_reader = ReaderAgent(config=None, cold_dir=knowledge_base)
    new_writer = WriterAgent(config=None, reader_agent=new_reader, forest=new_reader.forest)
    
    print(f"\n  [SUCCESS] New Reader m_dim: {new_reader.m_dim}")
    print(f"  [SUCCESS] New Reader vocabulary size: {len(new_reader.config.concepts)}")
    
    # Check if dimensions match Stage 1
    if new_reader.m_dim == text_config.m_dim:
        print(f"  [SUCCESS] Dimension D correctly autodetected and restored.")
    else:
        print(f"  [FAILURE] Dimension mismatch! Expected {text_config.m_dim}, got {new_reader.m_dim}")

    # Check if vocabulary was restored
    if len(new_reader.config.concepts) == len(text_config.concepts):
        print(f"  [SUCCESS] Vocabulary and manifold index correctly reconstructed from HFNs.")
    else:
        print(f"  [FAILURE] Vocabulary mismatch! Expected {len(text_config.concepts)}, got {len(new_reader.config.concepts)}")

    # Phase 3: Verification via QA
    print("\nPhase 3: Verification via QA (Answering from memory)...")
    q1 = "What do HPM agents learn?"
    print(f"  Q: {q1}")
    a1 = new_writer.answer_natural(q1)
    print(f"  A: {a1}")
    
    if "hierarchical patterns" in a1.lower():
        print("  [SUCCESS] Agent successfully answered from restored fractal memory.")
    else:
        print("  [FAILURE] Agent could not retrieve the correct answer.")

    print("\n[SUCCESS] SP-Web 3 Fractal Configuration experiment completed.")

if __name__ == "__main__":
    run_experiment()
