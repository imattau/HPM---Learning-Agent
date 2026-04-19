"""
SP-Web5: Cross-Domain Synthesis Test (v2 - Specialized Society).
Tests WriterAgent's ability to synthesize knowledge from Physics, Math, and Sentiment using specialized domains.
"""
import os
import time
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.physics_domain import PhysicsDomainConfig
from hpm_ai_v2.domains.math_domain import MathDomainConfig
from hpm_ai_v2.domains.sentiment_domain import SentimentDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.physics_agent import PhysicsAgent
from hpm_ai_v2.agents.math_agent import MathAgent
from hpm_ai_v2.agents.sentiment_agent import SentimentAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

def run_synthesis_test():
    print("================================================================================")
    print("SP-Web5: Cross-Domain Knowledge Synthesis (Specialized Society)")
    print("================================================================================")

    knowledge_base = "data/scientific_curiosity"
    if not os.path.exists(knowledge_base):
        print(f"Error: Knowledge base {knowledge_base} not found. Run SP-Web4 first.")
        return

    # 1. Initialize Society from existing Forest
    print("Phase 1: Re-Initializing Scientific Society from Disk...")
    t0 = time.time()
    shared_forest = TieredForest(D=None, cold_dir=knowledge_base, library_dirs=["data/library"])
    print(f"  [TIME] Forest Loaded ({len(shared_forest)} nodes, D={shared_forest._D}): {time.time()-t0:.2f}s")
    
    # 2. Specialized Configs (All sharing the same M-dimension)
    D = shared_forest._D
    s_dim = 4
    orchestrator = AgentOrchestrator(shared_forest)
    
    print("\nPhase 2: Initializing Specialized Agents...")
    
    # Reconstruct concepts to match D
    def make_config(cls, base_concepts):
        # Pad concepts to match forest dimension D
        needed = D - (2 * s_dim)
        concepts = list(base_concepts)
        if len(concepts) < needed:
            concepts += [f"pad_{i}" for i in range(needed - len(concepts))]
        elif len(concepts) > needed:
            concepts = concepts[:needed]
            
        if cls == TextDomainConfig:
            return cls(concepts=concepts, idf={}, s_dim=s_dim)
        return cls(concepts=concepts, s_dim=s_dim)

    # Text Domain (Reader & Dictionary)
    text_config = make_config(TextDomainConfig, [f"concept_{i}" for i in range(100)])
    
    t_reader = time.time()
    reader = ReaderAgent(text_config, forest=shared_forest)
    print(f"    - Reader initialized: {time.time()-t_reader:.2f}s")
    
    dictionary = DictionaryAgent(text_config, reader_agent=reader, forest=shared_forest)
    reader.dictionary_agent = dictionary
    
    # Physics Domain
    physics_concepts = [
        "VAR_MASS", "VAR_ACCEL", "VAR_FORCE", "VAR_VELOCITY", "VAR_TIME", "VAR_DIST",
        "OP_NEWTON_2", "OP_VELOCITY", "OP_ACCEL", "OP_GRAVITY",
        "UNIT_KG", "UNIT_MS2", "UNIT_N", "UNIT_M", "UNIT_S",
        "PHYS_LAW", "PHYS_PROBLEM"
    ]
    physics_config = make_config(PhysicsDomainConfig, physics_concepts)
    physics = PhysicsAgent(physics_config, forest=shared_forest)
    print(f"    - Physics initialized")
    
    # Math Domain
    math_concepts = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_POW", "OP_SIN", "OP_COS", "OP_DERIVE", "OP_INTEGRATE"]
    math_config = make_config(MathDomainConfig, math_concepts)
    math = MathAgent(math_config, forest=shared_forest)
    print(f"    - Math initialized")
    
    # Sentiment Domain
    sentiment_concepts = ["SENTIMENT_POSITIVE", "SENTIMENT_NEGATIVE", "SENTIMENT_NEUTRAL", "AFFECT_VALENCE", "AFFECT_AROUSAL"]
    sentiment_config = make_config(SentimentDomainConfig, sentiment_concepts)
    sentiment = SentimentAgent(sentiment_config, forest=shared_forest)
    print(f"    - Sentiment initialized")
    
    # Writer (Synthesizer)
    writer = WriterAgent(text_config, reader_agent=reader, forest=shared_forest)
    print(f"    - Writer initialized")
    
    # Register all
    orchestrator.register(reader)
    orchestrator.register(dictionary)
    orchestrator.register(physics)
    orchestrator.register(math)
    orchestrator.register(sentiment)
    orchestrator.register(writer)
    
    # Set up collaborations
    writer.sentiment_agent = sentiment
    writer.math_agent = math
    writer.physics_agent = physics
    writer.dictionary_agent = dictionary

    # 3. Synthesis Prompt
    prompt = "Discuss the emotional impact of physical forces like gravity in mathematical terms."
    print(f"\nPhase 3: Processing Cross-Domain Prompt...")
    print(f"  PROMPT: '{prompt}'")
    
    t_gen = time.time()
    response = writer.answer_natural(prompt)
    print(f"  [TIME] Response Generated: {time.time()-t_gen:.2f}s")
    
    print("\nPhase 4: Agent Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)

    # 5. Chain-of-Thought Inspection
    print("\nPhase 5: Synthesis Inspection...")
    
    # Check for sentiment on the answer itself
    res_node = sentiment.analyze_sentence(reader.ingest_text(response, title="agent_response"))
    print(f"  [AFFECT] Response Sentiment: {res_node.metadata.get('score'):.2f} ({res_node.metadata.get('label')})")

    print("\n[SUCCESS] SP-Web5 Synthesis test completed.")

if __name__ == "__main__":
    run_synthesis_test()
