"""
SP-Orchestra Test 1: Agent Orchestrator & Collaborative Science.
Verifies centralized synchronization and agent discovery.
"""
import os
import shutil
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.math_domain import MathDomainConfig
from hpm_ai_v2.domains.physics_domain import PhysicsDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.math_agent import MathAgent
from hpm_ai_v2.agents.physics_agent import PhysicsAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Orchestra Test 1: Agent Orchestrator & Collaborative Science")
    print("================================================================================")

    knowledge_base = "data/orchestra_test"
    # Note: No shutil.rmtree here, complying with persistence mandate.

    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Orchestrated Environment...")
    text_config = TextDomainConfig.from_passages(["orchestration basics"], max_vocab=500, include_char_primitives=True)
    math_config = MathDomainConfig(s_dim=20)
    phys_config = PhysicsDomainConfig(s_dim=20)
    
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    
    # Create the Orchestrator
    orchestrator = AgentOrchestrator(shared_forest)
    
    # Create agents
    math_agent = MathAgent(math_config, forest=shared_forest)
    physics_agent = PhysicsAgent(phys_config, forest=shared_forest)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, math_agent=math_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, math_agent=math_agent, physics_agent=physics_agent, forest=shared_forest)

    # 2. Register Agents with Orchestrator
    print("\nPhase 2: Registering Agents...")
    orchestrator.register(reader_agent, "primary_reader")
    orchestrator.register(math_agent, "math_specialist")
    orchestrator.register(physics_agent, "physics_specialist")
    orchestrator.register(writer_agent, "chief_writer")
    
    # Verify discovery
    phys_specialists = orchestrator.find_specialists("physics")
    print(f"  Discovered Physics Specialists: {[a.__class__.__name__ for a in phys_specialists]}")

    # 3. Trigger Vocabulary Expansion (Testing Broadcast)
    print("\nPhase 3: Triggering Manifold Expansion (New Vocab)...")
    expansion_text = "Kinematics is the subfield of physics that describes the motion of points, bodies, and systems without considering the forces that cause them to move."
    reader_agent.ingest_text(expansion_text, title="Kinematics Intro")
    
    print(f"  New Forest Dimension: {shared_forest._D}")
    print(f"  MathAgent Dimension: {math_agent.m_dim}")
    print(f"  PhysicsAgent Dimension: {physics_agent.m_dim}")
    
    assert math_agent.m_dim == shared_forest._D, "MathAgent dimension mismatch!"
    assert physics_agent.m_dim == shared_forest._D, "PhysicsAgent dimension mismatch!"
    print("  [SUCCESS] All agents synchronized to new dimension via broadcast.")

    # 4. Final Verification: Solving a Problem
    print("\nPhase 4: Collaborative Solving...")
    q = "What is the force of an object with a mass of 15.0 kg and an acceleration of 2.0 m/s2?"
    answer = writer_agent.answer_natural(q)
    print(f"    Q: {q}")
    print(f"    A: {answer}")
    
    if "30.0 N" in answer:
        print("    [RESULT] CORRECT (Collaborative reasoning intact)")
    else:
        print("    [RESULT] INCORRECT")

    print("\n[SUCCESS] SP-Orchestra Test 1 completed.")

if __name__ == "__main__":
    run_experiment()
