"""
SP-Sci Test 1: The Unified Science Challenge (Physics).
Agent learns physics laws from text and solves multi-step word problems.
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
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Sci Test 1: Unified Science Challenge (Physics)")
    print("================================================================================")

    knowledge_base = "data/science_exam_test"

    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Multi-Agent Environment...")
    text_config = TextDomainConfig.from_passages(["physics reasoning"], max_vocab=1000, include_char_primitives=True)
    math_config = MathDomainConfig(s_dim=20)
    phys_config = PhysicsDomainConfig(s_dim=20)
    
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    
    math_agent = MathAgent(math_config, forest=shared_forest)
    physics_agent = PhysicsAgent(phys_config, forest=shared_forest)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, math_agent=math_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, math_agent=math_agent, physics_agent=physics_agent, forest=shared_forest)
    
    # Register collaborators for consistent dimension updates
    reader_agent.collaborators.extend([physics_agent, writer_agent])

    # 2. Reading Physics Laws
    print("\nPhase 2: Learning Physics Laws from Text...")
    physics_text = """
    Newton's second law of motion describes the relationship between an object's mass and the amount of force needed to accelerate it.
    The law states that force is equal to mass multiplied by acceleration (F = m * a).
    In the International System of Units (SI), mass is measured in kilograms (kg), 
    acceleration is measured in meters per second squared (m/s2), and force is measured in Newtons (N).
    
    Example: An object with a mass of 10.0 kg and an acceleration of 5.0 m/s2 has a force of 50.0 N.
    """
    
    print("  Ingesting Physics content...")
    reader_agent.ingest_text(physics_text, title="Physics: Newton's Laws")
    print(f"  Forest Nodes: {len(shared_forest)}")

    # 3. The Science Exam
    print("\nPhase 3: The Physics Exam...")
    
    exam_questions = [
        {
            "id": "Q1 (Retrieval)",
            "q": "What is Newton's second law?",
            "eval": lambda a: "force" in a.lower() and "mass" in a.lower() and "acceleration" in a.lower()
        },
        {
            "id": "Q2 (Physics Derivation)",
            "q": "What is the force of an object with a mass of 12.0 kg and an acceleration of 3.0 m/s2?",
            "eval": lambda a: "36.0" in a or "36 N" in a
        },
        {
            "id": "Q3 (Physics Derivation)",
            "q": "Calculate the acceleration of an object with a force of 100.0 N and a mass of 20.0 kg.",
            "eval": lambda a: "5.0" in a or "5 m/s^2" in a
        }
    ]
    
    passed_count = 0
    for eq in exam_questions:
        print(f"\n  [{eq['id']}] Q: {eq['q']}")
        answer = writer_agent.answer_natural(eq['q'])
        print(f"    A: {answer}")
        
        is_correct = eq['eval'](answer)
        if is_correct:
            print("    [RESULT] CORRECT")
            passed_count += 1
        else:
            print("    [RESULT] INCORRECT")

    # 4. Final Grade
    print(f"\nPhase 4: Final Grade...")
    score = (passed_count / len(exam_questions)) * 100
    print(f"  Final Score: {score:.1f}% ({passed_count}/{len(exam_questions)})")

    if score >= 60:
        print("  [STATUS] AGENT PASSED THE SCIENCE EXAM")
    else:
        print("  [STATUS] AGENT FAILED THE SCIENCE EXAM")

    print("\n[SUCCESS] SP-Sci Test 1 completed.")

if __name__ == "__main__":
    run_experiment()
