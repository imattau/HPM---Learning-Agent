"""
SP-Math 2: Symbolic Derivation & Generative NLP Summary.
Verifies that the agent can COMPUTE math answers and frame them naturally, 
rather than just retrieving pre-existing sentences.
"""
import os
import shutil
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.math_domain import MathDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.math_agent import MathAgent
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Math 2: Symbolic Derivation - Generative Reasoning Loop")
    print("================================================================================")

    knowledge_base = "data/math_derivation_test"

    # 1. Initialize Agents
    print("Phase 1: Initializing Multi-Agent Environment...")
    text_config = TextDomainConfig.from_passages(["math reasoning"], max_vocab=500, include_char_primitives=True)
    math_config = MathDomainConfig(s_dim=20)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=1000)
    
    math_agent = MathAgent(math_config, forest=shared_forest)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, math_agent=math_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, math_agent=math_agent, forest=shared_forest)

    # 2. Seed Basic Textual Context (No specific answer sentences)
    print("\nPhase 2: Seeding Context (General information only)...")
    context = """
    Differentiation is the process of finding the derivative of a function.
    The result of differentiating a sum of functions is the sum of their derivatives.
    """
    reader_agent.ingest_text(context, title="Differentiation Context")
    print(f"  Forest Nodes: {len(shared_forest)}")

    # 3. Test Symbolic Derivation (Computing the answer)
    print("\nPhase 3: Symbolic Derivation (Computing new knowledge)...")
    
    # This specific expression is NOT in the context.
    # The agent must: 
    # 1. Recognize 'differentiate' intent.
    # 2. Parse 'x**4 + sin(x)'.
    # 3. Apply symbolic primitives to derive '4*x**3 + cos(x)'.
    # 4. Use WriterAgent to frame it.
    
    query = "Differentiate x**4 + sin(x)"
    print(f"  Query: {query}")
    
    # The answer_natural method now prioritizes derivation
    summary = writer_agent.answer_natural(query)
    
    print("\n  [DERIVED & AUDITED SUMMARY]:")
    print("-" * 60)
    print(summary)
    print("-" * 60)

    # 4. Verification
    print("\nPhase 4: Verification...")
    # Expected result components: 4*x**3 and cos(x)
    has_power = "4 * x**3" in summary or "4.0 * x**3.0" in summary
    has_trig = "cos(x)" in summary
    
    if has_power and has_trig:
        print("  [SUCCESS] Agent successfully DERIVED the symbolic answer.")
    elif has_power or has_trig:
        print("  [PARTIAL] Agent derived part of the answer.")
    else:
        print("  [FAILURE] Agent failed to derive the symbolic answer.")

    # Verify self-correction (NLP quality)
    report = writer_agent.auditor.score_text(summary)
    print(f"  Summary Aesthetics: {report['aesthetics']:.2f}")
    if report['aesthetics'] >= 1.0:
        print("  [SUCCESS] NLP framing is polished (capitalized, punctuated).")

    print("\n[SUCCESS] SP-Math 2 Symbolic Derivation experiment completed.")

if __name__ == "__main__":
    run_experiment()
