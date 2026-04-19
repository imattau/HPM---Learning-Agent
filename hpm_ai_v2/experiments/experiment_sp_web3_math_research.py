"""
SP-Web 3: Autonomous Math Research & NLP Summary Experiment.
Integrates WebAgent, ReaderAgent, MathAgent, and WriterAgent (with NLPAuditor).
"""
import os
import shutil
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.math_domain import MathDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.math_agent import MathAgent
from hpm_ai_v2.agents.web_agent import WebAgent
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Web 3: Autonomous Math Research - Integrated Discovery Loop")
    print("================================================================================")

    knowledge_base = "data/math_research_test"

    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Multi-Agent Environment...")
    text_config = TextDomainConfig.from_passages(["math research"], max_vocab=1000, include_char_primitives=True)
    math_config = MathDomainConfig(s_dim=20)
    
    # Shared Forest for all agents
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    
    # Initialize Agents
    web_agent = WebAgent(text_config, forest=shared_forest)
    math_agent = MathAgent(math_config, forest=shared_forest)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, web_agent=web_agent, math_agent=math_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, math_agent=math_agent, forest=shared_forest)

    # 2. Simulated Web Discovery (Learning about Calculus)
    print("\nPhase 2: Simulated Web Discovery...")
    math_web_content = """
    Calculus is a branch of mathematics focused on limits, functions, derivatives, integrals, and infinite series.
    The power rule is a fundamental rule in differentiation. 
    It states that the derivative of x**n is n*x**(n-1).
    For example, the derivative of x**2 is 2*x**1.
    Trigonometric functions also have derivatives.
    The derivative of sin(x) is cos(x).
    Combining these rules, the derivative of x**2 + sin(x) is 2*x + cos(x).
    """
    
    # Ingest the content via ReaderAgent (which now uses MathAgent internally)
    print("  Ingesting mathematical text...")
    doc_node = reader_agent.ingest_text(math_web_content, title="Calculus Basics")
    
    math_nodes = [n for n in shared_forest.active_nodes() if n.relation_type == "math_expr"]
    print(f"  Forest Nodes: {len(shared_forest)}")
    print(f"  Math Nodes Discovered: {len(math_nodes)}")
    for mn in math_nodes[:3]:
        print(f"    - Found Math: {math_agent.renderer.render(mn)}")

    # 3. Generating Audited NLP Summary
    print("\nPhase 3: Generating Audited NLP Summary...")
    
    query = "What is the derivative of x**2 + sin(x)?"
    print(f"  Q: {query}")
    
    # WriterAgent uses its internal NLPAuditor to select the best summary framing
    summary = writer_agent.answer_natural(query)
    
    print("\n  [FINAL AUDITED SUMMARY]:")
    print("-" * 60)
    print(summary)
    print("-" * 60)

    # 4. Verification
    print("\nPhase 4: Verification...")
    if "2 * x + cos(x)" in summary or "2*x + cos(x)" in summary:
        print("  [SUCCESS] Math result correctly extracted and rendered in summary.")
    else:
        # Check if the parts are there
        if "2 * x" in summary and "cos(x)" in summary:
             print("  [SUCCESS] Math components found in summary.")
        else:
             print("  [FAILURE] Math result missing or corrupted in summary.")

    # Check Audit Score (internal)
    report = writer_agent.auditor.score_text(summary)
    print(f"  Summary Quality Score: {report['total_utility']:.3f} (Aes: {report['aesthetics']:.2f})")

    if report['total_utility'] > 0.5:
        print("  [SUCCESS] Summary met quality threshold.")
    else:
        print("  [WARNING] Summary quality is low.")

    print("\n[SUCCESS] SP-Web 3 Math Research experiment completed.")

if __name__ == "__main__":
    run_experiment()
