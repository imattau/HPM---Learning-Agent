"""
SP-Web Test 1: The Wikipedia Calculus Exam.
Agent reads a Wikipedia-style entry and answers real test questions using retrieval and derivation.
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
    print("SP-Web Test 1: The Wikipedia Calculus Exam")
    print("================================================================================")

    knowledge_base = "data/wiki_exam_test"

    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Multi-Agent Environment...")
    text_config = TextDomainConfig.from_passages(["calculus exam"], max_vocab=1000, include_char_primitives=True)
    math_config = MathDomainConfig(s_dim=20)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    
    math_agent = MathAgent(math_config, forest=shared_forest)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, math_agent=math_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, math_agent=math_agent, forest=shared_forest)

    # 2. Reading the Wikipedia Content
    print("\nPhase 2: Reading 'Differentiation Rules' Wikipedia Snippet...")
    wiki_snippet = """
    # Differentiation Rules
    
    In calculus, differentiation rules are formulas for computing the derivative of a function.
    
    ## The Power Rule
    The power rule states that for any real number n, the derivative of f(x) = x**n is f'(x) = n * x**(n-1).
    This rule allows us to find the slope of any polynomial function.
    
    ## Derivative of Sine and Cosine
    Trigonometric functions also follow specific rules. 
    The derivative of sin(x) is cos(x).
    The derivative of cos(x) is -sin(x).
    
    ## Sum Rule
    The derivative of a sum of two functions is the sum of their derivatives. 
    Mathematically, d/dx [f(x) + g(x)] = f'(x) + g'(x).
    """
    
    print("  Ingesting Wikipedia content...")
    reader_agent.ingest_text(wiki_snippet, title="Wikipedia: Differentiation Rules")
    
    math_nodes = [n for n in shared_forest.active_nodes() if n.relation_type == "math_expr"]
    print(f"  Forest Nodes: {len(shared_forest)}")
    print(f"  Math Concepts Discovered: {len(math_nodes)}")

    # 3. The Exam
    print("\nPhase 3: Taking the Exam...")
    
    exam_questions = [
        {
            "id": "Q1 (Retrieval)",
            "q": "What does the power rule state?",
            "eval": lambda a: "n * x**" in a.lower() or "n*x**" in a.lower()
        },
        {
            "id": "Q2 (Derivation)",
            "q": "What is the derivative of x**6?",
            "eval": lambda a: "6.0 * x**5.0" in a or "6 * x**5" in a
        },
        {
            "id": "Q3 (Complex Derivation)",
            "q": "Differentiate x**3 + sin(x)",
            "eval": lambda a: "3.0 * x**2.0" in a and "cos(x)" in a
        }
    ]
    
    passed_count = 0
    for eq in exam_questions:
        print(f"\n  [{eq['id']}] Q: {eq['q']}")
        
        # Agent formulates answer
        answer = writer_agent.answer_natural(eq['q'])
        
        print(f"    A: {answer}")
        
        # Scoring
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
        print("  [STATUS] AGENT PASSED THE EXAM")
    else:
        print("  [STATUS] AGENT FAILED THE EXAM")

    print("\n[SUCCESS] SP-Web Test 1 completed.")

if __name__ == "__main__":
    run_experiment()
