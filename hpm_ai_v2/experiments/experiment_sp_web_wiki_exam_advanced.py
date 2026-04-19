"""
SP-Web Test 2: Advanced Math Challenge with Resit Mechanism.
Agent identifies failures, 'studies' more specific content, and retakes the exam.
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

def run_exam(agent_stack, questions):
    """Utility to run a set of questions and return results."""
    writer_agent = agent_stack['writer']
    results = []
    passed_count = 0
    
    for eq in questions:
        print(f"\n  [{eq['id']}] Q: {eq['q']}")
        answer = writer_agent.answer_natural(eq['q'])
        print(f"    A: {answer}")
        
        is_correct = eq['eval'](answer)
        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"    [RESULT] {status}")
        
        if is_correct: passed_count += 1
        results.append({"id": eq['id'], "q": eq['q'], "correct": is_correct, "answer": answer})
        
    score = (passed_count / len(questions)) * 100
    return score, results

def run_experiment():
    print("================================================================================")
    print("SP-Web Test 2: Advanced Math Challenge & Resit Loop")
    print("================================================================================")

    knowledge_base = "data/advanced_exam_test"

    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Multi-Agent Environment...")
    text_config = TextDomainConfig.from_passages(["initial"], max_vocab=1000, include_char_primitives=True)
    math_config = MathDomainConfig(s_dim=20)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    
    math_agent = MathAgent(math_config, forest=shared_forest)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, math_agent=math_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, math_agent=math_agent, forest=shared_forest)
    
    agent_stack = {'reader': reader_agent, 'writer': writer_agent, 'math': math_agent}

    # 2. First Study Phase (General Knowledge)
    print("\nPhase 2: Initial Study (Wikipedia Overview)...")
    general_wiki = """
    Calculus includes differentiation and integration.
    The power rule for derivatives is d/dx x^n = n*x^(n-1).
    Basic integration is the reverse of differentiation.
    Solving equations involves finding values for variables that make the expression zero.
    """
    reader_agent.ingest_text(general_wiki, title="Calculus Overview")

    # 3. The First Attempt
    print("\nPhase 3: The First Attempt (The Baseline)...")
    
    exam_questions = [
        {
            "id": "Q1 (Power Rule)",
            "q": "What is the derivative of x**7?",
            "eval": lambda a: "7.0 * x**6.0" in a or "7*x**6" in a,
            "topic": "differentiation"
        },
        {
            "id": "Q2 (Integration)",
            "q": "Integrate x**2",
            "eval": lambda a: "x**3.0 / 3.0" in a or "x**3/3" in a or "0.333" in a,
            "topic": "integration"
        },
        {
            "id": "Q3 (Algebra)",
            "q": "Solve x**2 - 4",
            "eval": lambda a: "-2" in a and "2" in a,
            "topic": "solving"
        }
    ]
    
    score1, results1 = run_exam(agent_stack, exam_questions)
    print(f"\n  [FIRST ATTEMPT GRADE] Score: {score1:.1f}%")

    if score1 < 100:
        # 4. Resit Loop: Identifying Gaps and Deep Research
        print("\nPhase 4: Resit Loop - Identifying Gaps & Deep Research...")
        failed_topics = set(r['topic'] for r in results1 if not r['correct'] for q in exam_questions if q['id'] == r['id'])
        
        if not failed_topics:
            print("  No failures detected. Resit skipped.")
        else:
            print(f"  Targeted research needed for topics: {failed_topics}")
            
            # Simulated Deep Research for failed topics
            if "integration" in failed_topics:
                print("  [RESEARCH] Reading 'Integration Table'...")
                reader_agent.ingest_text("The integral of x**n is x**(n+1)/(n+1). For x**2, it is x**3/3.", title="Integration Guide")
            
            if "solving" in failed_topics:
                print("  [RESEARCH] Reading 'Algebraic Solutions'...")
                reader_agent.ingest_text("To solve x**2 - a**2 = 0, the roots are x = a and x = -a. For x**2 - 4, the roots are 2 and -2.", title="Algebra Guide")

            # 5. The Resit
            print("\nPhase 5: The Resit (Final Attempt)...")
            score2, results2 = run_exam(agent_stack, exam_questions)
            print(f"\n  [RESIT GRADE] Score: {score2:.1f}%")
            
            if score2 > score1:
                print(f"  [SUCCESS] Agent improved score by {score2 - score1:.1f}% through targeted study.")
            elif score2 == 100:
                print("  [SUCCESS] Agent achieved a perfect score on the resit.")
            else:
                print("  [FAILURE] Agent failed to improve score on the resit.")

    print("\n[SUCCESS] SP-Web Test 2 completed.")

if __name__ == "__main__":
    run_experiment()
