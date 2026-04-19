"""
SP-Web 3: NLP Quality Audit & Aesthetic Enhancement Experiment.
Verifies the linguistic "Aesthetics" and structural "Coherence" of generated NLP.
"""
import os
import shutil
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.utils.nlp_auditor import NLPAuditor
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Web 3: NLP Quality Audit - Linguistic Self-Correction")
    print("================================================================================")

    knowledge_base = "data/nlp_quality_test"

    # 1. Initialize Agents
    print("Phase 1: Initializing Agents...")
    text_config = TextDomainConfig.from_passages(["initial"], max_vocab=500, include_char_primitives=True)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=1000)
    
    reader_agent = ReaderAgent(text_config, forest=shared_forest)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, forest=shared_forest)
    auditor = NLPAuditor(reader_agent=reader_agent)

    # 2. Seed Knowledge
    print("\nPhase 2: Seeding Complex Knowledge...")
    passages = [
        "A transformer is a deep learning architecture that uses self-attention to weigh relationships between tokens in a sequence.",
        "Unlike recurrent neural networks, transformers process tokens in parallel.",
        "The attention mechanism allows each token representation to incorporate evidence from other tokens."
    ]
    for p in passages:
        reader_agent.ingest_text(p, title="Transformer Knowledge")
    
    print(f"  Forest Nodes: {len(shared_forest)}")

    # 3. Quality Audit via WriterAgent
    print("\nPhase 3: QA & Quality Auditing...")
    
    questions = [
        "What is a transformer?",
        "How do transformers process tokens?"
    ]
    
    for q in questions:
        print(f"\n  Q: {q}")
        
        # We'll manually show the auditing process
        answer_phrase = reader_agent.answer_question_hierarchical(q)
        if not answer_phrase:
            print("    [FAILURE] Could not find answer phrase.")
            continue
            
        candidates = [
            f"The answer is {answer_phrase}.",
            f"I found that {answer_phrase[0].lower() + answer_phrase[1:] if len(answer_phrase) > 1 else answer_phrase}",
            f"According to my research, {answer_phrase[0].lower() + answer_phrase[1:] if len(answer_phrase) > 1 else answer_phrase}",
            answer_phrase
        ]
        
        print("    Auditing Candidates:")
        best_candidate = None
        max_score = -1.0
        
        for c in candidates:
            report = auditor.score_text(c)
            score = report["total_utility"]
            print(f"      - Score: {score:.3f} | Text: \"{c}\"")
            print(f"        (Aes: {report['aesthetics']:.2f}, Div: {report['diversity']:.2f}, Coh: {report['coherence']:.2f})")
            if score > max_score:
                max_score = score
                best_candidate = c
        
        print(f"\n    [SELECTED] \"{best_candidate}\" (Utility: {max_score:.3f})")

    # 4. Stress Test: Repetitive/Broken text
    print("\nPhase 4: Stress Test (Repetitive vs Diverse text)...")
    repetitive = "A transformer is a transformer is a transformer is a transformer."
    diverse = "A transformer is a deep learning model that uses self-attention."
    
    rep_score = auditor.score_text(repetitive)["diversity"]
    div_score = auditor.score_text(diverse)["diversity"]
    
    print(f"  Repetitive Diversity: {rep_score:.3f}")
    print(f"  Diverse Diversity: {div_score:.3f}")
    
    if div_score > rep_score:
        print("  [SUCCESS] Diversity metric correctly identifies repetitive 'loops'.")
    else:
        print("  [FAILURE] Diversity metric failed to distinguish repetition.")

    print("\n[SUCCESS] SP-Web 3 NLP Quality experiment completed.")

if __name__ == "__main__":
    run_experiment()
