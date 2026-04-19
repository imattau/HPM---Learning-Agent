"""
SP-Sent Test 1: Few-Shot Sentiment Classification.
Verifies SentimentAgent's ability to handle negations, intensifiers, and persistent learning.
"""
import os
import shutil
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.sentiment_domain import SentimentDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.sentiment_agent import SentimentAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Sent Test 1: Few-Shot Sentiment Classification")
    print("================================================================================")

    knowledge_base = "data/sentiment_forest"
    # No shutil.rmtree here - Persistence Mandate

    # 1. Initialize Domains and Forest
    print("Phase 1: Initializing Orchestrated Sentiment Environment...")
    text_config = TextDomainConfig.from_passages(["sentiment analysis"], max_vocab=500, include_char_primitives=True)
    sent_config = SentimentDomainConfig(s_dim=20)
    
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    orchestrator = AgentOrchestrator(shared_forest)
    
    # Initialize agents
    reader_agent = ReaderAgent(text_config, forest=shared_forest)
    dict_agent = DictionaryAgent(text_config, reader_agent=reader_agent, forest=shared_forest)
    reader_agent.dictionary_agent = dict_agent
    
    sent_agent = SentimentAgent(sent_config, forest=shared_forest, dictionary_agent=dict_agent)
    writer_agent = WriterAgent(text_config, reader_agent=reader_agent, dictionary_agent=dict_agent, sentiment_agent=sent_agent, forest=shared_forest)

    # Register with orchestrator
    orchestrator.register(dict_agent)
    orchestrator.register(sent_agent)
    orchestrator.register(reader_agent)
    orchestrator.register(writer_agent)
    
    # 2. Seed Lexicon
    print("\nPhase 2: Seeding Sentiment Lexicon...")
    sent_agent.seed_lexicon()

    # 3. Sentiment Testing
    print("\nPhase 3: Testing Sentiment Classification...")
    
    test_cases = [
        {
            "id": "T1 (Positive)",
            "text": "I love this excellent product!",
            "q": "What is the sentiment of this review?",
            "eval": lambda a: "positive" in a.lower()
        },
        {
            "id": "T2 (Negative)",
            "text": "This was a terrible and awful experience.",
            "q": "What is the opinion of the customer?",
            "eval": lambda a: "negative" in a.lower()
        },
        {
            "id": "T3 (Negation)",
            "text": "The food was not good.",
            "q": "How did the person feel about the food?",
            "eval": lambda a: "negative" in a.lower()
        },
        {
            "id": "T4 (Intensifier)",
            "text": "I really hate this terrible movie.",
            "q": "What is the sentiment of the viewer?",
            "eval": lambda a: "negative" in a.lower() and "score" in a.lower() # Verify it reported score/intensity
        },
        {
            "id": "T5 (Neutral)",
            "text": "The box is sitting on the floor.",
            "q": "What is the sentiment?",
            "eval": lambda a: "neutral" in a.lower()
        }
    ]
    
    passed_count = 0
    for tc in test_cases:
        print(f"\n  [{tc['id']}] Text: \"{tc['text']}\"")
        # 1. Reader ingests text (creates sentence nodes)
        reader_agent.ingest_text(tc['text'], title=tc['id'])
        
        # 2. Writer answers (triggers sentiment analysis)
        answer = writer_agent.answer_natural(tc['q'])
        print(f"    Q: {tc['q']}")
        print(f"    A: {answer}")
        
        is_correct = tc['eval'](answer)
        if is_correct:
            print("    [RESULT] CORRECT")
            passed_count += 1
        else:
            print("    [RESULT] INCORRECT")

    # 4. Final Grade
    print(f"\nPhase 4: Final Grade...")
    score = (passed_count / len(test_cases)) * 100
    print(f"  Final Score: {score:.1f}% ({passed_count}/{len(test_cases)})")

    if score >= 80:
        print("  [STATUS] SENTIMENT AGENT PASSED EMOTIONAL INTELLIGENCE TEST")
    else:
        print("  [STATUS] SENTIMENT AGENT FAILED EMOTIONAL INTELLIGENCE TEST")

    print("\n[SUCCESS] SP-Sent Test 1 completed.")

if __name__ == "__main__":
    run_experiment()
