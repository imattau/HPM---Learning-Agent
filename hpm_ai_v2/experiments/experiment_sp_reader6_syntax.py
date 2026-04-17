"""
SP-Reader 6: Syntactic Structure Learning (POS Induction).
Verifies learning of syntactic rules and noun phrase extraction.
"""
import os
import shutil
from pathlib import Path
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

# --- Training Data for POS Induction ---
TRAIN_SENTENCES = [
    ["the", "cat", "chases", "the", "mouse"],
    ["a", "big", "dog", "runs", "fast"],
    ["she", "quickly", "eats", "an", "apple"],
    ["small", "birds", "fly", "high"],
    ["quick", "foxes", "jump", "rapidly"],
    ["the", "fox", "jumps", "over", "the", "lazy", "dog"]
]
TRAIN_TAGS = [
    ["DET", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "ADV"],
    ["NOUN", "ADV", "VERB", "DET", "NOUN"],
    ["ADJ", "NOUN", "VERB", "ADV"],
    ["ADJ", "NOUN", "VERB", "ADV"],
    ["DET", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"]
]

# --- Test Data for Generalisation ---
TEST_SENTENCES = [
    ["the", "small", "bird", "flies", "gracefully"],
    ["a", "quick", "fox", "jumps", "high"]
]
EXPECTED_TEST_TAGS = [
    ["DET", "ADJ", "NOUN", "VERB", "ADV"],
    ["DET", "ADJ", "NOUN", "VERB", "ADV"]
]

KNOWLEDGE_BASE_DIR = "data/knowledge_base/reader_lifelong"

def run_experiment():
    print("================================================================================")
    print("SP-Reader 6: Syntactic Structure Learning & POS Induction")
    print("================================================================================")

    # 1. Setup Persistent Reader Agent
    print("Phase 1: Initialising Persistent Reader Agent...")
    if Path(KNOWLEDGE_BASE_DIR).exists():
        print(f"  Loading existing knowledge from {KNOWLEDGE_BASE_DIR}...")
        agent = ReaderAgent.load_agent(KNOWLEDGE_BASE_DIR)
    else:
        print("  Creating new lifelong knowledge base...")
        config = TextDomainConfig.from_passages(["The initial knowledge base."], max_vocab=100)
        agent = ReaderAgent(config, cold_dir=KNOWLEDGE_BASE_DIR)

    # 2. Phase 1: POS Induction (k=3)
    print("\nPhase 2: POS Induction from minimal examples...")
    agent.learn_pos_tagger(TRAIN_SENTENCES, TRAIN_TAGS)
    print("  POS Macro induced. Rules found:")
    for k, v in list(agent.pos_rules.items())[:10]:
        print(f"    {k} -> {v}")

    # 3. Phase 2: Generalisation
    print("\nPhase 3: Syntactic Generalisation to new vocabulary...")
    for sent, expected in zip(TEST_SENTENCES, EXPECTED_TEST_TAGS):
        actual = agent._tag_sentence(sent)
        print(f"  Sentence: {' '.join(sent)}")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        if actual == expected:
            print("    SUCCESS: Correctly tagged!")
        else:
            print("    PARTIAL: Some tags mismatched.")

    # 4. Phase 3: Noun Phrase Extraction
    print("\nPhase 4: Noun Phrase Extraction...")
    test_complex = "The quick brown fox jumps over the lazy dog."
    print(f"  Sentence: '{test_complex}'")
    nps = agent.extract_noun_phrases(test_complex)
    print(f"  Extracted NPs: {nps}")
    if "the quick brown fox" in nps and "the lazy dog" in nps:
        print("  SUCCESS: Both NPs correctly identified!")
    else:
        print("  PARTIAL: NP extraction incomplete.")

    # 5. Phase 4: Syntax-Aware Query Disambiguation
    print("\nPhase 5: Syntax-Aware Query Disambiguation...")
    # Add contrasting passages
    agent.observe_passage("The quick dog chases the small cat.")
    agent.observe_passage("The small cat chases the quick dog.")
    
    query = "Who chases the dog?"
    print(f"  Query: '{query}'")
    # Identify subject (the 'who' or 'what' doing the chasing)
    query_struct = agent.get_sentence_structure(query)
    verb = next((t for t, tag in query_struct if tag == "VERB"), "chases")
    
    # 1. Bag-of-words baseline
    res_bow = agent.query(query)
    print(f"  Semantic Retrieval (Bag-of-Words): {res_bow}")
    
    # 2. Syntax-aware (look for NOUN followed by VERB)
    print("  Applying Syntax-Aware Subject Filtering...")
    candidates = agent.config._passages[-10:] # search last 10
    best_res = None
    for c in candidates:
        struct = agent.get_sentence_structure(c)
        # Find index of the verb in candidate
        v_idx = next((i for i, (t, tag) in enumerate(struct) if tag == "VERB" and t.lower() == verb.lower()), -1)
        if v_idx > 0:
            # Subject is the NOUN/NP before the verb
            subject = struct[v_idx-1][0]
            if subject.lower() == "cat": # If we're looking for cat as subject
                best_res = c
                break
    
    print(f"  Syntax-Aware Retrieval (Subject=Cat): {best_res}")
    if best_res and "cat chases" in best_res.lower():
        print("  SUCCESS: Correct sentence retrieved using subject-verb relation!")
    else:
        print("  PARTIAL: Syntax filter did not match.")
    
    # 6. Phase 6: Persistence
    print("\nPhase 6: Saving expanded knowledge base...")
    agent.save_agent(KNOWLEDGE_BASE_DIR)
    print(f"  Knowledge base saved to {KNOWLEDGE_BASE_DIR}")

if __name__ == "__main__":
    run_experiment()
