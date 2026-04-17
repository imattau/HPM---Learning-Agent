"""
SP-Reader 8: Spelling Induction & Misspelling Detection.
Verifies character-level learning and detection of orthographic errors.
"""
import os
from pathlib import Path
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

KNOWLEDGE_BASE_DIR = "data/knowledge_base/reader_lifelong"

def run_experiment():
    print("================================================================================")
    print("SP-Reader 8: Spelling Induction & Misspelling Detection")
    print("================================================================================")

    # 1. Setup Reader Agent with character primitives
    print("Phase 1: Initialising Reader Agent with Character Primitives...")
    if Path(KNOWLEDGE_BASE_DIR).exists():
        print(f"  Loading existing knowledge from {KNOWLEDGE_BASE_DIR}...")
        agent = ReaderAgent.load_agent(KNOWLEDGE_BASE_DIR)
        # Ensure char primitives are enabled if they weren't before
        if not agent.config.include_char_primitives:
             print("  Re-initialising config with character primitives...")
             agent.config.include_char_primitives = True
             # Manually trigger expansion to include chars
             agent.config.__init__(agent.config.concepts, agent.config.idf, include_char_primitives=True)
             agent.reindex_knowledge_base()
    else:
        print("  Creating new lifelong knowledge base with character primitives...")
        config = TextDomainConfig.from_passages(["The initial knowledge base."], max_vocab=100, include_char_primitives=True)
        agent = ReaderAgent(config, cold_dir=KNOWLEDGE_BASE_DIR)

    # 2. Learn spellings of a set of words
    print("\nPhase 2: Learning correct word spellings as macros...")
    words = ["cat", "dog", "mouse", "elephant", "quick", "brown", "jumps"]
    for w in words:
        agent.learn_word_spelling(w, case_sensitive=False)
    print(f"  Learned {len(words)} word macros.")

    # 3. Test misspelling detection
    print("\nPhase 3: Detecting misspellings via Edit Distance...")
    test_cases = [
        ("cat", True),       # correct
        ("kat", False),      # common typo
        ("mous", False),     # missing e
        ("elefant", False),  # ph -> f
        ("quick", True),
        ("kwik", False),
    ]

    success_count = 0
    for word, expected_correct in test_cases:
        is_correct, closest, dist = agent.detect_misspelling(word.lower(), words)
        matches = (is_correct == expected_correct)
        if matches: success_count += 1
        status = "✓" if matches else "✗"
        print(f"  '{word}' -> correct={is_correct}, closest='{closest}', dist={dist} {status}")

    if success_count == len(test_cases):
        print("  SUCCESS: All misspelling test cases passed!")
    else:
        print(f"  PARTIAL: {success_count}/{len(test_cases)} cases passed.")

    # 4. Case sensitivity test
    print("\nPhase 4: Case Sensitivity...")
    agent.learn_word_spelling("Apple", case_sensitive=True)
    
    # Test case: lowercase 'apple' should be dist 1 from 'Apple' if case sensitive
    is_correct, closest, dist = agent.detect_misspelling("apple", ["Apple"])
    print(f"  'apple' vs 'Apple' (case-sensitive): correct={is_correct}, closest='{closest}', dist={dist}")
    
    if not is_correct and dist > 0:
        print("  SUCCESS: Case sensitivity handled correctly!")
    else:
        print("  FAILURE: Case mismatch not detected.")

    # 5. Integration with existing POS tagging (no regression)
    print("\nPhase 5: Regression Test (POS Tagging & SRL)...")
    train_sent = [["the", "cat", "chases"]]
    train_tags = [["DET", "NOUN", "VERB"]]
    agent.learn_pos_tagger(train_sent, train_tags)
    
    test_struct = agent.get_sentence_structure("The cat chases.")
    print(f"  POS tagging result: {test_struct}")
    
    # Check if 'cat' is still NOUN and 'chases' is VERB
    tags = {t: tag for t, tag in test_struct}
    if tags.get("cat") == "NOUN" and tags.get("chases") == "VERB":
        print("  SUCCESS: POS tagging still works correctly!")
    else:
        print("  FAILURE: Syntactic regression detected.")

    # 6. Persistence
    print("\nPhase 6: Saving expanded knowledge base...")
    agent.save_agent(KNOWLEDGE_BASE_DIR)
    print(f"  Knowledge base saved to {KNOWLEDGE_BASE_DIR}")

if __name__ == "__main__":
    run_experiment()
