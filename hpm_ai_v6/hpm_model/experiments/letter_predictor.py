"""
Production-grade HPM Letter Predictor using the hpm_model core.
Learns letter sequences and predicts missing letters.
"""

import numpy as np
import string
import os
import sys
from typing import List, Optional

# Ensure the project root is in the path
sys.path.append(os.getcwd())

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.learning_agent import LearningAgent

class LetterPredictorExperiment:
    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        # Initialize 0-cells for letters a-z with small random embeddings
        self.letter_cells = {
            l: Cell(name=l, dim=0, embedding=np.random.randn(embedding_dim) * 0.1)
            for l in string.ascii_lowercase
        }
        self.objects = list(self.letter_cells.values())
        
        # We start with an empty set of 1-cells (patterns)
        self.patterns = []
        self.agent = None
        self._pattern_count = 0

    def _ensure_pattern(self, src_letter: str, tgt_letter: str):
        """Ensures a 1-cell exists for the transition."""
        name = f"{src_letter}->{tgt_letter}"
        for p in self.patterns:
            if p.name == name:
                return p
        
        src = self.letter_cells[src_letter]
        tgt = self.letter_cells[tgt_letter]
        # Transition embedding: tgt - src
        new_p = Cell(name=name, dim=1, embedding=tgt.embedding - src.embedding, 
                     source=src, target=tgt)
        self.patterns.append(new_p)
        return new_p

    def train(self, word: str, episodes: int = 5):
        """Trains the agent on a specific word multiple times."""
        if len(word) < 2:
            return
            
        # Ensure all transitions in the word have patterns
        for i in range(len(word) - 1):
            self._ensure_pattern(word[i], word[i+1])
            
        # Re-initialize agent if patterns changed
        if self.agent is None or len(self.patterns) > self._pattern_count:
            # We must use a copy of the patterns list to ensure LearningAgent 
            # and MetaPatternRule stay in sync with the snapshot
            self.agent = LearningAgent(patterns=list(self.patterns))
            self._pattern_count = len(self.patterns)
            
        # Convert word to sequence of cells
        seq = [self.letter_cells[l] for l in word]
        
        # Train
        for _ in range(episodes):
            self.agent.perceive(seq, self.objects, context={"consensus_vec": np.zeros(self.embedding_dim)})

    def predict_next(self, letter: str) -> Optional[str]:
        """Predicts the next letter after the given one."""
        if self.agent is None:
            return None
            
        src_cell = self.letter_cells.get(letter)
        if not src_cell:
            return None
            
        # Find highest weight pattern starting from this letter
        best_p = None
        max_w = -1.0
        
        weights = self.agent.get_weights()
        for i, p in enumerate(self.agent.patterns):
            if p.source and p.source.name == letter:
                if weights[i] > max_w:
                    max_w = weights[i]
                    best_p = p
                    
        return best_p.target.name if best_p else None

    def guess_missing(self, partial: str) -> str:
        """Fills in '_' in a word like 'ca_'."""
        idx = partial.find('_')
        if idx <= 0:
            return partial
            
        prev = partial[idx-1]
        next_l = self.predict_next(prev)
        if next_l:
            return partial.replace('_', next_l, 1)
        return partial

def run_experiment():
    print("=== HPM Production Letter Predictor ===")
    predictor = LetterPredictorExperiment()
    
    # Training
    print("Training on 'alphabet' fragments and words...")
    predictor.train("abcde", episodes=10)
    predictor.train("cat", episodes=10)
    predictor.train("dog", episodes=10)
    predictor.train("bird", episodes=10)
    predictor.train("ant", episodes=10)
    
    # Testing
    test_cases = [("c", "a"), ("a", "t"), ("d", "o"), ("o", "g"), ("b", "i")]
    print("\n--- Predictions ---")
    for src, expected in test_cases:
        pred = predictor.predict_next(src)
        print(f"  '{src}' -> '{pred}' (Expected: '{expected}') -> {'✓' if pred == expected else '✗'}")
        
    print("\n--- Missing Letter Guessing ---")
    guess_cases = ["ca_", "do_", "bi_d", "an_"]
    for case in guess_cases:
        result = predictor.guess_missing(case)
        print(f"  '{case}' -> '{result}'")

if __name__ == "__main__":
    run_experiment()
