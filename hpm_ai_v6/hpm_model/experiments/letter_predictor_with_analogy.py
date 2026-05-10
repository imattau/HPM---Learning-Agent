#!/usr/bin/env python3
"""
HPM Letter Predictor with 2‑cell Analogies.
Learns 1‑cells (transitions) and 2‑cells (relations between consecutive transitions).
Predicts missing letters by analogy.
"""

import numpy as np
import string
from collections import defaultdict

# ------------------------------------------------------------
# 1. Cell class (0,1,2 cells)
# ------------------------------------------------------------
class Cell:
    def __init__(self, dim, embedding, source=None, target=None, name=""):
        self.dim = dim
        self.emb = np.array(embedding, dtype=float)
        self.src = source
        self.tgt = target
        self.name = name

    def __repr__(self):
        return f"{self.name}(d={self.dim})"

# ------------------------------------------------------------
# 2. Meta Pattern Rule (handles both 1‑ and 2‑cells)
# ------------------------------------------------------------
class MetaPatternRule:
    def __init__(self, patterns, learning_rate=0.2, decay=0.99):
        self.patterns = patterns
        self.eta = learning_rate
        self.decay = decay
        self.weights = np.ones(len(patterns)) / max(1, len(patterns))

    def update(self, scores):
        """scores: list of total scores for each pattern. Uses exponential fitness for better selection."""
        # Convert log-likelihoods/scores to positive fitness values
        fitness = np.exp(np.array(scores))
        avg = np.dot(self.weights, fitness)
        new_w = self.weights * (fitness / (avg + 1e-9))
        new_w *= self.decay
        new_w = np.maximum(new_w, 0.0)
        if new_w.sum() > 0:
            self.weights = new_w / new_w.sum()
        else:
            self.weights = np.ones(len(self.patterns)) / len(self.patterns)

    def best_pattern_from_source(self, src_cell, dim=1):
        """Highest‑weight pattern of given dimension with matching source."""
        best = None
        best_w = -np.inf
        for i, p in enumerate(self.patterns):
            if p.dim == dim and hasattr(p, 'src') and p.src == src_cell and self.weights[i] > best_w:
                best_w = self.weights[i]
                best = p
        return best

# ------------------------------------------------------------
# 3. HPM Agent with 1‑cells and 2‑cells
# ------------------------------------------------------------
class LetterHPMAgent:
    def __init__(self, embedding_dim=8, learning_rate=0.3, decay=0.98):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.decay = decay
        # 0‑cells for letters
        self.letter_to_cell = {}
        self.cells = []
        for letter in string.ascii_lowercase:
            emb = np.random.randn(embedding_dim) * 0.1
            cell = Cell(0, emb, name=letter)
            self.letter_to_cell[letter] = cell
            self.cells.append(cell)

        self.patterns = []          # 1‑cells and 2‑cells
        self.mpr = None

    # ----- 1‑cell management -----
    def _get_1cell(self, src_letter, tgt_letter, create=True):
        src_cell = self.letter_to_cell[src_letter]
        tgt_cell = self.letter_to_cell[tgt_letter]
        for p in self.patterns:
            if p.dim == 1 and p.src == src_cell and p.tgt == tgt_cell:
                return p
        if not create:
            return None
        emb = tgt_cell.emb - src_cell.emb
        new_p = Cell(1, emb, source=src_cell, target=tgt_cell,
                     name=f"{src_letter}→{tgt_letter}")
        self.patterns.append(new_p)
        self._rebuild_mpr()
        return new_p

    # ----- 2‑cell management (between two 1‑cells) -----
    def _get_2cell(self, prev_1cell, next_1cell, create=True):
        for p in self.patterns:
            if p.dim == 2 and p.src == prev_1cell and p.tgt == next_1cell:
                return p
        if not create:
            return None
        # Embedding: difference of next and prev 1‑cell embeddings
        emb = next_1cell.emb - prev_1cell.emb
        new_p = Cell(2, emb, source=prev_1cell, target=next_1cell,
                     name=f"({prev_1cell.name})→({next_1cell.name})")
        self.patterns.append(new_p)
        self._rebuild_mpr()
        return new_p

    def _rebuild_mpr(self):
        self.mpr = MetaPatternRule(self.patterns, self.learning_rate, self.decay)

    # ----- Likelihood for 1‑cell (accuracy) -----
    def _log_likelihood_1cell(self, pattern, next_cell):
        scores = np.array([np.dot(pattern.emb, c.emb) for c in self.cells])
        log_probs = scores - np.log(np.sum(np.exp(scores)) + 1e-9)
        idx = self.cells.index(next_cell)
        return log_probs[idx]

    # ----- Likelihood for 2‑cell (probability of next 1‑cell given a 2‑cell) -----
    def _log_likelihood_2cell(self, pattern, next_1cell):
        # pattern is 2‑cell; next_1cell is a candidate 1‑cell
        # Softmax over all 1‑cells in the population
        one_cells = [p for p in self.patterns if p.dim == 1]
        if not one_cells:
            return 0.0
        scores = np.array([np.dot(pattern.emb, c.emb) for c in one_cells])
        log_probs = scores - np.log(np.sum(np.exp(scores)) + 1e-9)
        try:
            idx = one_cells.index(next_1cell)
        except ValueError:
            return -1e6
        return log_probs[idx]

    # ----- Learning from a sequence (word) -----
    def learn_sequence(self, word):
        """word: string of letters, e.g., 'cat'"""
        # Learn 1‑cells from consecutive pairs
        transitions = []
        for i in range(len(word)-1):
            src, tgt = word[i], word[i+1]
            pat1 = self._get_1cell(src, tgt, create=True)
            transitions.append(pat1)
            # Update 1‑cell weight based on accuracy
            src_cell = self.letter_to_cell[src]
            tgt_cell = self.letter_to_cell[tgt]
            acc = self._log_likelihood_1cell(pat1, tgt_cell)
            scores = [acc if p == pat1 else 0.0 for p in self.patterns]
            self.mpr.update(scores)

        # Learn 2‑cells between consecutive transitions (if at least 2 transitions)
        for i in range(len(transitions)-1):
            prev = transitions[i]
            nxt = transitions[i+1]
            pat2 = self._get_2cell(prev, nxt, create=True)
            # For 2‑cell, reward when it correctly predicts the next transition
            acc2 = self._log_likelihood_2cell(pat2, nxt)
            scores = [acc2 if p == pat2 else 0.0 for p in self.patterns]
            self.mpr.update(scores)

    # ----- Prediction -----
    def predict_next_by_analogy(self, prefix):
        """
        Given a prefix string (e.g., 'ca'), predict the next letter.
        Uses 1‑cell from last letter, then 2‑cell analogy to infer next transition.
        """
        if len(prefix) < 1:
            return None
        last_letter = prefix[-1]
        src_cell = self.letter_to_cell[last_letter]
        # Step 1: find best 1‑cell from last letter
        best_1 = self.mpr.best_pattern_from_source(src_cell, dim=1)
        if best_1 is None:
            return None
        # Step 2: find best 2‑cell whose source is that 1‑cell
        best_2 = self.mpr.best_pattern_from_source(best_1, dim=2)
        if best_2 is None:
            # If no analogy, just use the target of the 1‑cell
            return best_1.tgt.name
        # The target of the 2‑cell is another 1‑cell
        next_1 = best_2.tgt
        # The letter we want is the target of that next 1‑cell
        return next_1.tgt.name

    def guess_missing_letter(self, partial):
        """Fill underscore in partial word (e.g., 'ca_') using analogy."""
        pos = partial.find('_')
        if pos == -1:
            return partial
        if pos == 0:
            return None
        prefix = partial[:pos]
        predicted_letter = self.predict_next_by_analogy(prefix)
        if predicted_letter is None:
            return partial
        return partial.replace('_', predicted_letter, 1)

# ------------------------------------------------------------
# 4. Demonstration
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== HPM Letter Predictor with 2‑cell Analogies ===\n")
    agent = LetterHPMAgent(embedding_dim=8, learning_rate=0.3, decay=0.98)

    # Training data
    alphabet = list(string.ascii_lowercase)
    words = ["cat", "dog", "bird", "fish", "ant", "bee"]
    print(f"Training on alphabet and words: {words}")
    agent.learn_sequence(''.join(alphabet))
    for w in words:
        for _ in range(15): # Increase frequency for words
            agent.learn_sequence(w)

    # Test basic predictions (direct 1‑cell)
    print("\n--- Direct predictions (1‑cell only) ---")
    test_pairs = [('c', 'a'), ('a', 't'), ('d', 'o'), ('o', 'g'), ('b', 'i')]
    for src, expected in test_pairs:
        # Using 1‑cell only (ignore analogies)
        src_cell = agent.letter_to_cell[src]
        best1 = agent.mpr.best_pattern_from_source(src_cell, dim=1)
        pred = best1.tgt.name if best1 else None
        print(f"  '{src}' → '{pred}' (expected '{expected}') -> {'✓' if pred == expected else '✗'}")

    # Test analogy‑based predictions for partial words
    print("\n--- Missing letter guessing (using analogies) ---")
    test_cases = ["ca_", "do_", "bi_d", "an_"]
    for case in test_cases:
        result = agent.guess_missing_letter(case)
        print(f"  '{case}' → '{result}'")
        # Also show internal steps
        pos = case.find('_')
        prefix = case[:pos]
        pred_letter = agent.predict_next_by_analogy(prefix)
        print(f"      (prefix '{prefix}' → predicted next letter '{pred_letter}')")

    # Show some learned 2‑cell weights
    print("\n--- Example 2‑cells (analogies) ---")
    for p in agent.patterns:
        if p.dim == 2 and (p.src.name.startswith('c→a') or p.src.name.startswith('d→o')):
            idx = agent.patterns.index(p)
            print(f"  {p.name}: weight = {agent.mpr.weights[idx]:.4f}")
