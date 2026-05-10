#!/usr/bin/env python3
"""
Simple HPM Agent using all core features:
- 0,1,2,3 cells
- Epistemic, affective, social evaluators
- Meta Pattern Rule (replicator dynamics + forgetting)
- Pattern field, density, recombination, insight, replication institution
"""

import numpy as np
import string
import random
from collections import defaultdict

# ------------------------------------------------------------
# 1. Cell class (0,1,2,3)
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
# 2. Meta Pattern Rule (with forgetting)
# ------------------------------------------------------------
class MetaPatternRule:
    def __init__(self, patterns, lr=0.2, decay=0.99, conflict_scale=0.0):
        self.patterns = patterns
        self.lr = lr
        self.decay = decay
        self.beta_c = conflict_scale
        self.n = len(patterns)
        self.weights = np.ones(self.n) / (self.n + 1e-9)
        self.kappa = np.zeros((self.n, self.n))  # no conflict for simplicity

    def update(self, scores, field_freq=None):
        if field_freq is not None:
            scores = scores + 0.2 * field_freq
        avg = np.dot(self.weights, scores)
        conflict = np.array([np.sum(self.kappa[i] * self.weights) for i in range(self.n)])
        new_w = self.weights + self.lr * (scores - avg) * self.weights - self.beta_c * conflict * self.weights
        new_w *= self.decay
        new_w = np.maximum(new_w, 0.0)
        if new_w.sum() > 0:
            self.weights = new_w / new_w.sum()
        else:
            self.weights = np.ones(self.n) / (self.n + 1e-9)

    def best_from_source(self, src_cell, dim):
        best = None
        best_w = -np.inf
        for i, p in enumerate(self.patterns):
            if p.dim == dim and hasattr(p, 'src') and p.src == src_cell and self.weights[i] > best_w:
                best_w = self.weights[i]
                best = p
        return best

# ------------------------------------------------------------
# 3. Full HPM Agent
# ------------------------------------------------------------
class FullHPMAgent:
    def __init__(self, emb_dim=8, lr=0.3, decay=0.98, curiosity_target_entropy=None,
                 consensus_vec=None):
        self.emb_dim = emb_dim
        self.lr = lr
        self.decay = decay
        self.curiosity_target = curiosity_target_entropy or 1.5  # target entropy for inverted‑U
        self.consensus_vec = consensus_vec if consensus_vec is not None else np.zeros(emb_dim)

        # 0‑cells
        self.letter_to_cell = {}
        self.cells = []
        for letter in string.ascii_lowercase:
            emb = np.random.randn(emb_dim) * 0.1
            cell = Cell(0, emb, name=letter)
            self.letter_to_cell[letter] = cell
            self.cells.append(cell)

        self.patterns = []   # 1‑,2‑,3‑cells
        self.mpr = MetaPatternRule(self.patterns, lr, decay)
        self.field_weights = defaultdict(float)  # pattern field amplification

    # ---------- pattern creation ----------
    def _get_1cell(self, src_letter, tgt_letter):
        src = self.letter_to_cell[src_letter]
        tgt = self.letter_to_cell[tgt_letter]
        for p in self.patterns:
            if p.dim == 1 and p.src == src and p.tgt == tgt:
                return p
        emb = tgt.emb - src.emb
        new = Cell(1, emb, src, tgt, f"{src_letter}\u2192{tgt_letter}")
        self.patterns.append(new)
        self._rebuild_mpr()
        return new

    def _get_2cell(self, pat1, pat2):
        for p in self.patterns:
            if p.dim == 2 and p.src == pat1 and p.tgt == pat2:
                return p
        emb = pat2.emb - pat1.emb
        new = Cell(2, emb, pat1, pat2, f"({pat1.name})\u2192({pat2.name})")
        self.patterns.append(new)
        self._rebuild_mpr()
        return new

    def _get_3cell(self, pat2a, pat2b):
        for p in self.patterns:
            if p.dim == 3 and p.src == pat2a and p.tgt == pat2b:
                return p
        emb = pat2b.emb - pat2a.emb
        new = Cell(3, emb, pat2a, pat2b, f"({pat2a.name})\u2192({pat2b.name})")
        self.patterns.append(new)
        self._rebuild_mpr()
        return new

    def _rebuild_mpr(self):
        # Preserve weights of existing patterns
        old_weights = self.mpr.weights if hasattr(self.mpr, 'weights') else np.array([])
        self.mpr = MetaPatternRule(self.patterns, self.lr, self.decay)
        if len(old_weights) > 0:
            # New patterns get initial uniform weight, old ones preserved (re‑normalised)
            new_n = len(self.patterns)
            new_weights = np.ones(new_n) / new_n
            new_weights[:len(old_weights)] = old_weights
            self.mpr.weights = new_weights / (new_weights.sum() + 1e-9)

    # ---------- evaluators ----------
    def _log_likelihood_1cell(self, pat, next_cell):
        cell_embs = np.array([c.emb for c in self.cells])
        scores = np.dot(cell_embs, pat.emb)
        logp = scores - np.log(np.sum(np.exp(scores)) + 1e-9)
        idx = self.cells.index(next_cell)
        return logp[idx]

    def _log_likelihood_2cell(self, pat, next_1cell):
        one_cells = [p for p in self.patterns if p.dim == 1]
        if not one_cells:
            return 0.0
        one_cell_embs = np.array([p.emb for p in one_cells])
        scores = np.dot(one_cell_embs, pat.emb)
        logp = scores - np.log(np.sum(np.exp(scores)) + 1e-9)
        try:
            idx = one_cells.index(next_1cell)
        except ValueError:
            return -1e6
        return logp[idx]

    def _log_likelihood_3cell(self, pat, next_2cell):
        two_cells = [p for p in self.patterns if p.dim == 2]
        if not two_cells:
            return 0.0
        two_cell_embs = np.array([p.emb for p in two_cells])
        scores = np.dot(two_cell_embs, pat.emb)
        logp = scores - np.log(np.sum(np.exp(scores)) + 1e-9)
        try:
            idx = two_cells.index(next_2cell)
        except ValueError:
            return -1e6
        return logp[idx]

    def _affective_score(self, cell, data_ctx):
        # curiosity: negative absolute difference between predictive entropy and target
        if cell.dim == 1:
            # compute entropy of next‑letter distribution
            cell_embs = np.array([c.emb for c in self.cells])
            scores = np.dot(cell_embs, cell.emb)
            probs = np.exp(scores - np.log(np.sum(np.exp(scores)) + 1e-9))
            entropy = -np.sum(probs * np.log(probs + 1e-8))
        else:
            entropy = 0.0  # simplified
        return -abs(entropy - self.curiosity_target)

    def _social_score(self, cell):
        return np.dot(cell.emb, self.consensus_vec)

    def _density_score(self, cell):
        # simplified: fraction of other cells that are composable (1‑cells only)
        if cell.dim != 1:
            return 0.0
        connected = sum(1 for p in self.patterns if p.dim == 1 and p != cell and p.src == cell.tgt)
        return connected / max(1, len([p for p in self.patterns if p.dim == 1]) - 1)

    def total_score(self, cell, obs_seq, objects_cache=None):
        # For simplicity: use accuracy (log likelihood) + affective + social + density bias
        if cell.dim == 1 and len(obs_seq) >= 2:
            ll = self._log_likelihood_1cell(cell, obs_seq[-1])
        elif cell.dim == 2 and len(obs_seq) >= 2 and obs_seq[-1].dim == 1:
            ll = self._log_likelihood_2cell(cell, obs_seq[-1])
        elif cell.dim == 3 and len(obs_seq) >= 2 and obs_seq[-1].dim == 2:
            ll = self._log_likelihood_3cell(cell, obs_seq[-1])
        else:
            ll = 0.0
        aff = self._affective_score(cell, None)
        soc = self._social_score(cell)
        dens = self._density_score(cell)
        return ll + 0.3*aff + 0.2*soc + 0.1*dens + self.field_weights[cell.name]

    # ---------- learning from a word ----------
    def learn_word(self, word):
        """word: string like 'cat'"""
        # 1‑cell learning
        trans = []
        for i in range(len(word)-1):
            src, tgt = word[i], word[i+1]
            pat1 = self._get_1cell(src, tgt)
            trans.append(pat1)
            score = self.total_score(pat1, [self.letter_to_cell[src], self.letter_to_cell[tgt]])
            scores = np.array([score if p == pat1 else 0.0 for p in self.patterns])
            self.mpr.update(scores, None)

        # 2‑cell learning (between consecutive transitions)
        for i in range(len(trans)-1):
            pat2 = self._get_2cell(trans[i], trans[i+1])
            score = self.total_score(pat2, [trans[i], trans[i+1]])
            scores = np.array([score if p == pat2 else 0.0 for p in self.patterns])
            self.mpr.update(scores, None)

        # 3‑cell learning (if enough 2‑cells)
        two_cells = [p for p in self.patterns if p.dim == 2]
        for i in range(len(two_cells)-1):
            pat3 = self._get_3cell(two_cells[i], two_cells[i+1])
            score = self.total_score(pat3, [two_cells[i], two_cells[i+1]])
            scores = np.array([score if p == pat3 else 0.0 for p in self.patterns])
            self.mpr.update(scores, None)

    # ---------- recombination (creativity) ----------
    def recombine_1cells(self, pat1, pat2):
        """Create new 1‑cell by averaging embeddings of two 1‑cells."""
        new_emb = (pat1.emb + pat2.emb) / 2
        # source and target: heuristic – take src from first, tgt from second
        new = Cell(1, new_emb, pat1.src, pat2.tgt, f"recomb({pat1.name},{pat2.name})")
        self.patterns.append(new)
        self._rebuild_mpr()
        return new

    # ---------- replication institution (prune worst patterns) ----------
    def replicate(self, validation_words, prune_ratio=0.2):
        """Prune patterns with lowest average total score on validation words."""
        if not self.patterns: return
        scores = []
        for p in self.patterns:
            total = 0.0
            count = 0
            for w in validation_words:
                # quick score: for 1‑cells, use random sample; simplified
                if p.dim == 1:
                    for i in range(len(w)-1):
                        src, tgt = w[i], w[i+1]
                        if p.src == self.letter_to_cell[src] and p.tgt == self.letter_to_cell[tgt]:
                            total += 1.0
                            count += 1
            avg = total / count if count > 0 else -1e6
            scores.append(avg)
        threshold = np.percentile(scores, prune_ratio * 100)
        for i, p in enumerate(self.patterns):
            if scores[i] < threshold:
                self.mpr.weights[i] = 0.0
        # renormalise
        total_w = self.mpr.weights.sum()
        if total_w > 0:
            self.mpr.weights /= total_w
        else:
            self.mpr.weights = np.ones(len(self.patterns)) / (len(self.patterns) + 1e-9)

    # ---------- prediction ----------
    def predict_next_letter(self, prefix):
        """prefix is a string; predict next letter using best 1‑cell from last letter."""
        if not prefix:
            return None
        last = prefix[-1]
        src_cell = self.letter_to_cell[last]
        best = self.mpr.best_from_source(src_cell, dim=1)
        if best is None:
            return None
        return best.tgt.name

    def guess_missing(self, partial):
        pos = partial.find('_')
        if pos <= 0:
            return partial
        prefix = partial[:pos]
        pred = self.predict_next_letter(prefix)
        if pred:
            return partial.replace('_', pred, 1)
        return partial

# ------------------------------------------------------------
# 4. Demonstration
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== HPM Full Agent (all features) ===\n")
    agent = FullHPMAgent(emb_dim=8, lr=0.3, decay=0.98)

    # Training words
    words = ["cat", "dog", "bird", "fish", "ant", "bee"]
    print("Training on words:", words)
    for w in words:
        for _ in range(10): # Train multiple times
            agent.learn_word(w)

    # Show pattern count
    print(f"\nPatterns created: {len(agent.patterns)} (1‑,2‑,3‑cells)")

    # Recombination example
    pat1 = agent._get_1cell('c','a')
    pat2 = agent._get_1cell('d','o')
    new_pat = agent.recombine_1cells(pat1, pat2)
    print(f"\nRecombination created: {new_pat.name}")

    # Run replication institution
    agent.replicate(words, prune_ratio=0.1)
    print(f"After replication, patterns left: {len([p for i,p in enumerate(agent.patterns) if agent.mpr.weights[i] > 0])}")

    # Test missing letter prediction
    test_cases = ["ca_", "do_", "bi_d", "an_"]
    print("\nMissing letter guessing:")
    for case in test_cases:
        result = agent.guess_missing(case)
        print(f"  {case} -> {result}")

    # Show some 2‑cell weights
    print("\nExample 2‑cell weights (top 5):")
    two_cells = [(p, agent.mpr.weights[i]) for i, p in enumerate(agent.patterns) if p.dim == 2]
    two_cells.sort(key=lambda x: x[1], reverse=True)
    for p, w in two_cells[:5]:
        print(f"  {p.name}: weight = {w:.4f}")
