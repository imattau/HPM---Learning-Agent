"""
Toy implementation of Hierarchical Pattern Modelling (HPM) v2.
Includes 2-cells (analogies/transformations) as per Section 3.3 and 2.3.
Based on: "Human Learning as Hierarchical Pattern Modelling" (Thomson)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from copy import deepcopy

# ------------------------------------------------------------
# 1. Hierarchical pattern: 0‑cell, 1‑cell, 2‑cell with embeddings
# ------------------------------------------------------------
class Cell:
    """Polygraph cell with dimension (0,1,2,...) and vector embedding."""
    def __init__(self, dim: int, embedding: np.ndarray,
                 source: Optional['Cell'] = None,
                 target: Optional['Cell'] = None,
                 name: str = ""):
        self.dim = dim
        self.emb = embedding.copy()
        self.src = source
        self.tgt = target
        self.name = name if name else f"cell_{id(self)}"

    def __repr__(self):
        return f"{self.name}(d={self.dim}, emb={self.emb[:2].round(2)}...)"

def compose(c1: Cell, c2: Cell, compose_fn: Callable = lambda u,v: u+v) -> Cell:
    """Compose two 1‑cells (or higher when boundaries match). Section 3.4."""
    if c1.dim != 1 or c2.dim != 1:
        raise NotImplementedError("Only 1‑cell composition for toy")
    if c1.tgt != c2.src:
        raise ValueError(f"Type mismatch: {c1.tgt} != {c2.src}")
    new_emb = compose_fn(c1.emb, c2.emb)
    return Cell(1, new_emb, source=c1.src, target=c2.tgt,
                name=f"({c1.name}∘{c2.name})")

def make_2cell(source_1cell: Cell, target_1cell: Cell,
               embedding: np.ndarray = None,
               name: str = "") -> Cell:
    """Create a 2‑cell expressing a transformation/analogy between two 1‑cells."""
    if source_1cell.dim != 1 or target_1cell.dim != 1:
        raise ValueError("2‑cell must connect two 1‑cells")
    if embedding is None:
        embedding = (source_1cell.emb + target_1cell.emb) / 2   # heuristic
    return Cell(dim=2, embedding=embedding,
                source=source_1cell, target=target_1cell, name=name)

# ------------------------------------------------------------
# 2. Generative model and hierarchical loss (Appendix A.2, A.3)
# ------------------------------------------------------------
def log_likelihood_1cell(cell: Cell, next_object: Cell, objects: List[Cell],
                         temperature: float = 1.0) -> float:
    """p(data | pattern) for 1‑cell pattern. Data = observed next object."""
    scores = np.array([np.dot(cell.emb, obj.emb) for obj in objects])
    log_probs = scores / temperature - np.log(np.sum(np.exp(scores / temperature)))
    idx = objects.index(next_object)
    return log_probs[idx]

def log_likelihood_twocell(twocell: Cell, observed_1cell: Cell,
                           all_one_cells: List[Cell], temperature=1.0):
    """p(1-cell | 2-cell) using dot product similarity."""
    scores = np.array([np.dot(twocell.emb, c.emb) for c in all_one_cells])
    log_probs = scores/temperature - np.log(np.sum(np.exp(scores/temperature)))
    idx = all_one_cells.index(observed_1cell)
    return log_probs[idx]

def hierarchical_loss(cell: Cell,
                      observation_seq: List[Cell],
                      objects: List[Cell],          # for 1‑cell patterns
                      one_cells: List[Cell] = None, # for 2‑cell patterns
                      temperature=1.0) -> float:
    """Hierarchical loss L_hier = negative log likelihood."""
    if cell.dim == 1:
        nll = 0.0
        for obj in observation_seq:
            if obj in objects:
                nll -= log_likelihood_1cell(cell, obj, objects, temperature)
        return nll
    elif cell.dim == 2:
        if one_cells is None:
            return 0.0 # or raise ValueError
        nll = 0.0
        for obs in observation_seq:
            if obs.dim == 1 and obs in one_cells:
                nll -= log_likelihood_twocell(cell, obs, one_cells, temperature)
        return nll
    else:
        return 0.0

def compression(cell: Cell, lower_cells: List[Cell]) -> float:
    """
    Mutual information proxy.
    Sects A.3, D7.
    """
    if cell.dim == 1:
        if not lower_cells: return 0.0
        lower_embs = np.array([c.emb for c in lower_cells])
        high_emb = cell.emb
        # Fallback to simple mean correlation
        corrs = []
        for l_emb in lower_embs:
            if np.std(l_emb) == 0 or np.std(high_emb) == 0:
                corrs.append(0.0)
            else:
                corrs.append(np.abs(np.corrcoef(l_emb, high_emb)[0,1]))
        return float(np.mean(corrs)) if corrs else 0.0
    elif cell.dim == 2:
        related_1cells = []
        if cell.src and cell.src.dim == 1: related_1cells.append(cell.src)
        if cell.tgt and cell.tgt.dim == 1: related_1cells.append(cell.tgt)
        if not related_1cells: return 0.0
        sims = [np.dot(cell.emb, c.emb)/(np.linalg.norm(cell.emb)*np.linalg.norm(c.emb)+1e-6)
                for c in related_1cells]
        return float(np.mean(sims))
    else:
        return 0.0

# ------------------------------------------------------------
# 3. Evaluators and pattern density
# ------------------------------------------------------------
def affective_evaluator(cell: Cell, curiosity_vector: np.ndarray) -> float:
    return -np.linalg.norm(cell.emb - curiosity_vector)

def social_evaluator(cell: Cell, consensus_vector: np.ndarray) -> float:
    return np.dot(cell.emb, consensus_vector)

def structural_connectivity(cell: Cell, all_cells: List[Cell]) -> float:
    connected = 0
    for other in all_cells:
        if other is cell: continue
        if cell.dim == 1 and other.dim == 1 and cell.tgt == other.src:
            connected += 1
        elif cell.dim == 2 and other.dim == 2 and cell.tgt == other.src:
            connected += 1
    return connected / max(1, len(all_cells)-1)

def pattern_density(cell: Cell, all_cells: List[Cell],
                    data_ctx: Dict, field_weights: Dict,
                    alpha=0.3, beta=0.4, gamma=0.3) -> float:
    C = structural_connectivity(cell, all_cells)
    aff = abs(affective_evaluator(cell, data_ctx.get('curiosity', np.zeros_like(cell.emb))))
    soc = abs(social_evaluator(cell, data_ctx.get('consensus', np.zeros_like(cell.emb))))
    E = aff + soc
    F = field_weights.get(cell.name, 0.0)
    return alpha*C + beta*E + gamma*F

# ------------------------------------------------------------
# 4. Hierarchical total score
# ------------------------------------------------------------
def hierarchical_total_score(cell: Cell,
                             observation_seq: List[Cell],
                             lower_cells: List[Cell],
                             objects: List[Cell],
                             curiosity_vec: np.ndarray,
                             consensus_vec: np.ndarray,
                             field_weights: Dict,
                             all_cells: List[Cell],
                             beta_comp: float = 0.5,
                             beta_aff: float = 0.5,
                             gamma_soc: float = 0.5,
                             density_bias: float = 0.2) -> float:
    one_cells = [c for c in all_cells if c.dim == 1] if cell.dim == 2 else None
    L_hier = hierarchical_loss(cell, observation_seq, objects, one_cells)
    comp = compression(cell, lower_cells)
    aff = affective_evaluator(cell, curiosity_vec)
    soc = social_evaluator(cell, consensus_vec)
    dens = pattern_density(cell, all_cells,
                           {'curiosity': curiosity_vec, 'consensus': consensus_vec},
                           field_weights)
    J = beta_aff * aff + gamma_soc * soc + density_bias * dens
    return -L_hier + beta_comp * comp + J

# ------------------------------------------------------------
# 5. Recombination
# ------------------------------------------------------------
def recombine(parent1: Cell, parent2: Cell,
              compose_fn: Callable = lambda u,v: (u+v)/2) -> Cell:
    new_dim = max(parent1.dim, parent2.dim)
    new_emb = compose_fn(parent1.emb, parent2.emb)
    if new_dim == 2:
        new_src = parent1.src if parent1.dim == 2 else parent1
        new_tgt = parent2.tgt if parent2.dim == 2 else parent2
        return Cell(2, new_emb, source=new_src, target=new_tgt,
                    name=f"recomb({parent1.name},{parent2.name})")
    else:
        return Cell(1, new_emb, source=parent1.src, target=parent2.tgt,
                    name=f"recomb({parent1.name},{parent2.name})")

def insight_evaluator(new_cell: Cell, parent_cells: List[Cell],
                      test_observations: List[Cell], objects: List[Cell],
                      all_cells: List[Cell]) -> float:
    if parent_cells:
        sims = [np.dot(new_cell.emb, p.emb)/(np.linalg.norm(new_cell.emb)*np.linalg.norm(p.emb)+1e-6)
                for p in parent_cells]
        novelty = 1 - np.mean(sims)
    else:
        novelty = 1.0
    
    one_cells = [c for c in all_cells if c.dim == 1] if new_cell.dim == 2 else None
    loss_val = hierarchical_loss(new_cell, test_observations, objects, one_cells)
    effectiveness = -loss_val
    return 0.5 * novelty + 0.5 * effectiveness

# ------------------------------------------------------------
# 6. Meta Pattern Rule dynamics
# ------------------------------------------------------------
class MetaPatternRule:
    def __init__(self, patterns: List[Cell],
                 learning_rate: float = 0.1,
                 conflict_scale: float = 0.05,
                 incompatibility_matrix: Optional[np.ndarray] = None):
        self.patterns = patterns
        self.n = len(patterns)
        self.eta = learning_rate
        self.beta_c = conflict_scale
        self.weights = np.ones(self.n) / self.n
        if incompatibility_matrix is None:
            self.kappa = np.zeros((self.n, self.n))
        else:
            self.kappa = incompatibility_matrix
        self.field_strength = 0.2

    def update(self, scores: np.ndarray, pattern_field_freq: np.ndarray = None):
        if pattern_field_freq is not None:
            scores = scores + self.field_strength * pattern_field_freq
        avg_score = np.dot(self.weights, scores)
        conflict = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j: conflict[i] += self.kappa[i, j] * self.weights[j]
        new_weights = (self.weights
                       + self.eta * (scores - avg_score) * self.weights
                       - self.beta_c * conflict * self.weights)
        new_weights = np.maximum(new_weights, 0.0)
        if new_weights.sum() > 0: self.weights = new_weights / new_weights.sum()
        else: self.weights = np.ones(self.n) / self.n

    def best_pattern(self) -> Cell:
        return self.patterns[np.argmax(self.weights)]

# ------------------------------------------------------------
# 7. Learning loop
# ------------------------------------------------------------
def learn_step(mpr: MetaPatternRule,
               observation_seq: List[Cell],
               lower_cells: List[Cell],
               objects: List[Cell],
               curiosity_vec: np.ndarray,
               consensus_vec: np.ndarray,
               field_weights: Dict,
               all_cells: List[Cell],
               embed_lr: float = 0.01):
    scores = np.zeros(mpr.n)
    for i, pat in enumerate(mpr.patterns):
        scores[i] = hierarchical_total_score(
            pat, observation_seq, lower_cells, objects,
            curiosity_vec, consensus_vec, field_weights, all_cells
        )
    freq = mpr.weights / mpr.weights.sum()
    mpr.update(scores, pattern_field_freq=freq)

    for i, pat in enumerate(mpr.patterns):
        grad = np.zeros_like(pat.emb)
        eps = 1e-3
        for d in range(len(pat.emb)):
            original = pat.emb[d]
            pat.emb[d] = original + eps
            s_plus = hierarchical_total_score(pat, observation_seq, lower_cells, objects, curiosity_vec, consensus_vec, field_weights, all_cells)
            pat.emb[d] = original - eps
            s_minus = hierarchical_total_score(pat, observation_seq, lower_cells, objects, curiosity_vec, consensus_vec, field_weights, all_cells)
            grad[d] = (s_plus - s_minus) / (2*eps)
            pat.emb[d] = original
        pat.emb += embed_lr * grad
        pat.emb = pat.emb / (np.linalg.norm(pat.emb) + 1e-6)
    return scores

# ------------------------------------------------------------
# 8. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    apple = Cell(0, np.array([1.0, 0.2]), name="apple")
    banana = Cell(0, np.array([0.8, 0.9]), name="banana")
    cherry = Cell(0, np.array([0.1, 1.2]), name="cherry")
    objects = [apple, banana, cherry]

    patterns = [
        Cell(1, np.random.randn(2)*0.5, source=apple, target=banana, name="A_to_B"),
        Cell(1, np.random.randn(2)*0.5, source=banana, target=cherry, name="B_to_C"),
        Cell(1, np.random.randn(2)*0.5, source=cherry, target=apple, name="C_to_A"),
    ]

    # Create 2-cell: analogy between A->B and B->C
    analogy = make_2cell(patterns[0], patterns[1], name="AtoB_is_like_BtoC")
    patterns.append(analogy)

    curiosity_vec = np.array([0.5, 0.5])
    consensus_vec = np.array([1.0, 0.0])
    field_weights = {p.name: 0.1 for p in patterns}
    mpr = MetaPatternRule(patterns, learning_rate=0.2)

    # Episodes: sequences of 0-cells and 1-cells
    # 2-cells learn from sequences of 1-cells
    episodes = [
        [banana, patterns[0], cherry, patterns[1]],
        [banana, patterns[0], patterns[1]],
        [cherry, patterns[2], banana],
    ]

    print("=== Toy HPM v2 (with 2-cells) ===")
    for idx, ep in enumerate(episodes):
        learn_step(mpr, ep, objects, objects, curiosity_vec, consensus_vec, field_weights, patterns)
        print(f"Episode {idx}: Best Pattern = {mpr.best_pattern().name} (Weight: {max(mpr.weights):.3f})")

    # Demonstrate 2-cell recombination
    new_analogy = recombine(patterns[3], patterns[0])
    print(f"\nRecombined 2-cell: {new_analogy.name}")
    print(f"Insight: {insight_evaluator(new_analogy, [patterns[3], patterns[0]], [patterns[1]], objects, patterns):.3f}")
