"""
Toy implementation of Hierarchical Pattern Modelling (HPM) v3.
Includes 0-3 cells (objects, transitions, analogies, meta-analogies).
Introduces Dynamic Pattern Field (Section 2.5.4) and Prediction Testing (Section 9).
Based on: "Human Learning as Hierarchical Pattern Modelling" (Thomson)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from copy import deepcopy

# ------------------------------------------------------------
# 1. Hierarchical pattern: 0‑cell to 3-cell
# ------------------------------------------------------------
class Cell:
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

def make_1cell(source_0cell: Cell, target_0cell: Cell, embedding: np.ndarray, name: str = "") -> Cell:
    return Cell(dim=1, embedding=embedding, source=source_0cell, target=target_0cell, name=name)

def make_2cell(source_1cell: Cell, target_1cell: Cell, embedding: np.ndarray = None, name: str = "") -> Cell:
    if embedding is None: embedding = (source_1cell.emb + target_1cell.emb) / 2
    return Cell(dim=2, embedding=embedding, source=source_1cell, target=target_1cell, name=name)

def make_3cell(source_2cell: Cell, target_2cell: Cell, embedding: np.ndarray = None, name: str = "") -> Cell:
    if embedding is None: embedding = (source_2cell.emb + target_2cell.emb) / 2
    return Cell(dim=3, embedding=embedding, source=source_2cell, target=target_2cell, name=name)

# ------------------------------------------------------------
# 2. Loss and Compression
# ------------------------------------------------------------
def log_likelihood_softmax(pattern: Cell, observed: Cell, population: List[Cell], temperature: float = 1.0) -> float:
    """General p(lower | higher) via softmax over embeddings."""
    if not population: return 0.0
    scores = np.array([np.dot(pattern.emb, c.emb) for c in population])
    log_probs = scores / temperature - np.log(np.sum(np.exp(scores / temperature)) + 1e-9)
    try:
        idx = population.index(observed)
        return log_probs[idx]
    except (ValueError, IndexError):
        return -10.0 # High penalty for unseen

def hierarchical_loss(cell: Cell, observation_seq: List[Cell], all_cells: List[Cell], temperature=1.0) -> float:
    """L_hier for cells of any dimension (predicting cells of dim-1)."""
    if cell.dim == 0: return 0.0
    lower_dim = cell.dim - 1
    # For a cell of dim D, the 'population' of things it can predict are cells of dim D-1.
    # In the toy, 1-cells predict 0-cells (objects), 2-cells predict 1-cells, 3-cells predict 2-cells.
    population = [c for c in all_cells if c.dim == lower_dim]
    nll = 0.0
    for obs in observation_seq:
        if obs.dim == lower_dim:
            nll -= log_likelihood_softmax(cell, obs, population, temperature)
    return nll

def compression(cell: Cell) -> float:
    """MI proxy based on similarity to source/target (Appendix A.3)."""
    if cell.dim == 0 or not cell.src or not cell.tgt: return 0.0
    # Higher level explains lower level boundaries
    sim_src = np.dot(cell.emb, cell.src.emb)/(np.linalg.norm(cell.emb)*np.linalg.norm(cell.src.emb)+1e-6)
    sim_tgt = np.dot(cell.emb, cell.tgt.emb)/(np.linalg.norm(cell.emb)*np.linalg.norm(cell.tgt.emb)+1e-6)
    return float((sim_src + sim_tgt) / 2)

# ------------------------------------------------------------
# 3. Dynamic Pattern Field (Section 2.5.4)
# ------------------------------------------------------------
class DynamicPatternField:
    def __init__(self, patterns: List[Cell], initial_bias=0.1, decay=0.95):
        self.weights = {p.name: initial_bias for p in patterns}
        self.decay = decay

    def update(self, patterns: List[Cell], pattern_weights: np.ndarray, social_scores: np.ndarray):
        """Field evolves based on population frequency and social consensus."""
        for i, p in enumerate(patterns):
            # Field strength increases with social approval and current weight
            inc = 0.02 * (social_scores[i] + pattern_weights[i])
            self.weights[p.name] = self.weights.get(p.name, 0.1) * self.decay + inc

    def get_amplification(self, pattern_name: str) -> float:
        return self.weights.get(pattern_name, 0.0)

# ------------------------------------------------------------
# 4. Evaluators and Score
# ------------------------------------------------------------
def affective_evaluator(cell: Cell, curiosity_vector: np.ndarray) -> float:
    return -np.linalg.norm(cell.emb - curiosity_vector)

def social_evaluator(cell: Cell, consensus_vector: np.ndarray) -> float:
    return np.dot(cell.emb, consensus_vector)

def pattern_density(cell: Cell, all_cells: List[Cell], curiosity_vec: np.ndarray, 
                    consensus_vec: np.ndarray, field: DynamicPatternField) -> float:
    # Connectivity: same-dim connections
    connected = sum(1 for o in all_cells if o != cell and o.dim == cell.dim and cell.tgt == o.src)
    C = connected / max(1, len(all_cells)-1)
    
    # Saturation: Evaluator magnitude
    E = abs(affective_evaluator(cell, curiosity_vec)) + abs(social_evaluator(cell, consensus_vec))
    
    # Field: Dynamic amplification
    F = field.get_amplification(cell.name)
    
    return 0.3*C + 0.4*E + 0.3*F

def hierarchical_total_score(cell: Cell, observation_seq: List[Cell], 
                             all_cells: List[Cell], curiosity_vec: np.ndarray, 
                             consensus_vec: np.ndarray, field: DynamicPatternField,
                             beta_comp=0.5, beta_aff=0.5, gamma_soc=0.5, density_bias=0.2) -> float:
    L_hier = hierarchical_loss(cell, observation_seq, all_cells)
    comp = compression(cell)
    aff = affective_evaluator(cell, curiosity_vec)
    soc = social_evaluator(cell, consensus_vec)
    dens = pattern_density(cell, all_cells, curiosity_vec, consensus_vec, field)
    
    J = beta_aff * aff + gamma_soc * soc + density_bias * dens
    return -L_hier + beta_comp * comp + J

# ------------------------------------------------------------
# 5. Dynamics (Meta Pattern Rule + Learning Loop)
# ------------------------------------------------------------
class MetaPatternRule:
    def __init__(self, patterns: List[Cell], learning_rate=0.2, conflict_scale=0.05):
        self.patterns = patterns
        self.n = len(patterns)
        self.eta = learning_rate
        self.beta_c = conflict_scale
        self.weights = np.ones(self.n) / self.n
        self.field_strength = 0.2

    def update(self, scores: np.ndarray, field_freq: np.ndarray):
        augmented_scores = scores + self.field_strength * field_freq
        avg_score = np.dot(self.weights, augmented_scores)
        # Simple local competition (conflict)
        conflict = self.weights * 0.1 # Placeholder for structured kappa
        
        new_weights = (self.weights 
                       + self.eta * (augmented_scores - avg_score) * self.weights
                       - self.beta_c * conflict)
        new_weights = np.maximum(new_weights, 0.0)
    def best_pattern(self) -> Cell:
        return self.patterns[np.argmax(self.weights)]

def learn_step(mpr: MetaPatternRule, field: DynamicPatternField, observation_seq: List[Cell], 
               all_cells: List[Cell], curiosity_vec: np.ndarray, consensus_vec: np.ndarray, embed_lr=0.05):
    scores = np.array([hierarchical_total_score(p, observation_seq, all_cells, curiosity_vec, consensus_vec, field) 
                      for p in mpr.patterns])
    social_scores = np.array([social_evaluator(p, consensus_vec) for p in mpr.patterns])
    
    field_freq = np.array([field.get_amplification(p.name) for p in mpr.patterns])
    mpr.update(scores, field_freq)
    field.update(mpr.patterns, mpr.weights, social_scores)
    
    # Embedding update (Numeric gradient)
    for pat in mpr.patterns:
        grad = np.zeros_like(pat.emb)
        eps = 1e-3
        for d in range(len(pat.emb)):
            orig = pat.emb[d]
            pat.emb[d] = orig + eps
            s_plus = hierarchical_total_score(pat, observation_seq, all_cells, curiosity_vec, consensus_vec, field)
            pat.emb[d] = orig - eps
            s_minus = hierarchical_total_score(pat, observation_seq, all_cells, curiosity_vec, consensus_vec, field)
            grad[d] = (s_plus - s_minus) / (2*eps)
            pat.emb[d] = orig
        pat.emb += embed_lr * grad
        pat.emb /= (np.linalg.norm(pat.emb) + 1e-9)
    return scores

# ------------------------------------------------------------
# 6. Prediction Testing (Section 9)
# ------------------------------------------------------------
def run_curiosity_test(all_cells: List[Cell], curiosity_vec: np.ndarray, consensus_vec: np.ndarray):
    """Prediction 9.4: Score should peak at intermediate complexity."""
    print("\n--- Prediction 9.4: Curiosity & Complexity ---")
    # Low complexity: Repeated identical sequence
    low_seq = [all_cells[0]] * 5
    # High complexity: Random shuffle
    high_seq = list(np.random.choice(all_cells, 5))
    # Intermediate: Structured patterns
    mid_seq = all_cells[:5]
    
    field = DynamicPatternField(all_cells)
    scores = []
    for seq, label in [(low_seq, "Low"), (mid_seq, "Intermediate"), (high_seq, "High")]:
        s = hierarchical_total_score(all_cells[0], seq, all_cells, curiosity_vec, consensus_vec, field)
        scores.append((label, s))
        print(f"Complexity: {label:12} | Total Score: {s:.3f}")

# ------------------------------------------------------------
# 7. Main Demonstration
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    # 0-cells
    o = [Cell(0, np.random.randn(2), name=f"obj_{i}") for i in range(4)]
    
    # 1-cells (Transitions)
    p1 = [make_1cell(o[i], o[(i+1)%4], np.random.randn(2), name=f"t_{i}") for i in range(4)]
    
    # 2-cells (Analogies)
    p2 = [make_2cell(p1[0], p1[2], name="analogy_0_2"),
          make_2cell(p1[1], p1[3], name="analogy_1_3")]
    
    # 3-cells (Meta-Analogies)
    p3 = [make_3cell(p2[0], p2[1], name="meta_analogy")]
    
    all_patterns = p1 + p2 + p3
    mpr = MetaPatternRule(all_patterns)
    field = DynamicPatternField(all_patterns)
    
    curiosity_vec = np.array([0.5, 0.5])
    consensus_vec = np.array([1.0, 0.0])
    
    # Episode: sequence containing objects, transitions, and analogies
    episode = [o[1], p1[0], o[3], p1[2], p2[0]]
    
    print("=== Toy HPM v3 (0-3 Cells + Dynamic Field) ===")
    print(f"Initial population: {len(all_patterns)} patterns")
    
    for i in range(5):
        learn_step(mpr, field, episode, all_patterns + o, curiosity_vec, consensus_vec)
        best = mpr.best_pattern()
        print(f"Step {i}: Best = {best.name:15} (dim={best.dim}, weight={max(mpr.weights):.3f})")
    
    # Field state
    print("\nFinal Field Amplifications:")
    for name, val in field.weights.items():
        print(f"  {name:15}: {val:.3f}")
        
    run_curiosity_test(all_patterns + o, curiosity_vec, consensus_vec)
