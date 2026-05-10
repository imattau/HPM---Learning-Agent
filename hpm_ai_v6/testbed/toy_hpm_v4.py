"""
Toy implementation of Hierarchical Pattern Modelling (HPM) v4.
Refines Affective Evaluator to use Predictive Entropy for Curiosity (Prediction 9.4).
Based on: "Human Learning as Hierarchical Pattern Modelling" (Thomson)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

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

    def predict_probs(self, population: List['Cell'], temperature: float = 1.0) -> np.ndarray:
        """Predict probability distribution over a lower-level population."""
        if not population: return np.array([])
        scores = np.array([np.dot(self.emb, c.emb) for c in population])
        exp_scores = np.exp(scores / temperature)
        return exp_scores / (np.sum(exp_scores) + 1e-9)

def make_1cell(source_0cell: Cell, target_0cell: Cell, embedding: np.ndarray, name: str = "") -> Cell:
    return Cell(dim=1, embedding=embedding, source=source_0cell, target=target_0cell, name=name)

def make_2cell(source_1cell: Cell, target_1cell: Cell, embedding: np.ndarray = None, name: str = "") -> Cell:
    if embedding is None: embedding = (source_1cell.emb + target_1cell.emb) / 2
    return Cell(dim=2, embedding=embedding, source=source_1cell, target=target_1cell, name=name)

# ------------------------------------------------------------
# 2. Loss and Compression
# ------------------------------------------------------------
def hierarchical_loss(cell: Cell, observation_seq: List[Cell], all_cells: List[Cell], temperature=1.0) -> float:
    if cell.dim == 0 or len(observation_seq) < 2: return 0.0
    population = [c for c in all_cells if c.dim == cell.dim - 1]
    nll = 0.0
    count = 0
    
    # 1-cell context-aware loss: P(obj_t+1 | obj_t, pattern)
    if cell.dim == 1:
        probs = cell.predict_probs(population, temperature)
        for i in range(len(observation_seq) - 1):
            curr, next_obj = observation_seq[i], observation_seq[i+1]
            if curr == cell.src: # Only score if source matches
                try:
                    idx = population.index(next_obj)
                    nll -= np.log(probs[idx] + 1e-9)
                    count += 1
                except (ValueError, IndexError):
                    nll += 5.0
                    count += 1
        return nll / count if count > 0 else 2.0 # Penalty for no matching context
    
    # Fallback for higher dims (simplified)
    probs = cell.predict_probs(population, temperature)
    for obs in observation_seq:
        if obs.dim == cell.dim - 1:
            try:
                idx = population.index(obs)
                nll -= np.log(probs[idx] + 1e-9)
                count += 1
            except (ValueError, IndexError):
                nll += 5.0
                count += 1
    return nll / count if count > 0 else 0.0

def compression(cell: Cell) -> float:
    if cell.dim == 0 or not cell.src or not cell.tgt: return 0.0
    sim_src = np.dot(cell.emb, cell.src.emb)/(np.linalg.norm(cell.emb)*np.linalg.norm(cell.src.emb)+1e-6)
    sim_tgt = np.dot(cell.emb, cell.tgt.emb)/(np.linalg.norm(cell.emb)*np.linalg.norm(cell.tgt.emb)+1e-6)
    return float((sim_src + sim_tgt) / 2)

# ------------------------------------------------------------
# 3. Dynamic Pattern Field
# ------------------------------------------------------------
class DynamicPatternField:
    def __init__(self, patterns: List[Cell], initial_bias=0.1, decay=0.95):
        self.weights = {p.name: initial_bias for p in patterns}
        self.decay = decay

    def update(self, patterns: List[Cell], pattern_weights: np.ndarray, social_scores: np.ndarray):
        for i, p in enumerate(patterns):
            inc = 0.02 * (social_scores[i] + pattern_weights[i])
            self.weights[p.name] = self.weights.get(p.name, 0.1) * self.decay + inc

    def get_amplification(self, pattern_name: str) -> float:
        return self.weights.get(pattern_name, 0.0)

# ------------------------------------------------------------
# 4. Evaluators (including Entropy-based Curiosity)
# ------------------------------------------------------------
def affective_evaluator(cell: Cell, all_cells: List[Cell], target_entropy_ratio: float = 0.5) -> float:
    """
    Curiosity peaks at intermediate entropy (Prediction 9.4).
    Target entropy is a ratio of max possible entropy for the population.
    """
    if cell.dim == 0: return 0.0
    population = [c for c in all_cells if c.dim == cell.dim - 1]
    if not population: return 0.0
    
    probs = cell.predict_probs(population)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(population))
    target_entropy = target_entropy_ratio * max_entropy
    
    # Curiosity is high when entropy is close to target (intermediate)
    return -abs(entropy - target_entropy)

def social_evaluator(cell: Cell, consensus_vector: np.ndarray) -> float:
    return np.dot(cell.emb, consensus_vector)

def pattern_density(cell: Cell, all_cells: List[Cell], consensus_vec: np.ndarray, field: DynamicPatternField) -> float:
    connected = sum(1 for o in all_cells if o != cell and o.dim == cell.dim and cell.tgt == o.src)
    C = connected / max(1, len(all_cells)-1)
    # Saturation (simplified for v4)
    E = abs(social_evaluator(cell, consensus_vec))
    F = field.get_amplification(cell.name)
    return 0.3*C + 0.4*E + 0.3*F

def hierarchical_total_score(cell: Cell, observation_seq: List[Cell], 
                             all_cells: List[Cell], consensus_vec: np.ndarray, 
                             field: DynamicPatternField,
                             beta_comp=0.5, beta_aff=0.5, gamma_soc=0.5, density_bias=0.2) -> float:
    L_hier = hierarchical_loss(cell, observation_seq, all_cells)
    comp = compression(cell)
    aff = affective_evaluator(cell, all_cells)
    soc = social_evaluator(cell, consensus_vec)
    dens = pattern_density(cell, all_cells, consensus_vec, field)
    
    J = beta_aff * aff + gamma_soc * soc + density_bias * dens
    return -L_hier + beta_comp * comp + J

# ------------------------------------------------------------
# 5. Dynamics (Meta Pattern Rule)
# ------------------------------------------------------------
class MetaPatternRule:
    def __init__(self, patterns: List[Cell], learning_rate=0.2, conflict_scale=0.05, decay=1.0):
        self.patterns = patterns
        self.n = len(patterns)
        self.eta = learning_rate
        self.beta_c = conflict_scale
        self.decay = decay
        self.weights = np.ones(self.n) / self.n
        self.field_strength = 0.2

    def update(self, scores: np.ndarray, field_freq: np.ndarray):
        augmented_scores = scores + self.field_strength * field_freq
        avg_score = np.dot(self.weights, augmented_scores)
        conflict = self.weights * 0.1
        
        # Meta Pattern Rule with forgetting (decay)
        new_weights = (self.decay * self.weights 
                       + self.eta * (augmented_scores - avg_score) * self.weights
                       - self.beta_c * conflict)
        new_weights = np.maximum(new_weights, 0.0)
        self.weights = new_weights / (new_weights.sum() + 1e-9)

    def best_pattern(self) -> Cell:
        return self.patterns[np.argmax(self.weights)]

def learn_step(mpr: MetaPatternRule, field: DynamicPatternField, observation_seq: List[Cell], 
               all_cells: List[Cell], consensus_vec: np.ndarray, embed_lr=0.05,
               beta_comp=0.5, beta_aff=0.5, gamma_soc=0.5, density_bias=0.2):
    scores = np.array([hierarchical_total_score(p, observation_seq, all_cells, consensus_vec, field,
                                                beta_comp=beta_comp, beta_aff=beta_aff, 
                                                gamma_soc=gamma_soc, density_bias=density_bias) 
                      for p in mpr.patterns])
    social_scores = np.array([social_evaluator(p, consensus_vec) for p in mpr.patterns])
    field_freq = np.array([field.get_amplification(p.name) for p in mpr.patterns])
    mpr.update(scores, field_freq)
    field.update(mpr.patterns, mpr.weights, social_scores)
    
    for pat in mpr.patterns:
        grad = np.zeros_like(pat.emb)
        eps = 1e-3
        for d in range(len(pat.emb)):
            orig = pat.emb[d]
            pat.emb[d] = orig + eps
            s_plus = hierarchical_total_score(pat, observation_seq, all_cells, consensus_vec, field,
                                             beta_comp=beta_comp, beta_aff=beta_aff, 
                                             gamma_soc=gamma_soc, density_bias=density_bias)
            pat.emb[d] = orig - eps
            s_minus = hierarchical_total_score(pat, observation_seq, all_cells, consensus_vec, field,
                                              beta_comp=beta_comp, beta_aff=beta_aff, 
                                              gamma_soc=gamma_soc, density_bias=density_bias)
            grad[d] = (s_plus - s_minus) / (2*eps)
            pat.emb[d] = orig
        pat.emb += embed_lr * grad
        pat.emb /= (np.linalg.norm(pat.emb) + 1e-9)
    return scores
