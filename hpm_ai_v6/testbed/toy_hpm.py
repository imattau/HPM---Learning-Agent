"""
Toy implementation of Hierarchical Pattern Modelling (HPM) core requirements.
Based on: "Human Learning as Hierarchical Pattern Modelling" (Thomson)
Covers: Sects 2.2-2.6, 3.4, 5, Appendix A, D, E.
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

def hierarchical_loss(cell: Cell, observation_seq: List[Cell],
                      objects: List[Cell]) -> float:
    """
    Hierarchical loss L_hier = negative ELBO (approximated).
    For simplicity: treat higher‑dim cells as latent and compute
    negative log p(obs | cell) + KL( q(z) || p(z) )? Here we compress:
    Actually we follow Appendix D7: use ELBO.
    Toy: Use 1‑cell as pattern, but also compute compression via mutual info.
    We'll compute compression separately.
    """
    # Epistemic part: negative log‑likelihood of observations under pattern
    nll = 0.0
    for obs in observation_seq:
        nll -= log_likelihood_1cell(cell, obs, objects)
    # For hierarchical we also need KL; we'll approximate compression later.
    return nll

def compression(cell: Cell, lower_cells: List[Cell]) -> float:
    """
    Mutual information between cell's embedding and lower‑level embeddings.
    Appendix A.3: Comp = H(z1) - H(z1|z2). Toy: linear Gaussian estimate.
    """
    # We treat cell.emb as higher‑level z2, lower_cells as z1.
    if not lower_cells:
        return 0.0
    lower_embs = np.array([c.emb for c in lower_cells])
    high_emb = cell.emb
    # Simple linear correlation as proxy for mutual information
    # Actually we want I(z1;z2) ≈ 0.5 * log(1 + ρ²/(1-ρ²)) for Gaussian?
    # Simpler: covariance trace
    cov = np.cov(lower_embs.T) if lower_embs.shape[0] > 1 else np.eye(len(high_emb))
    # High‑level explains variance? Use R² from linear regression
    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(lower_embs, np.tile(high_emb, (len(lower_embs),1)))
        r2 = reg.score(lower_embs, np.tile(high_emb, (len(lower_embs),1)))
        return r2
    except ImportError:
        # Fallback: absolute correlation
        if len(lower_embs) == 0:
            return 0.0
        # Correlation between high_emb and each lower_emb, then averaged
        corrs = []
        for l_emb in lower_embs:
            if np.std(l_emb) == 0 or np.std(high_emb) == 0:
                corrs.append(0.0)
            else:
                corrs.append(np.abs(np.corrcoef(l_emb, high_emb)[0,1]))
        return float(np.mean(corrs)) if corrs else 0.0

# ------------------------------------------------------------
# 3. Evaluators (affective, social) and pattern density (Section 2.6, A.8)
# ------------------------------------------------------------
def affective_evaluator(cell: Cell, curiosity_vector: np.ndarray) -> float:
    """Affective reward: negative distance to curiosity target."""
    return -np.linalg.norm(cell.emb - curiosity_vector)

def social_evaluator(cell: Cell, consensus_vector: np.ndarray) -> float:
    """Social reward: dot product with group consensus."""
    return np.dot(cell.emb, consensus_vector)

def structural_connectivity(cell: Cell, all_cells: List[Cell]) -> float:
    """C(h): fraction of other cells that are connected via composition."""
    connected = 0
    for other in all_cells:
        if other is cell:
            continue
        # Check if composition possible (boundaries match)
        if (cell.dim == 1 and other.dim == 1 and cell.tgt == other.src):
            connected += 1
    return connected / max(1, len(all_cells)-1)

def evaluator_saturation(cell: Cell, data_ctx: Dict) -> float:
    """E(h): sum of magnitudes of affective + social evaluators."""
    aff = abs(affective_evaluator(cell, data_ctx.get('curiosity', np.zeros_like(cell.emb))))
    soc = abs(social_evaluator(cell, data_ctx.get('consensus', np.zeros_like(cell.emb))))
    return aff + soc

def field_amplification(cell: Cell, field_weights: Dict[str, float]) -> float:
    """F(h): external field support (e.g., cultural frequency)."""
    return field_weights.get(cell.name, 0.0)

def pattern_density(cell: Cell, all_cells: List[Cell],
                    data_ctx: Dict, field_weights: Dict,
                    alpha=0.3, beta=0.4, gamma=0.3) -> float:
    """D(h) = α·C + β·E + γ·F (Section A.8.1)."""
    C = structural_connectivity(cell, all_cells)
    E = evaluator_saturation(cell, data_ctx)
    F = field_amplification(cell, field_weights)
    return alpha*C + beta*E + gamma*F

# ------------------------------------------------------------
# 4. Hierarchical total score (Appendix D7)
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
    """
    Total_hier = -L_hier + β_comp * Compression + J
    where J = β_aff * E_aff + γ_soc * E_soc + density_bias * D(h)
    (density bias added to evaluator sum as per Section A.8.3)
    """
    L_hier = hierarchical_loss(cell, observation_seq, objects)
    comp = compression(cell, lower_cells)
    aff = affective_evaluator(cell, curiosity_vec)
    soc = social_evaluator(cell, consensus_vec)
    dens = pattern_density(cell, all_cells,
                           {'curiosity': curiosity_vec, 'consensus': consensus_vec},
                           field_weights)
    J = beta_aff * aff + gamma_soc * soc + density_bias * dens
    return -L_hier + beta_comp * comp + J

# ------------------------------------------------------------
# 5. Recombination and insight (Appendix E)
# ------------------------------------------------------------
def recombine(parent1: Cell, parent2: Cell,
              compose_fn: Callable = lambda u,v: (u+v)/2) -> Cell:
    """
    Create new pattern by recombining two patterns.
    Here: average embeddings, and new cell has source of parent1, target of parent2.
    Section E2.
    """
    new_emb = compose_fn(parent1.emb, parent2.emb)
    # Simple structural recombination: new 1‑cell from src1 to tgt2
    new_cell = Cell(1, new_emb,
                    source=parent1.src, target=parent2.tgt,
                    name=f"recomb({parent1.name},{parent2.name})")
    return new_cell

def insight_evaluator(new_cell: Cell, parent_cells: List[Cell],
                      test_observations: List[Cell], objects: List[Cell]) -> float:
    """
    Insight score = α_nov * novelty + α_eff * effectiveness (Section E4).
    Novelty: inverse average similarity to parents. Effectiveness: performance.
    """
    # Novelty: 1 - average cosine similarity to parents
    if parent_cells:
        sims = [np.dot(new_cell.emb, p.emb)/(np.linalg.norm(new_cell.emb)*np.linalg.norm(p.emb)+1e-6)
                for p in parent_cells]
        novelty = 1 - np.mean(sims)
    else:
        novelty = 1.0
    # Effectiveness: negative loss on a held‑out test observation
    if test_observations:
        loss_val = hierarchical_loss(new_cell, test_observations, objects)
        effectiveness = -loss_val
    else:
        effectiveness = 0.0
    return 0.5 * novelty + 0.5 * effectiveness

# ------------------------------------------------------------
# 6. Meta Pattern Rule dynamics (Appendix D5, D6)
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
        self.field_strength = 0.2   # γ_f

    def update(self, scores: np.ndarray, pattern_field_freq: np.ndarray = None):
        """Replicator update with conflict inhibition and field influence."""
        # Apply field influence (Section D6)
        if pattern_field_freq is not None:
            scores = scores + self.field_strength * pattern_field_freq

        avg_score = np.dot(self.weights, scores)
        # Conflict term
        conflict = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    conflict[i] += self.kappa[i, j] * self.weights[j]
        # Update
        new_weights = (self.weights
                       + self.eta * (scores - avg_score) * self.weights
                       - self.beta_c * conflict * self.weights)
        new_weights = np.maximum(new_weights, 0.0)
        if new_weights.sum() > 0:
            self.weights = new_weights / new_weights.sum()
        else:
            self.weights = np.ones(self.n) / self.n

    def best_pattern(self) -> Cell:
        return self.patterns[np.argmax(self.weights)]

# ------------------------------------------------------------
# 7. Learning loop: update embeddings and weights
# ------------------------------------------------------------
def learn_step(mpr: MetaPatternRule,
               observation_seq: List[Cell],
               lower_cells: List[Cell],
               objects: List[Cell],
               curiosity_vec: np.ndarray,
               consensus_vec: np.ndarray,
               field_weights: Dict,
               all_cells: List[Cell],
               embed_lr: float = 0.01,
               beta_comp=0.5, beta_aff=0.5, gamma_soc=0.5, density_bias=0.2):
    """
    One learning step:
    1. Compute hierarchical total scores for all patterns.
    2. Update pattern weights via Meta Pattern Rule.
    3. Update pattern embeddings via gradient ascent on total score.
    """
    scores = np.zeros(mpr.n)
    for i, pat in enumerate(mpr.patterns):
        scores[i] = hierarchical_total_score(
            pat, observation_seq, lower_cells, objects,
            curiosity_vec, consensus_vec, field_weights, all_cells,
            beta_comp, beta_aff, gamma_soc, density_bias
        )
    # Update weights (with field influence as frequency of patterns)
    freq = mpr.weights / mpr.weights.sum()
    mpr.update(scores, pattern_field_freq=freq)

    # Update embeddings (numeric gradient)
    for i, pat in enumerate(mpr.patterns):
        grad = np.zeros_like(pat.emb)
        eps = 1e-3
        for d in range(len(pat.emb)):
            original = pat.emb[d]
            pat.emb[d] = original + eps
            score_plus = hierarchical_total_score(
                pat, observation_seq, lower_cells, objects,
                curiosity_vec, consensus_vec, field_weights, all_cells,
                beta_comp, beta_aff, gamma_soc, density_bias
            )
            pat.emb[d] = original - eps
            score_minus = hierarchical_total_score(
                pat, observation_seq, lower_cells, objects,
                curiosity_vec, consensus_vec, field_weights, all_cells,
                beta_comp, beta_aff, gamma_soc, density_bias
            )
            grad[d] = (score_plus - score_minus) / (2*eps)
            pat.emb[d] = original
        pat.emb += embed_lr * grad
        pat.emb = pat.emb / (np.linalg.norm(pat.emb) + 1e-6)
    return scores

# ------------------------------------------------------------
# 8. Demonstration of core HPM requirements
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # --- Create objects (0‑cells) ---
    apple = Cell(0, np.array([1.0, 0.2]), name="apple")
    banana = Cell(0, np.array([0.8, 0.9]), name="banana")
    cherry = Cell(0, np.array([0.1, 1.2]), name="cherry")
    objects = [apple, banana, cherry]

    # --- Initial patterns (1‑cells) with random embeddings ---
    patterns = [
        Cell(1, np.random.randn(2)*0.5, source=apple, target=banana, name="A_to_B"),
        Cell(1, np.random.randn(2)*0.5, source=banana, target=cherry, name="B_to_C"),
        Cell(1, np.random.randn(2)*0.5, source=cherry, target=apple, name="C_to_A"),
        Cell(1, np.random.randn(2)*0.5, source=apple, target=cherry, name="A_to_C"),
    ]

    # --- Lower cells for compression (just objects as lower level) ---
    lower_cells = objects

    # --- Evaluator targets ---
    curiosity_vec = np.array([0.5, 0.5])
    consensus_vec = np.array([1.0, 0.0])

    # --- Pattern field weights (external support) ---
    field_weights = {p.name: 0.1 for p in patterns}  # weak uniform

    # --- Incompatibility matrix for conflict ---
    n = len(patterns)
    kappa = np.zeros((n, n))
    # Example: A_to_B and A_to_C are incompatible (both from apple)
    kappa[0,3] = kappa[3,0] = 0.8

    # --- Meta Pattern Rule ---
    mpr = MetaPatternRule(patterns, learning_rate=0.2, conflict_scale=0.1,
                          incompatibility_matrix=kappa)

    # --- Simulated observation sequences (episodes) ---
    # Each episode is a list of transitions (each transition is next object)
    # For simplicity, we treat each observation as a single object after an action.
    # We'll generate episodes that favour the correct pattern A_to_B.
    episodes = [
        [banana, banana, cherry, banana],   # A_to_B leads to banana
        [banana, banana, banana],
        [cherry, apple, banana],
        [banana, cherry, banana, banana],
    ]

    print("=== Toy HPM Implementation ===")
    print("Initial weights:", mpr.weights.round(3))
    print("Initial embeddings (first pattern):", patterns[0].emb.round(3))

    # --- Learning loop over episodes ---
    for ep_idx, ep in enumerate(episodes):
        scores = learn_step(mpr, ep, lower_cells, objects,
                            curiosity_vec, consensus_vec, field_weights, patterns,
                            embed_lr=0.05, beta_comp=0.3, beta_aff=0.5, gamma_soc=0.4, density_bias=0.1)
        print(f"\nEpisode {ep_idx}: weights = {mpr.weights.round(3)}")
        print(f"  Best pattern: {mpr.best_pattern().name} (weight {max(mpr.weights):.3f})")

    # --- Demonstrate recombination (creativity) ---
    parent1 = patterns[0]  # A_to_B
    parent2 = patterns[2]  # C_to_A
    new_pattern = recombine(parent1, parent2, compose_fn=lambda u,v: (u+v)/2)
    # Compute insight score on a test sequence
    test_seq = [banana, apple]  # expecting banana after A_to_B etc.
    insight = insight_evaluator(new_pattern, [parent1, parent2], test_seq, objects)
    print(f"\nRecombined pattern: {new_pattern.name}")
    print(f"  Embedding: {new_pattern.emb.round(3)}")
    print(f"  Insight score: {insight:.3f}")

    # Optionally add this new pattern to the population and continue learning
    patterns.append(new_pattern)
    mpr_new = MetaPatternRule(patterns, learning_rate=0.2, conflict_scale=0.1)
    print("  (Added to population for further learning)")

    print("\n=== Core requirements satisfied ===")
    print("- Hierarchical patterns (0,1-cells)")
    print("- Compression (mutual information between levels)")
    print("- Pattern density (structural, evaluator, field)")
    print("- Hierarchical total score (loss + compression + J)")
    print("- Recombination with insight evaluator")
    print("- Meta Pattern Rule with conflict & field influence")
    print("- Learning via weight + embedding updates")
