# HPM AI v2: Specification (Comprehensive)

## Objective
Create a new iteration of the HPM AI from scratch, strictly adhering to the "Human Learning as Hierarchical Pattern Modelling v1.25" working paper. This version implements the complete lifecycle of pattern discovery, stabilization, and progression across developmental levels with explicit qualitative differentiation.

## 1. Core Mathematical Precision

### 1.1 Composite Pattern Structure ($h$)
Every pattern $h$ is a **Generative Model** factorized as:
$p(x, z^{(1)}, z^{(2)}) = p(z^{(2)}) \cdot p(z^{(1)} \mid z^{(2)}) \cdot p(x \mid z^{(1)})$
- **Constituent Features ($S$)**: Parameter set $\theta$. Connectivity $C(h)$ is the graph density of the dependency matrix.
- **Substrate ID ($M$)**:
    - `INTERNAL_FLEX`: Neural-like. High decay, high flexibility.
    - `INTERNAL_PROC`: Motor-like. Low decay, zero flexibility.
    - `EXTERNAL_SYM`: Symbolic/Shared. Zero decay, high verification.

### 1.2 Developmental Level Qualitative Specs (Section 7.4)
- **Level 1 (Surface-Feature)**: Direct encoding of $x$ into $z^{(1)}$. Minimal abstraction. Highly context-bound.
- **Level 2 (Local Structural)**: $z^{(1)}$ captures co-occurrence of Level 1 features. Concrete/context-bound generalization.
- **Level 3 (Relational)**: $z^{(1)}$ factorizes into **Relational Variables**, abstracting away surface identity. Genuine structure detection.
- **Level 4 (Abstract Structural)**: $z^{(2)}$ represents **Invariants** for internal manipulation/simulation without surface cues. Dependency on surface noise ($\alpha_L$) approaches zero.
- **Level 5 (Generative Expert)**: Full hierarchical generative capacity. High density and cross-domain transfer. Includes a **Generative Utility** term in the total score reflecting its role as a "foundation" for recombinative insight.

### 1.3 Epistemic & Hierarchical Loss
- **Instantaneous Loss**: $\ell_i(t) = -\log p_i(x_t \mid h_i)$
- **Sensitivity Split**: $\ell_i(t) = \alpha_L \cdot \text{Loss}_{\text{surface}} + \beta_L \cdot \text{Loss}_{\text{structural}}$. 
    - Level 1: $\alpha=0.9, \beta=0.1$.
    - Level 2: $\alpha=0.6, \beta=0.4$.
    - Level 3: $\alpha=0.2, \beta=0.8$.
    - Level 4: $\alpha=0.05, \beta=0.95$ (Internalized Schemas).
    - Level 5: $\alpha=0.01, \beta=0.99$ (Pure Generative Structure).
- **Hierarchical Loss**: $L_i^{\text{hier}}(t) = \mathbb{E}_{q_i(t)}[\log q_i(t)(z^{(1)}, z^{(2)})] - \mathbb{E}_{q_i(t)}[\log p_i(x, z^{(1)}, z^{(2)})]$
- **Compression (Mutual Information)**: $Comp_i(t) = H_{q_i(t)}[z^{(1)}] - H_{q_i(t)}[z^{(1)} \mid z^{(2)}]$

### 1.4 Meta Pattern Rule Dynamics
Weights $w_i$ evolve via:
$w_i(t+1) = w_i(t) + \eta(Total_i(t) - \bar{Total}(t)) w_i(t) - \beta_c \sum_{j \neq i} \kappa_{ij} w_i(t) w_j(t)$
- **Dynamic Conflict ($\kappa_{ij}$)**: **Hellinger Distance** between predictive distributions.
- **Maturation Gate**: Penalty multiplier if lower-level density is insufficient.
- **Stability Bias**: $Total_i(t) \leftarrow Total_i(t) + \kappa_D D(h_i)$.

### 1.5 Evaluator Signals ($J_i$)
- **Affective ($E_{\text{aff}}$)**: Curiosity, Comfort, and **Insight** ($I(h^*) = \beta_{\text{orig}} (\alpha_{\text{nov}} Nov(h^*) + \alpha_{\text{eff}} Eff(h^*))$).
- **Social ($E_{\text{soc}}$)**: Field alignment $F_i(t)$.
- **Resource ($E_{\text{res}}$)**: Substrate-dependent cost. Scales with level (working memory load).
- **Total Score (Hierarchical)**: $Total_i^{\text{hier}}(t) = -L_i^{\text{hier}}(t) + \beta_{\text{comp}} Comp_i(t) + J_i(t)$

### 1.6 Institutional Fields
- **InstitutionalField (Science)**: Applies a **Replication Filter**. Patterns gain amplification only if they consistently reduce epistemic loss across time (low variance).

## 2. Implementation Plan

- [ ] **Step 1: Foundational Math**: Define distributions and the `CompositePattern` with Level-specific $\alpha_L, \beta_L$ sensitivity.
- [ ] **Step 2: Replicator & Conflict**: Build `PatternPopulation` with Meta Pattern Rule and Hellinger-based $\kappa_{ij}$ discovery.
- [ ] **Step 3: Level 1-2 Progression**: Verify Level 1 stabilization $\to$ Internalization $\to$ Level 2 emergence.
- [ ] **Step 4: Relational Abstraction (Level 3)**: Verify Step 9.1: Level 3 patterns survive surface noise but fail on structural perturbations.
- [ ] **Step 5: Externalization & Scientific Fields**: Implement `EXTERNAL_SYM` and the Replication Filter.
- [ ] **Step 6: Recombinative Insight**: Implement the Appendix E4 boost for novel structural merges.
- [ ] **Step 7: Final Validation**: Full trajectory sweep Level 1 $\to$ 5 verifying the Section 7.4 progression.
