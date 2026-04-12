# SP59: Manifold-Guided Operator Induction (Design)

## 1. Objective
To elevate the HPM framework to true **Manifold-Guided Function Induction**. This experiment will replace heuristic, brute-force beam search with a search process that is actively guided by the HFN latent space. The agent will use residual error vectors to query the HFN for the most relevant operators, proving that the geometric manifold can actively direct symbolic synthesis.

## 2. Key Upgrades from SP58

### A. HFN-Guided Beam Expansion
In SP58, the beam search expanded by testing every available primitive. In SP59, the search is guided by the manifold:
*   **Mechanism**: For a partial operator chain $f(x)$, the agent calculates the predicted next state $\hat{x}_{t+1} = f(x_t)$. 
*   **Residual Encoding**: It calculates the residual delta $\Delta = x_{t+1} - \hat{x}_{t+1}$ and encodes this as a 30D vector in the Relational (L2) slice of the manifold.
*   **Latent Retrieval**: It queries the `retriever` with this residual vector. The HFN returns the top-K operators whose geometric signatures best match the required correction.
*   **Targeted Expansion**: The beam only expands using these retrieved, highly-relevant operators.

### B. Multi-Step Rollout Scoring
Instead of evaluating operators on isolated 1-step pairs, the agent will score candidates by rolling them out over a temporal horizon (e.g., $t=0 \dots 3$). This heavily penalizes operators that are locally correct but globally divergent (a common issue with non-linear dynamics under noise).

### C. Robust Operator Deduplication
Deduplication in the beam search will be based on the functional parameters of the operators (e.g., `(round(weight, 2), round(bias, 2), round(modulus, 2))`) rather than string names, ensuring structurally identical chains are properly pruned.

## 3. Experimental Curriculum

### Phase 1 & 2: Operator Pre-training & Stabilization
Train the agent on the same primitives as SP58: `Add_1`, `Mul_2`, `Mod_10`.
Store these heavily in the HFN manifold so they can be retrieved.

### Phase 3: Manifold-Guided Disambiguation Test
*   **Test Sequence**: $x_{t+1} = ((x_t \times 2) + 1) \pmod{10}$
*   **Values**: `1 -> 3 -> 7 -> 5 -> 1 -> 3 ...`
*   **Priming**: Present the first 4 noisy steps.
*   **Task**: The agent must synthesize the depth-3 operator chain `Mod_10 ∘ Add_1 ∘ Mul_2`.
*   **Constraint**: The search must query the HFN at each expansion step. For example, after applying `Mul_2(1) = 2`, the residual to the target `3` is `+1`. The HFN must successfully retrieve `Add_1` to fix this residual.

## 4. Success Metrics and Baselines

| Baseline | Mechanism | Expected Result |
| :--- | :--- | :--- |
| **Unguided Beam Search (SP58)** | Expands using all primitives; 1-step scoring. | **Baseline Performance** (Computationally expensive) |
| **SP59 Manifold-Guided Search** | Expands using HFN retrieval; multi-step rollout. | **SUCCESS** (Requires fewer evaluations, selects robust long-term models) |

## 5. Verification
If the agent successfully synthesizes the correct non-linear operator chain by actively querying the HFN with residual error vectors, it provides definitive proof that the HPM framework unifies **Geometric Latent Representations** with **Symbolic Executable Reasoning**.
