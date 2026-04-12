# SP58: Heuristic Operator Induction and Disambiguation (Design)

## 1. Objective
To evolve the HPM framework from exhaustive, brute-force operator composition (SP57) to **Scalable, Noise-Robust Function Induction**. This experiment will demonstrate that an HFN agent can use its latent manifold to heuristically guide the synthesis of deep, non-linear operator chains, successfully disambiguating competing hypotheses under perceptual noise.

## 2. Key Upgrades from SP57

### A. Expanded Hypothesis Space (Non-Linear Operators)
We move beyond strictly affine transformations ($Ax + B$) to include non-linear and piecewise operations. 
*   **New Primitive**: `Mod_N` ($x \to x \pmod N$). This introduces periodicity and breaks the purely continuous geometric growth of previous tests.

### B. Heuristic-Guided Synthesis (Breaking $O(n^2)$)
Instead of exhaustively enumerating all $1$ and $2$-step chains, the agent will use a **Beam Search** guided by the HFN's residual error. 
*   **Mechanism**: If a partial chain $f(x)$ leaves a residual error $\Delta$, the agent queries the HFN Retriever for a primitive operator $g(x)$ that best explains that $\Delta$. It then forms the new candidate $g(f(x))$. This allows scaling to deeper chains (e.g., length 3) without combinatorial explosion.

### C. Robustness to Ambiguity and Noise
The true test of intelligence is maintaining multiple hypotheses when data is scarce and noisy.
*   **Ambiguity**: The first transition ($1 \to 3$) can be explained by multiple operators (`Add_2`, `Mul_3`, `Mul_2 ∘ Add_1`). The agent must maintain a "beam" of valid hypotheses.
*   **Disambiguation**: As the sequence unfolds ($1 \to 3 \to 7 \to 5 \dots$), the competing hypotheses will diverge. The agent must use subsequent noisy observations to prune the beam and converge on the true generative model.
*   **Perceptual Noise**: Gaussian noise is added to the L1 observations, forcing the agent to evaluate operators probabilistically rather than relying on perfect zero-error arithmetic.

## 3. Experimental Curriculum

### Phase 1: Operator Pre-training (L2 Primitives)
Train the agent on distinct, independent primitives.
*   `Add_1` ($x \to x + 1$)
*   `Mul_2` ($x \to x \times 2$)
*   `Mod_10` ($x \to x \pmod{10}$)

### Phase 2: Meta-Pattern Discovery (L3)
Stabilize the L2 operators by presenting constant sequences of each primitive.

### Phase 3: The Disambiguation Test (Generative Synthesis)
Present a highly non-linear, ambiguous sequence with perceptual noise.
*   **The Test Sequence**: $x_{t+1} = ((x_t \times 2) + 1) \pmod{10}$
*   **Values**: `1 -> 3 -> 7 -> 5 -> 1 -> 3 ...`
*   **The Task**:
    1. **Priming ($t=0 \dots 3$)**: The agent observes `[1.02, 2.98, 7.05, 4.99]` (noisy).
    2. **Hypothesis Generation**: At $t=1$, the agent generates multiple plausible chains (`Add_2`, `Mul_3`, `Mul_2 ∘ Add_1`).
    3. **Heuristic Beam Search**: The agent expands its beam up to length 3, guided by residual error retrieval.
    4. **Disambiguation**: By $t=3$ (value `~5`), the `Add_2` and `Mul_3` hypotheses fail spectacularly. Only the composed chain `Mod_10 ∘ Add_1 ∘ Mul_2` survives.
    5. **Prediction**: The agent uses the winning chain to autoregressively predict $t=4 \dots 9$.

## 4. Success Metrics and Baselines

| Baseline | Mechanism | Expected Result on Test Sequence |
| :--- | :--- | :--- |
| **Brute-Force (Depth 2)** | Exhaustive search up to length 2 | **FAIL** (Requires depth 3: Mul, Add, Mod) |
| **Greedy Heuristic** | Picks best 1-step operator at each $t$ | **FAIL** (Gets trapped in local minima early on) |
| **SP58 Heuristic Beam Search** | Manifold-guided search with depth 3 | **SUCCESS** (Correctly disambiguates and stabilizes) |

## 5. Verification
If the SP58 agent successfully synthesizes the depth-3 non-linear operator `Mod_10(Add_1(Mul_2(x)))` in the presence of noise and ambiguity, significantly outperforming greedy and shallow search baselines, it provides definitive proof that the HPM framework supports **Scalable, Noise-Robust Function Induction**.
