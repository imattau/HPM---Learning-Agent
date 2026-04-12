# SP57: Operator-Level Compositional Abstraction (Design)

## 1. Objective
To elevate the HPM framework from linear vector arithmetic (as demonstrated in SP56) to **True Operator-Level Composition**. This experiment will demonstrate that an HFN agent can synthesize complex, non-linear meta-patterns zero-shot by chaining parameterized mathematical transformations ($f(g(x))$), rather than merely adding static geometric deltas ($v_1 + v_2$).

## 2. The Paradigm Shift: From Vectors to Operators
In SP56, relations (L2) and meta-relations (L3) were represented as static displacement vectors ($\Delta x$). Composition was strictly linear addition ($L3 \approx L2_A + L2_B$). 

In SP57, the latent manifold must encode **Transformations (Operators)**. 
- **Representation**: A relational node is no longer just a vector `[+1, 0, 0]`; it is a mapping function, represented as a transformation matrix, a neural operator, or a parameterized geometric operation (e.g., $x_{t+1} = M \cdot x_t + b$).
- **Composition**: Composition becomes function application. Instead of $L3 = L2_A + L2_B$, the system discovers $L3(x) = L2_A(L2_B(x))$.
- **Why this matters**: This allows the system to handle non-linear dynamics, geometric growth, and true rule-chaining, breaking the dependence on purely additive state spaces.

## 3. Experimental Curriculum

### Phase 1: Operator Pre-training (L2 Primitives)
Train the agent on distinct, non-linear and linear primitives independently.
*   **Domain**: Numeric values mapped to a continuous continuous embedding space.
*   **Primitive A (Additive)**: `Add_1` ($x \to x + 1$)
*   **Primitive B (Multiplicative)**: `Mul_2` ($x \to x \times 2$)

### Phase 2: Meta-Pattern Discovery (L3)
Train the agent on prolonged sequences of constant operator application to stabilize L3 meta-nodes.
*   **Constant Addition**: `[1, 2, 3, 4]` (Constant `Add_1`)
*   **Constant Multiplication**: `[1, 2, 4, 8]` (Constant `Mul_2`)

### Phase 3: Out-of-Family Generative Composition (Zero-Shot Transfer)
Present a highly non-linear, out-of-family sequence that cannot be solved by simple addition or a single operator.
*   **The Test Sequence**: $1 \to 3 \to 7 \to 15 \to 31 \dots$
*   **The Underlying Dynamic**: $x_{t+1} = (x_t \times 2) + 1$
*   **The Task**: 
    1. The agent is primed with noisy observations of $t=0 \dots 3$.
    2. The agent must infer the non-linear trajectory.
    3. The agent searches its Long-Term Memory of operators and uses a search algorithm (e.g., beam search via `CognitiveSolver`) to compose a chain of operators.
    4. The agent discovers that `Add_1(Mul_2(x))` explains the sequence.
    5. The agent uses this composed operator to actively predict $t=4 \dots 9$.

## 4. Architecture Updates Required
1.  **OperatorOracle**: A new oracle that embeds states such that transformations can be extracted as matrices/operators rather than just subtraction deltas.
2.  **Multi-Step Generative Synthesis**: Replace the $O(n^2)$ pairwise vector addition search from SP56 with a combinatorial operator search. The agent must evaluate chains of length 1, 2, and potentially 3.
3.  **Non-Linear Autoregressive Loop**: The prediction loop must apply the composed operator to the current state ($State_{t+1} = ComposedOperator(State_t)$), propagating the non-linear constraint forward.

## 5. Success Metrics and Baselines

| Baseline | Mechanism | Expected Result on $x \times 2 + 1$ |
| :--- | :--- | :--- |
| **L2-Only (Constant Add)** | Assumes $x_{t+1} = x_t + \Delta$ | **FAIL (Diverges rapidly)** |
| **SP56 Vector Synthesis** | Assumes $x_{t+1} = x_t + (\Delta_A + \Delta_B)$ | **FAIL (Cannot handle multiplication)** |
| **Single Operator** | Assumes $x_{t+1} = Mul\_2(x_t)$ | **FAIL (Misses the +1 offset)** |
| **SP57 Operator Composition** | Synthesizes $x_{t+1} = Add\_1(Mul\_2(x_t))$ | **SUCCESS (Near-zero error)** |

## 6. Verification
Achieving near-zero error on the non-linear $x \times 2 + 1$ sequence exclusively in the SP57 condition will provide definitive proof that the HPM framework supports true **Symbolic/Operator-Level Compositional Abstraction**, moving beyond linear vector spaces into universal function induction.
