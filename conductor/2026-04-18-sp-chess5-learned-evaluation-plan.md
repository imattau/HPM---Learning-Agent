# SP-Chess5: Learned Evaluation via HFN

## Objective
Replace the hardcoded `material_balance` heuristic with a **learned HFN value function**. The agent will learn to predict the "Win Probability" or "Board Quality" directly from the fractal board representation based on the outcomes of self-play games.

## Key Changes

### 1. HFN Value Function (`hpm_ai_v2/utils/chess_evaluator.py`)
- **`ChessValueLearner`**:
    - Maintains an HFN node (or small forest) tasked with predicting game outcomes.
    - Input: `BoardNode` (832-D squares or 20-D summary).
    - Output: Scalar value in range [0, 1] (0 = Black win, 1 = White win, 0.5 = Draw).

### 2. Reinforcement Learning Update
- After each game, the agent collects (board_state, final_outcome) pairs.
- **Value Update**: Adjust the evaluation node's `mu` (weights) to minimize the error between prediction and actual outcome (Monte Carlo learning).
- `delta_mu = learning_rate * (outcome - predicted_value) * state_gradient` (simplified HPM update).

### 3. Integrated Planning Loop
- Update `MentalChessAgent` to use the `ChessValueLearner` for leaf-node evaluation.
- The agent now performs **Search over learned physics (L4) + Evaluation via learned strategy (L1/L2)**.

### 4. Experiment Script (`hpm_ai_v2/experiments/experiment_sp_chess5_learned_eval.py`)
- **Generational Learning**:
    1. Agent plays games against random/baseline.
    2. Learns value function from outcomes.
    3. Competes against "Hardcoded Material" version.
- Track "Value Alignment": Correlation between learned evaluation and ground-truth material balance.

## Verification Plan

### Phase 1: Value Prediction Training
- Train the value function on 1,000 completed games (random play).
- Verify that the model can correctly distinguish between "won" and "lost" positions in a test set.
- **Goal**: Correlation with final outcome > 0.4.

### Phase 2: Learned vs. Hardcoded Match
- Play a match between **Learned Evaluator Agent** and **Material Baseline Agent**.
- **Goal**: Learned agent should be competitive (Win rate > 40%) even without hardcoded material rules.

### Phase 3: Compositional Synergy
- Verify that the Depth-2 planning works with the learned evaluator.
- **Goal**: Depth-2 (Learned) > Depth-1 (Learned).
