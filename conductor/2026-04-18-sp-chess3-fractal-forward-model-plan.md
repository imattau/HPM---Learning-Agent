# SP-Chess3: Fractal Board Representation and L4 Forward Model

## Objective
Implement SP-Chess3 to demonstrate HPM's ability to:
1. Represent a complex, structured state (chess board) as a fractal HFN node.
2. Learn the deterministic dynamics of chess (transitions) as L4 delta macros.
3. Use learned transitions for mental simulation and tactical discovery without an external engine.

## Key Changes

### 1. Fractal State Representation (`hpm_ai_v2/domains/chess_hfn.py`)
- **`ChessHFNEncoder`**:
    - `square_to_node`: 13-D one-hot (empty + 12 piece types).
    - `board_to_node`: 64 square nodes + 20-D summary `mu` (material, counts).
    - `move_to_node`: [from_idx, to_idx, promotion] vector.

### 2. L4 Transition Learning (`hpm_ai_v2/agents/chess_l4_agent.py`)
- **`ChessTransitionLearner`**:
    - Collect (board, move, next_board) triples.
    - Compute `delta = next_board_mu - board_mu` (832-D for squares, 20-D for summary).
    - Store average deltas per move as L4 transition nodes.

### 3. Forward Model & Mental Simulation
- `predict_next_board(board_node, move_node)`: `pred_mu = board_mu + move_delta_mu`.
- Evaluate prediction accuracy (MSE) on held-out test data.

### 4. Tactical Discovery via Simulation (`hpm_ai_v2/experiments/experiment_sp_chess3_forward_model.py`)
- Use the learned forward model to score moves (e.g., predicted material gain).
- Compare with SP-Chess2 baseline (engine-simulated) to verify tactical alignment.

## Verification Plan

### Phase 1: Representation & Data
- Generate 5,000 random move triples.
- Verify that `board_to_node` correctly encodes material balance in its summary `mu`.

### Phase 2: L4 Accuracy
- Train L4 transition deltas on 80% of data.
- Test prediction accuracy on 20% held-out data.
- **Goal**: MSE < 0.1 for material balance prediction.

### Phase 3: Tactical Alignment
- For a set of tactical positions, compare move ranking using the **learned forward model** vs. the **chess library**.
- **Goal**: >80% agreement on the top-ranked tactical move.
