# SP-Chess4: Multi-Agent Competition with Fractal Forward Models

## Objective
Extend SP-Chess3 to a multi-agent competition setting where two HPM agents play against each other using a **shared fractal board node** and their **learned L4 forward models** (transition deltas). This experiment validates that the learned "mental simulation" is robust enough for competitive decision-making.

## Key Changes

### 1. Shared Fractal Game Space
- The game loop maintains a single `BoardNode` (fractal HFN representation) that both agents observe.
- After each physical move (by `python-chess`), the `BoardNode` is updated to reflect the new ground truth.

### 2. Mental Chess Agent (`hpm_ai_v2/agents/mental_chess_agent.py`)
- **`MentalChessAgent`**:
    - Uses the `ChessTransitionLearner` (learned L4 deltas) for board evaluation.
    - Instead of simulating moves with the library, it uses `predict(board_node, move)` to hallucinate the next board summary `mu`.
    - Evaluates the predicted `mu` to score the move (e.g., material gain).
    - Supports **multi-ply mental simulation** by composing deltas (depth 2+).

### 3. Competition Loop (`hpm_ai_v2/experiments/experiment_sp_chess4_multi_agent.py`)
- Orchestrates games between:
    - **Mental (L4) Agent** vs. **Baseline (Random)**.
    - **Mental (L4) Agent** vs. **Material-Only (Library-Simulated)**.
    - **L4 Agent (Depth 2)** vs. **L4 Agent (Depth 1)**.
- Tracks win rates and decision alignment with the physical ground truth.

## Verification Plan

### Phase 1: Shared Space & Basic Play
- Verify that both agents can correctly observe and interact with the same `BoardNode`.
- Run a small set of games (5-10) against a random baseline.
- **Goal**: Win rate > 80% against random baseline.

### Phase 2: Depth-1 Mental Simulation
- Play against a material-only baseline (which uses the library for simulation).
- **Goal**: Win rate > 50% (proving mental simulation is competitive with ground truth simulation).

### Phase 3: Depth-2 Mental Composition
- Enable 2-ply mental simulation (Agent imagines its move and the opponent's reply using its L4 deltas).
- **Goal**: Depth-2 Mental Agent > Depth-1 Mental Agent (proving delta composition works for planning).

### Phase 4: Shared Forest / Physics
- Have both agents share the same `ChessTransitionLearner` (learned physics of the domain).
- Verify that shared knowledge leads to stable, competitive play.
