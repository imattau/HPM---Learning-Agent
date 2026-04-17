# SP-Chess2: Autonomous Tactic Discovery via Self-Play

## Objective
Test whether HPM can autonomously discover reusable tactical patterns (forks, pins, etc.) from experience (self-play) without pre-defined examples. This proves HPM's ability to learn deep structure from raw experience.

## Phased Implementation Plan

### Phase 1: Chess Domain Infrastructure
Implement the HPM components for the chess domain.

- **`ChessDomainConfig`**:
    - Primitives: `LEGAL_MOVES`, `MAKE_MOVE`, `UNDO_MOVE`, `GET_PIECE`, `GET_COLOR`, `IS_ATTACKED`, `PIECE_VALUE`.
    - Mathematical ops for scoring.
- **`ChessRenderer`**:
    - Translates chess concepts into executable code using `python-chess`.
- **`ChessOracle`**:
    - Encodes board features (material balance, attack flags, mobility) into the 20-D scientific state vector.

### Phase 2: Self-Play and Data Gathering
Create the mechanism for agents to learn from experience.

- **Self-Play Loop**:
    - Agents play games using Minimax (Depth 1 or 2).
    - Record (Board, Move) pairs that result in material gain within 2 plies.
- **Data Filtering**:
    - Filter examples to ensure they represent robust material gain, not just noise.

### Phase 3: Tactic Discovery Loop
Implement the autonomous learning process.

- **Discovery Strategy**:
    - Run BFS over chess primitives to find macros that explain the material gain in the gathered examples.
    - Validate macros across multiple board states to ensure generalization.
- **Macro Registration**:
    - Register successful tactics as reusable HFN nodes.

### Phase 4: Competitive Evaluation
- **Win Rate Tracking**:
    - Measure learner's win rate against a material-only baseline over generations.
- **Library Growth**:
    - Audit the discovered tactics (e.g., does it find a fork? a pin?).
- **Generational Scaling**:
    - Run for 5-10 generations and plot performance.

## Key Files
- `hpm_ai_v2/domains/chess_domain.py`
- `hpm_ai_v2/domains/chess_renderer.py`
- `hpm_ai_v2/utils/oracle/chess_oracle.py`
- `hpm_ai_v2/experiments/experiment_sp_chess2_tactic_discovery.py`

## Verification & Metrics
- **Metric 1: Win Rate vs Baseline**: Significant improvement over generations (e.g., from 50% to >70%).
- **Metric 2: Tactic Generalization**: Macros must apply to board states not seen during discovery.
- **Metric 3: Structural Complexity**: Number of primitives combined in discovered macros (aim for depth 3-5).

## Prerequisites
- Install `python-chess`: `pip install python-chess`
