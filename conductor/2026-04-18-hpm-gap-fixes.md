# Plan: HPM 4-Gap Fix Implementation

## Objective
Implement four key architectural gaps in the HPM framework to improve learning dynamics, meta-cognition, and temporal pattern management.

## Key Files & Context
- `hpm_ai_v2/agents/base_agent.py`: Primary agent logic for decay, boredom, and temporal fields.
- `hpm_ai_v2/utils/meta_controller.py`: Meta-cognition logic for trend tracking.

## Implementation Steps

### 1. Gap 1: No Decay/Forgetting (L2-L4)
- Add `_pattern_decay_times` and `_decay_half_life` to `BaseHFNAgent.__init__`.
- Implement `get_weight_with_decay(node_id)` in `BaseHFNAgent`.
- Update `_try_bfs` in `base_agent.py` to use `get_weight_with_decay`.

### 2. Gap 2: No Boredom Mechanism (L5 Meta-Cognition)
- Add `_pattern_usage_count` and `_boredom_alpha` to `BaseHFNAgent.__init__`.
- Modify `select_next_task` in `base_agent.py` to apply boredom satiation to curiosity probabilities.
- Increment usage count in `solve()` upon successful pattern application.

### 3. Gap 3: L5 Meta-Pattern Layer Expansion (Trend Tracking)
- Add `_strategy_trends` to `MetaStrategyController.__init__` in `meta_controller.py`.
- Update `record()` to compute success rate trends (deltas).
- Implement `rank_strategies_with_trends()` to prioritize improving strategies.

### 4. Gap 4: Temporal Pattern Field (L3-L5 Recency)
- Add `_pattern_timestamps`, `_recency_half_life`, and `_recency_boost` to `BaseHFNAgent.__init__`.
- Implement `_apply_recency_weight(node_id, base_weight)` helper in `BaseHFNAgent`.
- Update `solve()` to record pattern usage timestamps.
- Integrate recency weighting into `_try_bfs` weight calculations.

## Verification & Testing
- **Decay Test**: Verify that pattern weights decrease over time without use.
- **Boredom Test**: Verify that the probability of selecting a task decreases with repeated successful solves.
- **Meta-Trend Test**: Verify that the meta-controller correctly identifies and prioritizes strategies with positive success trends.
- **Recency Test**: Verify that recently used patterns receive a temporary weight boost in BFS.
