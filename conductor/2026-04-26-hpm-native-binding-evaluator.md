# HPM-Native Binding Evaluator & L3 Grounding Plan

**Goal:** Implement a prediction-error driven binding evaluator that grounds discourse entities in the L3 latent state hierarchy, closing the HPM loop between symbolic relational state and probabilistic pattern substrates.

## Architecture

1.  **Evaluator Pressure:** Transition from static entity registration to a dynamic evaluator that reinforce/decays bindings based on predictive hits.
2.  **Latent Grounding (L3 Seam):** Map entities to L3 "soft-state" codes, allowing the agent to predict entities from the pattern hierarchy rather than just surface recency.
3.  **Top-Down Feedback:** Feed prediction success back into L3 via `topdown_gate`, making the hierarchy learn which latent states reliably predict specific entities.

## Implementation Steps

### 1. Core Agent Updates
- **HPMAgent (`hpm_ai_v4/agents/agent.py`):**
    - Add `self._pending_feedback` buffer to `__init__`.
    - Update `perceive_and_learn` to consume and merge `_pending_feedback` into the `feedback` signal.
- **LayeredAgent (`hpm_ai_v4/simulations/layered_agent.py`):**
    - Add `self._pending_feedback` buffer to `__init__`.
    - Update `perceive` to consume and merge `_pending_feedback`.
    - Add `l3_soft_state()` method to expose the discrete L3 latent state code.

### 2. Discourse System Updates (`hpm_ai_v4/simulations/chat_simulation.py`)
- **DiscourseState:** Update `entity_registry` default entry in `_update_entity_registry` to include:
    - `prediction_hits`, `prediction_misses` (counts)
    - `stability_score` (running accuracy)
    - `latent_states` (recent L3 states)
    - `dominant_latent_state` (mode of L3 states)
- **BasicChatSession:**
    - Add `_predict_next_entity()`: Combined latent-match + surface-heuristic predictor.
    - Add `_tick_binding_evaluator(predicted, observed_tokens)`: Evaluator step that updates confidence and stability.
    - Add `_feed_binding_prediction_to_l3(predicted_entity, matched)`: Feeds top-down gate/suppression back to the agent.
    - Implement `observe_text(text, role, dialogue_act, ...)`: Wraps `agent.observe_text` with sentence-level bracketing for the binding evaluator.
    - Update `_entity_registry_scores()` to include `stability_score` as a weighting factor.
    - Update `_update_entity_registry()` to record L3 latent states for each entity mention.

### 3. Integration
- Update `BasicChatSession.chat_turn` (and other observation points) to use `self.observe_text` instead of `self.agent.observe_text`.

## Verification
- Unit test for `LayeredAgent.l3_soft_state()`.
- Unit test for `BasicChatSession.observe_text` bracketing (verifying `prediction_hits` increments).
- Smoke test with a short dialogue to verify entity stability over multiple turns.

---
**Note:** This implementation satisfies HPM pattern criteria: Coherence (latent grounding), Predictive Utility (evaluator pressure), and Stability (selection-driven persistence).
