# SP69: Trust and Reputation in Social Pattern Fields Plan

## Objective
Implement Experiment SP69 to validate higher-order social pattern fields (HPM §9.5 and §9.7) by introducing trust and reputation mechanisms. This extends the multi-specialist social learning (MS-SL) framework to handle unreliable agents.

## Key Files & Context
- `hpm_ai_v2/agents/mixins/trust.py` (New): Will contain `TrustMixin`, `ReputationMixin`, and `DomainSpecificTrustMixin`.
- `hpm_ai_v2/experiments/experiment_sp69_trust.py` (New): The main experiment script simulating the phases defined in the design.

## Implementation Steps
1. **Implement Trust Mixins**:
   - `TrustMixin`: Local trust tracking via `trust_scores` dictionary, `update_trust` logic (increase on success, decrease on failure), `should_import`, and `weight_blackboard_entry`.
   - `ReputationMixin`: Broadcast trust to `SocialForest` (as an HFN node), aggregate peer trust via median/mean to compute reputation, and implement `should_exchange_with`.
   - `DomainSpecificTrustMixin`: Extend trust tracking to be per-domain or per-macro-type.
2. **Develop SP69 Experiment Script**:
   - **Setup**: Create 4 agents (Alice: reliable int, Bob: reliable string, Charlie: unreliable int/reliable string, Dave: unreliable random). They will use the new trust mixins alongside existing L2/Social capabilities.
   - **Phase 0**: Individual training with seeded macros. Charlie is seeded with incorrect macros for integers.
   - **Phase 1**: Baseline social exchange without trust. Measure wasted attempts and convergence speed.
   - **Phase 2**: Experimental social exchange with trust and reputation enabled.
   - **Phase 3**: Domain-specific trust test (Alice accepting Charlie's string macros but rejecting integer ones).
3. **Integration**:
   - Ensure the mixins correctly interface with the agent's solve loop (updating trust based on success of imported macros) and the social exchange mechanisms.

## Verification & Testing
- Run `experiment_sp69_trust.py` to verify the hypotheses:
  - **H1**: Trust group converges in ≤ 5 rounds; control takes ≥ 10 rounds.
  - **H2**: Charlie's reputation drops below 0.3 within 3 rounds.
  - **H3**: Trust group consults ≤ 20% of blackboard entries from low-trust peers.
  - **H4**: Domain-specific trust isolations (Alice's trust for Charlie in integer domain < 0.3, string domain > 0.5).
