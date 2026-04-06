# Experiment 16: Multi-Agent HFN Interaction (Social Learning)

**Script:** `experiment_multi_agent_social.py`

## Objective
To evaluate whether the HFN system can support social learning and knowledge transfer between independent agents. This experiment tests if HFN representations are universally decipherable and transmissible, allowing sovereign agents to bootstrap their world models using "dreams" (generated observations) from other agents.

## Setup
- **Curriculum:** Two distinct pattern concepts:
  - **Concept 1:** Localized in Dims 0-5.
  - **Concept 2:** Localized in Dims 10-15.
- **Agent Roles:**
  - **Agent A (Solo):** Learns the full curriculum (C1 + C2) from raw data.
  - **Agent B (Specialist 1):** Learns only Concept 1.
  - **Agent C (Specialist 2):** Learns only Concept 2.
- **Social Phase:** Agents B and C interact by broadcasting "dreams"—decoded concrete observations of their highest-weight structural nodes. They then perform **Social Refinement** by re-observing these dreams to integrate the foreign knowledge.
- **Probe:** Specialists are tested on the domain they only saw via social exchange (e.g., Agent B is probed on Concept 2).

## Results & Analysis
The experiment was a **Success**, demonstrating effective cross-agent knowledge transfer.

1. **Social Transfer Accuracy:** Both social agents reached a surprise level **< 1.0** on foreign domains (Agent B on C2: 0.94, Agent C on C1: 0.94). This proves they successfully integrated the structure of concepts they never saw in the raw data stream.
2. **Dream-Driven Bootstrapping:** The use of the `Decoder` to synthesize "dreams" allowed agents to communicate their abstract structural knowledge as perceived data, which the receiving `Observer` could then process using its standard HFN dynamics.
3. **Refinement Benefit:** Multi-epoch refinement during ingestion was critical for stabilizing social knowledge, allowing the recipient to find optimal co-occurrence and compression for foreign concepts.

### Key Takeaway
HFN knowledge is **transmissible**. Independent agents can parallelize the discovery of a world model and share their high-level abstractions instantly. This proves that HFN can support decentralized "Sovereign Clusters," where intelligence scales through communication and culture rather than being limited by individual experience.
