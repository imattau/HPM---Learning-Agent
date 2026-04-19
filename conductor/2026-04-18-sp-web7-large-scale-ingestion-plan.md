# Plan: SP-Web7 - Large-Scale Modular Ingestion & Thematic Synthesis

The goal is to verify the modular Reader/Librarian architecture by ingesting an expanded scientific corpus and performing cross-domain thematic synthesis.

## Objective
- Ingest 8 specialized scientific papers covering Physics, Math, Biology, Chemistry, Computer Science, Psychology, Astronomy, and Sociology.
- Verify that the `ReaderAgent` handles low-level perception correctly.
- Verify that the `LibrarianAgent` autonomously discovers topics and maps themes across these domains.
- Stress-test the recent performance optimizations (Lazy Initialization, Auto-Fitting).

## Key Files & Context
- `hpm_ai_v2/agents/reader_agent.py`: Perception specialist.
- `hpm_ai_v2/agents/librarian_agent.py`: Concept discovery specialist.
- `hpm_ai_v2/experiments/experiment_sp_web7_large_scale_ingestion.py`: New experiment script.
- `data/scientific_curiosity_v2`: New knowledge base directory.

## Implementation Steps

### 1. Preparation
- [ ] Define the `EXPANDED_SCIENTIFIC_CORPUS` in the new experiment file.
- [ ] Set up the `AgentOrchestrator` and `TieredForest`.

### 2. Society Initialization
- [ ] Initialize `LibrarianAgent`.
- [ ] Initialize `ReaderAgent` and link it to the `LibrarianAgent`.
- [ ] Initialize supporting agents: `DictionaryAgent`, `PhysicsAgent`, `MathAgent`, `SentimentAgent`.
- [ ] Register all agents with the `Orchestrator`.

### 3. Execution (Ingestion)
- [ ] Loop through the corpus and ingest each paper using `reader.ingest_text()`.
- [ ] Observe the `LibrarianAgent`'s autonomous discovery process (triggered by the Reader).

### 4. Verification & Synthesis
- [ ] **Topic Audit:** List all discovered topics from the `LibrarianAgent`.
- [ ] **Cross-Domain Search:** Perform semantic searches for concepts that span domains (e.g., "Energy", "Systems", "Evolution").
- [ ] **Thematic Summaries:** Generate summaries for at least 3 high-level themes.
- [ ] **Forest Diagnostics:** Check the final node count and relation type distribution.

## Verification & Testing
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp_web7_large_scale_ingestion.py`.
- Verify that no broadcasting errors occur during the large-scale reindexing.
- Verify that the initialization remains fast despite the growing forest.
