# Sub-project 11: DS-1000 Boss Fight Benchmark — 5-Level Symbolic Data Science

## Goal
Design and develop a new benchmark test based on the DS-1000 data science evaluation. This acts as a "Substrate Grounding" boss fight for the Hierarchical Pattern Modelling (HPM) framework. It evaluates whether the agent can map an abstract transformation (L3 Relational Law) onto a specific, messy Python library API (L1/L2) and use L4/L5 for generative prediction and strategic error monitoring.

## Architecture

We will implement a 5-level stack mapping for symbolic data science tasks spanning libraries like Pandas, NumPy, and Scikit-Learn.

### 1. Task Simulation (`benchmarks/ds1000_sim.py`)
To ensure fast, deterministic testing without requiring arbitrary code execution or large dataset downloads, we will build a mock DS-1000 task generator. 
- **Libraries simulated:** Pandas, NumPy, Scikit-learn.
- **Task format:** 
  - `prompt`: The natural language request (e.g., "Normalise this matrix by column").
  - `library`: The target library API.
  - `input_structure`: Structural features of the input (e.g., matrix, missing values).
  - `output_structure`: Structural features of the expected output.
  - `candidates`: List of code blocks or API sequences, with one `correct_idx`.

### 2. Encoders (`benchmarks/ds1000_encoders.py`)
The encoding pipeline translates the textual/structural data into numerical vectors for the HPM agents.

- **DS1000L1Encoder (Syntax/Tokens):**
  - Encodes the API surface area and boilerplate Python syntax.
  - Feature dim: 32.
  - Returns: A vector representing the surface-level syntax distribution.

- **DS1000L2Encoder (Structural Anatomy):**
  - Identifies specific objects (DataFrames, Tensors, missing value masks).
  - Feature dim: 16.
  - Returns: A structural summary of the data objects involved, threaded with L1 epistemic state.

- **DS1000L3Encoder (Relational Law):**
  - Represents the mathematical or logical transformation required (e.g., "Normalization", "Imputation").
  - Feature dim: 20.
  - Returns: A relational summary threaded with L2 epistemic state.

### 3. Orchestration & Generative Head (`benchmarks/structured_ds1000_l4l5.py`)
- **L4 (Generative Head):** Learns to predict the required L3 transformation directly from the L2 structural anatomy (e.g., recognizing that a DataFrame with NaNs implies an "Imputation" relational law).
- **L5 (Meta-Monitor):** Monitors L4's prediction error (surprise). If surprise is high (e.g., execution error or structural mismatch), it shifts `strategic_confidence` to favor full L3 analytical scoring over L4's intuitive prediction.

## Benchmark Execution
The benchmark will compare three conditions:
1. `l2l3`: Pure analytical scoring using L3 relational matching.
2. `l4_only`: Pure intuitive scoring using L4 generative predictions.
3. `l4l5_full`: Adaptive gating by L5, switching between L4 and L3 based on structural surprise.

## Success Criteria
- The 5-level architecture runs end-to-end on the simulated DS-1000 tasks.
- `l4l5_full` matches or outperforms the pure `l2l3` baseline by effectively using L4 intuitions for familiar structures while falling back to L3 when surprise is high.
- The `--smoke` flag allows for rapid testing in CI environments.