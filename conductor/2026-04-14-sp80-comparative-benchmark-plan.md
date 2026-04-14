# SP80: Comparative Benchmark – HPM vs. Published Few-Shot Results

## Objective
Demonstrate HPM's superior sample efficiency and accuracy on symbolic list-to-list transformations compared to published baselines (GPT-4, Fine-Tuned Transformers, MAML) using a purely evaluative benchmark script. No deep learning models will be trained; literature baselines will be cited directly.

## Key Files & Context
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp80_comparative_benchmark.py` (New file)

## Proposed Strategy
We will implement an automated benchmark script that evaluates HPM across 4 specific list transformation tasks at various $k$-shot levels (1, 2, 3, 5). The agent will attempt to solve the tasks using its built-in BFS and macro-composition strategies. The results will be aggregated and printed in a Markdown table alongside hardcoded, literature-backed baseline estimates.

### Task Definitions
1. `add_one`: Add 1 to each element (MAP).
2. `double`: Multiply each element by 2 (MAP).
3. `filter_positive`: Keep positive numbers (FILTER).
4. `compose_add1_double`: Add 1, then double (COMPOSED MAP).

### Agent Configuration
We will use the `SocialAnalogicalAgent` (or `InducedSchemaAgent`) configured for the List domain. It will have access to standard list primitives and use `exact` and `bfs` strategies.

### Evaluation Protocol
For each task and for $k \in \{1, 2, 3, 5\}$:
1. **Trials**: Run 5 independent trials to ensure stability.
2. **Training Phase**: Generate $k$ random lists (length 3-6, values -10 to 10) and compute their expected outputs. Provide these to the agent to solve and register a macro.
3. **Testing Phase**: Generate 20 new, unseen random lists. Use the agent's learned macro (via the `exact` strategy) to predict the outputs.
4. **Metric**: Compute exact-match accuracy across the 20 test lists.
5. **Reset**: Clear the agent's state (forest, macros) between trials to ensure a true zero-knowledge start for each run.

### Output Generation
The script will output a formatted Markdown table comparing the measured HPM accuracy against the illustrative baseline numbers provided in the specification.

## Verification & Testing
- The script should run without errors.
- HPM should reliably achieve 100% accuracy on the test sets, matching the expected performance outlined in the specification.
- The output table should perfectly align with the requested format.