# SP27: Sovereign ARC Solver — Design Specification

## 1. Overview and Rationale

The **Sovereign ARC Solver** (SP27) is the culminating integration of the multi-process, generative HPM architecture (SP18-SP26) applied to the Abstraction and Reasoning Corpus (ARC).

Previous ARC experiments were purely perceptual—they could evaluate if an input-output pair matched a known prior (e.g., "This looks like a rotation"). SP27 closes the cognitive loop. The system must observe the training examples of an ARC task, deduce the generative rule using decentralized "Stereo Vision," and then use the **Agnostic Decoder** to actively construct the output grid for the test example.

This tests the system's ability to transition from **Pattern Recognition** to **Pattern Application (Generativity)** in a complex, multi-domain environment.

## 2. The Multi-Process Architecture

The system utilizes a decentralized cluster coordinated by a Governor:

### 2.1 The Perceptual Cluster (Bottom-Up)
1.  **Spatial Specialist**: Observes 100D grid deltas ($\Delta = Output - Input$). Seeded with rigid geometric priors (rotations, translations).
2.  **Symbolic Specialist**: Observes 30D numerical invariants (color counts, dimensions, parity). Seeded with math priors.
3.  **Explorer (Generalist)**: Observes the full 150D mixed-modal vector. High plasticity to catch un-modeled novelty.

*Routing*: The Governor broadcasts the training examples to all active workers (SP26: Emergent Routing). Each worker uses its Competence Gate to claim or ignore the observation.

### 2.2 The Generative Cluster (Top-Down)
1.  **Generative Governor**: Responsible for formulating the abstract "Goal Node" based on the patterns discovered by the perceptual cluster.
2.  **Spatial Decoder (L1_Motor)**: A dedicated process that runs the `hfn.Decoder`. It takes an abstract geometric rule (e.g., `Rotate_90`) and collapses it into a concrete 100D pixel delta.

## 3. The Generative ARC Workflow

For a given ARC Task with $N$ train examples and 1 test example:

### Phase 1: Decentralized Observation (The "Aha!" Phase)
1.  **Broadcast**: The Governor streams the $N$ train examples (Input/Output pairs) to the Perceptual Cluster.
2.  **Stereo Vision Detection**: The Governor collects the "Explanation Winners" from the specialists. It looks for a consistent **Joint Identity** across all training examples (e.g., "In all 3 examples, Spatial claims 'Rotation' and Symbolic claims 'Parity=Even'").
3.  **Rule Formulation**: The Governor defines this consistent Joint Identity as the **Generative Rule Node ($R$)**.

### Phase 2: Goal Formulation
1.  The Governor observes the **Test Input** grid.
2.  It constructs an abstract **Goal Node ($G$)**:
    *   $G.\mu$ is seeded with the properties of the Test Input.
    *   $G$ is given a topological edge: `HAS_RULE -> R`.
    *   $G.\Sigma$ is set very high (it is an unsolved goal).

### Phase 3: Sovereign Decoding (Execution)
1.  The Governor dispatches $G$ to the **Spatial Decoder** process.
2.  **Variance Collapse**: The Decoder uses the `Agnostic Decoder` algorithm (SP22) to resolve $G$ into a concrete 100D pixel delta that satisfies the topological constraint of rule $R$.
3.  **Active Curiosity (Optional)**: If the Decoder stalls (e.g., it knows it needs to apply rule $R$, but lacks the specific concrete leaf for the Test Input's geometry), it emits a `ResolutionRequest` (SP24). The Governor triggers the Explorer to scan the context, learn the missing piece, and retries.

### Phase 4: Output Construction
1.  The Decoder returns a concrete 100D $\Delta$ vector.
2.  The Governor constructs the final output: `Test_Output = Test_Input + \Delta`.

## 4. Evaluation Metrics

1.  **Rule Consistency**: Does the system identify the same generative rule across all training examples of a task?
2.  **Generative Accuracy**: Does the decoded $\Delta$ vector correctly transform the Test Input into the ground-truth Test Output?
3.  **Decentralized Efficiency**: Does the broadcast/claim model effectively filter out noise (e.g., Symbolic specialist ignoring purely spatial tasks)?
4.  **Curiosity Triggers**: How often does the Decoder need to invoke Demand-Driven Learning to solve the test case?

## 5. Implementation Roadmap

1.  **Data Loader**: Reuse `arc_sovereign_loader.py` for multi-modal feature extraction.
2.  **Generative Target Forest**: Construct a `Target Forest` for the Spatial Decoder containing valid concrete grid deltas.
3.  **Experiment Script**: Implement `experiment_sovereign_arc_solver.py`.
    *   Setup the 3 perceptual workers and 1 generative worker.
    *   Implement Phase 1 (Broadcast/Rule Extraction).
    *   Implement Phase 2 & 3 (Goal Formulation and Decoding).
    *   Implement Phase 4 (Grid reconstruction and scoring).
