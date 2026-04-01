# SP24: Demand-Driven Learning — Design Specification

## 1. Overview and Rationale

The **Demand-Driven Learning** experiment (SP24) tests the interaction between the generative (top-down) and perceptual (bottom-up) halves of the HPM AI. It implements the "Fail-Learn-Retry" loop: when the Decoder encounters a "Generative Gap" (a missing concrete node needed to fulfill an abstract goal), it triggers the Observer to "learn" the missing information before resuming the action.

This proves that learning in HPM can be **active and functional**, driven by the necessity of achieving a goal, rather than just passive observation of a data stream. Hallucinations are prevented because the Observer must still find real environmental evidence before it will create the requested node.

## 2. The Architecture: "The Curiosity Engine"

The experiment requires a tight feedback loop between three components:

1.  **The Decoder**: Attempts "Variance Collapse." If it fails to find a concrete node matching a target $\mu$ and topological constraints, it returns a `ResolutionRequest`.
2.  **The Governor**: Catches the `ResolutionRequest` and transforms it into a targeted `SearchTemplate` for the Observer.
3.  **The Observer**: Receives the `SearchTemplate`. Instead of its normal passive expansion, it actively queries a "Historical Buffer" (or an external environment) to find a concrete vector $x$ that matches the template. If found, it creates the node.

## 3. The Task: "Point to the Missing Block"

We use a 1D block environment similar to SP22, but introduce an intentional knowledge gap.

*   **The World**: A 1D space containing 3 blocks: Red ($X=2.0$), Blue ($X=5.0$), and Green ($X=8.0$).
*   **The Prior Knowledge**: The system is seeded with priors for Red and Blue, but **it has no prior for Green**.
*   **The Environmental Buffer**: The system has "seen" all three blocks before (they are in a historical observation buffer), but because it had no goal related to Green, the Observer previously ignored the $X=8.0$ signal as residual noise.
*   **The Goal**: `Goal_Point(Target=Green_Block)`.

### 3.1 The Expected Workflow

1.  **Decode Attempt 1**: The Decoder tries to resolve the `Green_Block` concept into a spatial coordinate.
2.  **The Gap**: It fails. There is no concrete leaf node in the Target Forest with the required edge `HAS_COLOR -> GREEN`.
3.  **The Request**: The Decoder yields a `ResolutionRequest(mu=[8.0], constraint="HAS_COLOR->GREEN")`.
4.  **Targeted Observation**: The Governor asks the Observer to scan its Historical Buffer for any vector $x$ near $\mu=[8.0]$.
5.  **The A-ha Moment**: The Observer finds the previously ignored $x=8.0$ vector in the buffer. It creates a new leaf node (`leaf_green_pos`) and registers it in the Forest. Crucially, the Governor binds this new leaf to the `GREEN` concept edge.
6.  **Decode Attempt 2**: The Governor retries the decode. The Decoder now finds `leaf_green_pos` and successfully outputs `[X=8.0]`.

## 4. Decoder Modification

The `hfn.decoder.Decoder` must be updated to return a structured failure rather than an empty list.

*   **Current**: Returns `[]` if no candidates are found.
*   **New**: Returns a `ResolutionRequest` object containing the abstract node's $\mu$, $\Sigma$, and unfulfilled topological edges.

## 5. Evaluation Metrics

1.  **Gap Detection**: Does the Decoder correctly identify *what* is missing (the required $\mu$ and edges) rather than just crashing?
2.  **Targeted Learning**: Does the Observer successfully use the request to "rescue" the ignored data from the historical buffer?
3.  **Resumption Success**: Can the Decoder successfully complete the goal after the Observer has fulfilled the request?
4.  **Anti-Hallucination Guard**: If we ask it to point to a "Yellow Block" ($X=10.0$), but $X=10.0$ is NOT in the historical buffer, does it correctly refuse to create the node and definitively fail the goal?

## 6. Implementation Roadmap

1.  **Decoder Update**: Modify `hfn/decoder.py` to yield `ResolutionRequest` objects on failure.
2.  **Experiment Script**: Create `experiment_demand_driven_learning.py`.
3.  **Environment Setup**: Define the 1D priors, the `HistoricalBuffer`, and the intentional gap for "Green."
4.  **The Governor Loop**: Implement the `while not resolved` loop that catches requests, triggers the Observer, and retries.
