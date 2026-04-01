# SP23: Sovereign Decoder — Design Specification

## 1. Overview and Rationale

The **Sovereign Decoder** (SP23) extends the multi-process architecture to the generative (top-down) side of the HPM framework. Just as "Stereo Vision" uses multiple cores to perceive complex data, "Stereo Action" uses multiple cores to recursively decompose and execute complex goals.

This experiment proves that HPM's "Variance Collapse" algorithm can be parallelized across independent specialist processes, allowing for hierarchical pipelining and cross-domain constraint resolution without a central bottleneck.

## 2. Multi-Process Generative Architecture

The system uses a 3-process cluster coordinated by a Generative Governor:

### 2.1 The Specialists (Workers)
Each worker runs in its own process and holds an independent subset of the World Model (its "Target Forest").

*   **L2_Narrative (The Planner)**: 
    *   *Domain*: High-level rules and events.
    *   *Role*: Receives a highly abstract Goal Node. It does not have concrete tokens. It expands the goal into relational sub-goals (e.g., `[Agent_Slot]`, `[Action_Slot]`).
*   **L1_Lexical (The Speaker)**: 
    *   *Domain*: Vocabulary tokens (Strings).
    *   *Role*: Resolves abstract identity nodes (e.g., `Entity_Mum`) into specific concrete tokens (e.g., "mum").
*   **L1_Motor (The Actor)**: 
    *   *Domain*: Physical coordinates (1D or 2D).
    *   *Role*: Resolves spatial/action nodes into specific numerical coordinates (simulating a robotic action).

### 2.2 The Generative Governor (The Broker)
The Governor acts as the "Broker of Collapse":
1.  Sends the initial Goal to `L2_Narrative`.
2.  Receives the expanded sub-goals.
3.  Determines which L1 specialist "owns" the manifold required to resolve each sub-goal.
4.  Dispatches the sub-goals to `L1_Lexical` and `L1_Motor` in parallel.
5.  Assembles the final mixed-modal execution script.

## 3. The "Say and Point" Task

We will test the system with a multi-modal goal: **"Name the object and point to it."**

*   **The World**: A 1D environment with a `Red_Block` at $X=2.0$ and a `Blue_Block` at $X=5.0$.
*   **The Lexicon**: The words "red", "blue", "block".
*   **The Goal Node**: `Goal_Identify(Target=Red_Block)`.
*   **The Expected Path**:
    1.  `L2_Narrative` receives the goal and expands it into: `[Say_Target_Name]`, `[Point_To_Target]`.
    2.  Governor routes `[Say_Target_Name]` to `L1_Lexical` $\rightarrow$ resolves to "red_block".
    3.  Governor routes `[Point_To_Target]` to `L1_Motor` $\rightarrow$ resolves to coordinate $2.0$.
*   **Expected Output**: A synchronized mixed-modal array: `[Token('red_block'), Coord(2.0)]`.

## 4. Evaluation Metrics

1.  **Parallel Dispatch**: Does the Governor successfully route the sub-goals to the correct L1 specialists based on the topological constraints of the nodes?
2.  **Cross-Domain Consistency**: Do both the Lexical and Motor outputs refer to the exact same underlying entity (the Red Block)?
3.  **Process Isolation**: Do the workers operate successfully using only their localized Target Forests?

## 5. Implementation Roadmap

1.  **Worker Update**: Extend `SovereignWorker` to accept a `DECODE` command that invokes `hfn.Decoder`.
2.  **Priors Construction**: Build the segregated forests (Narrative, Lexical, Motor) ensuring they share identity keys for relational linking.
3.  **Governor Loop**: Implement the recursive dispatch logic in `experiment_sovereign_decoder.py`.
