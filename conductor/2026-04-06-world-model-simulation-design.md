# SP38: Experiment 14 — World Model Simulation (Imagination Test)

## 1. Overview and Rationale
The **World Model Simulation** experiment evaluates whether the HFN system can act as a true generative world model. A robust world model shouldn't just explain individual observations or perform single-step planning; it should be capable of "imagination"—simulating the future evolution of an environment over multiple time steps without requiring continuous external inputs.

By training the system on sequential state transitions (`state_t → state_t+1`) and then asking it to "dream" forward iteratively, we test the temporal coherence and stability of its learned structural representations.

## 2. Setup & Execution
- **Curriculum Design (Temporal Dynamics):**
  - The system is trained on sequences of states that follow a predictable trajectory (e.g., an object moving across a grid, or a vector rotating through a latent space).
  - The `Observer` is fed these `state_t → state_t+1` pairs. The manifold explicitly encodes `[Input_State, Output_State]`.
- **The Imagination Loop:**
  - After training, the external data stream is cut off.
  - The system is given an initial starting state `S_0`.
  - **Loop (k steps):**
    1. The `GoalConditionedRetriever` or `ContextualRetriever` finds the most likely transition rule for the current state.
    2. The `Decoder` synthesizes the predicted next state `S_{t+1}`.
    3. The generated `S_{t+1}` is fed back into the system as the new `S_t` for the next iteration.
- **Observation:** The sequence of "imagined" states is recorded and compared against the true theoretical trajectory.

## 3. Evaluation Metrics
1. **Simulation Coherence:** Does the system accurately predict the next logical state based on the underlying rules of the training environment?
2. **Drift vs. Stability:** Over `k` steps, does the continuous decoding and re-encoding cause the internal state to degrade into noise (drift), or does it maintain a stable trajectory?
3. **Horizon Length:** How many steps into the future can the system accurately simulate before the accumulated error causes the trajectory to collapse?

## 4. Why This Matters
*Can HFN act as a generative world model?*
To be useful for long-term planning (e.g., Monte Carlo Tree Search or reinforcement learning), an agent must be able to roll out possible futures in its "head." If the system's generated states drift too quickly, the imagination is useless for planning. Proving that HFN can sustain stable, multi-step generative loops validates its utility as a foundational world model for autonomous agents.

## 5. Implementation Roadmap
1. **Trajectory Generator:** Create a synthetic dataset of continuous or discrete sequences (e.g., a simple 1D or 2D physics simulation like a bouncing ball or rotating vector).
2. **Temporal Encoding:** Map the `t` and `t+1` states into the standard HFN manifold.
3. **Simulation Engine:** Implement the iterative `decode -> re-encode` loop in the experiment script.
4. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_world_model_simulation.py` to run the training and imagination phases, outputting the predicted vs. actual trajectories to measure drift.
