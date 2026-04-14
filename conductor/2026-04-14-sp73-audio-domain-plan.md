# SP73: Audio Domain Few-Shot Learning Plan

## Objective
Demonstrate that HPM can learn an audio transformation (e.g., pitch shift up by a semitone) from a single example and generalize to a novel audio input (different melody), building upon the architecture established in SP71.

## Scope & Impact
- **New Domain:** Adds 1D audio processing capabilities to `hpm_ai_v2`.
- **Dependency:** Introduces `librosa` for audio processing and spectral analysis.
- **Components:** Introduces `AudioDomainConfig`, `AudioRenderer`, and `AudioOracle`.

## Proposed Solution
Following the successful SP71 implementation pattern:

### 1. `hpm_ai_v2/domains/audio_domain.py`
- Create `AudioDomainConfig(DomainConfig)` with an `m_dim` combining state, concepts, and delta. Concepts include `PITCH_UP_2`, `PITCH_DOWN_2`, `VOLUME_UP`, etc.
- Create `get_audio_primitive_nodes(config)` to generate HFN primitives.

### 2. `hpm_ai_v2/domains/audio_renderer.py`
- Create `AudioRenderer(Renderer)` to translate concepts into executable `librosa` code statements.
- Ensure `render(node)` outputs a code block taking `inp` and assigning `res`, matching the execution environment fixed in SP71.

### 3. `hpm_ai_v2/utils/oracle.py`
- Add `AudioOracle(EmpiricalOracle)` to extract spectral centroid and RMS volume using `librosa`, encoding them into the 20D state vector.
- Ensure the class includes `call_count` initialization to maintain compatibility with `counting_oracle`.

### 4. `hpm_ai_v2/experiments/experiment_sp73_audio_fewshot.py`
- Create a script generating sine wave melodies.
- Phase 1 (One-Shot): Train the agent to shift pitch up by 2 semitones using a simple C-D-E-F melody.
- Phase 2 (Generalization): Test on a novel G-A-B-C melody to verify the learned macro applies correctly and is fetched via the `exact` strategy.

## Implementation Steps
- [ ] Add `librosa>=0.10.0` to `requirements.txt` and install via `uv pip install`.
- [ ] Implement `hpm_ai_v2/domains/audio_domain.py`.
- [ ] Implement `hpm_ai_v2/domains/audio_renderer.py`.
- [ ] Append `AudioOracle` to `hpm_ai_v2/utils/oracle.py`.
- [ ] Create `hpm_ai_v2/experiments/experiment_sp73_audio_fewshot.py`.
- [ ] Run experiment and verify expected correlation (>0.95).

## Verification
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp73_audio_fewshot.py`.
- Assert successful execution, macro storage via `register_pattern()`, and correct code generation (`y = librosa.effects.pitch_shift(...)`).
