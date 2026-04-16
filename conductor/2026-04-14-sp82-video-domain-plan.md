# SP82 Video Domain Few-Shot Learning Plan

## Objective
Implement and validate that the HPM agent can learn video transformations (e.g., rotate, flip, brightness, composition) from a few examples (k=1, 2, 3, 5).

## Background
Following the success of SP80 and SP81 in symbolic lists, SP82 extends the framework to the video domain, processing short grayscale clips (64x64, 8 frames).

## Implementation Steps
1. **Domain Configuration (`hpm_ai_v2/domains/video_domain.py`)**:
   - Define `VideoDomainConfig` inheriting from `DomainConfig`.
   - Specify structural dimensions and primitive concepts for video.
   - Implement `get_video_primitive_nodes` to return HFN nodes for the primitives.

2. **Renderer (`hpm_ai_v2/domains/video_renderer.py`)**:
   - Create `VideoRenderer` inheriting from `Renderer`.
   - Use `cv2` (opencv-python) to generate executable code that transforms frames.
   - Support `FOR_EACH_FRAME`, `FRAME_APPEND`, `ROTATE_90`, `FLIP_H`, `FLIP_V`, `BRIGHTNESS_UP`, `BRIGHTNESS_DOWN`, `COND_BRIGHTNESS_HIGH`, `RETURN`.

3. **Oracle (`hpm_ai_v2/utils/oracle/video_oracle.py`)**:
   - Create `VideoOracle` inheriting from `BaseOracle`.
   - Encode video features: valid flag, mean brightness, std dev (contrast proxy).
   - Encode structural flags: loops, appends, ifs, assignments.
   - Update `__init__.py` to export it.

4. **Experiment Script (`hpm_ai_v2/experiments/experiment_sp82_video_fewshot.py`)**:
   - Implement synthetic video generation (e.g., a moving square).
   - Define tasks: `rotate_video_90`, `flip_video_h`, `brightness_up_video`, `compose_rotate_then_flip`.
   - Test using `BaseHFNAgent` and `HybridRetriever`.
   - Output benchmark results.

## Verification
- Install `opencv-python` if missing.
- Run `experiment_sp82_video_fewshot.py`.
- Verify success criteria (100% accuracy for most tasks at k>=2, compositional tasks handled).