# SP71: Image Domain Mixin – Simple Version Plan

## Objective
Introduce image transformation learning to HPM AI using a domain-agnostic, compositional, few-shot mechanism. The agent will learn macros consisting of primitive image operations (e.g., rotate, flip, blur) from input-output image pairs and generalize to novel images without relying on deep learning. All operations are implemented natively in Python using PIL.

## Key Files & Context
- `hpm_ai_v2/domains/image_domain.py` (New): Domain configuration for images.
- `hpm_ai_v2/domains/image_codegen.py` (New): AST code generator for PIL image primitives.
- `hpm_ai_v2/domains/image_renderer.py` (New): Dedicated renderer for the image domain.
- `hpm_ai_v2/utils/oracle.py` (Modify): Update empirical oracle to support image state encoding.
- `hpm_ai_v2/experiments/experiment_sp71_image_fewshot.py` (New): Verification experiment.

## Implementation Steps

### 1. Domain Configuration
**Target:** `hpm_ai_v2/domains/image_domain.py`
- Create `ImageDomainConfig` extending `DomainConfig`.
- Define image concepts: `"ROTATE_90"`, `"ROTATE_180"`, `"ROTATE_270"`, `"FLIP_H"`, `"FLIP_V"`, `"BLUR"`, `"BRIGHTNESS_UP"`, `"BRIGHTNESS_DOWN"`, `"EDGE_DETECT"`, `"CONTRAST_UP"`.
- Set dimensions (`s_dim=20`, `m_dim=20 + len(concepts) + 20`).
- Create `get_primitive_nodes(config)` to instantiate primitive HFN nodes dynamically.

### 2. Code Generation
**Target:** `hpm_ai_v2/domains/image_codegen.py`
- Create `ImageCodeGenerator` extending `CodeGenerator` (or standalone).
- Implement `generate(concept, context)` to return Python AST representations of PIL operations (e.g., `img.rotate(-90)`, `img.filter(ImageFilter.GaussianBlur(1))`).

### 3. Image Renderer
**Target:** `hpm_ai_v2/domains/image_renderer.py`
- Create `ImageRenderer` extending `Renderer` (`hpm_ai_v2/utils/base_renderer.py`).
- Implement `render(node, func_name)` to traverse a multi-polygraph HFN (`node.inputs`) and sequence primitive operations.
- Translate extracted operations using `ImageCodeGenerator` and `ast.unparse()` to construct a single Python function containing PIL operations.

### 4. Oracle State Encoding
**Target:** `hpm_ai_v2/utils/oracle.py`
- Create `ImageOracle` extending `EmpiricalOracle` (or add a specialized method).
- Compute empirical state (20D vector) from PIL `outputs`:
  - `dim 0`: Validity (1.0 for success).
  - `dim 3`: Mean pixel brightness.
  - `dim 4`: Pixel standard deviation (contrast proxy).
  - `dim 12`: Edge presence (`1.0` if `std > 0.2`).

### 5. Experiment Script
**Target:** `hpm_ai_v2/experiments/experiment_sp71_image_fewshot.py`
- Instantiate `BaseHFNAgent` with `ImageDomainConfig`, `ImageRenderer`, and `ImageOracle`.
- Populate `agent._candidate_ops` with dynamically generated primitives.
- Execute a one-shot learning task:
  - Input/Output: Rotate a "digit 3" image by -90 degrees.
  - Observe input, then call `agent.solve()`.
  - Validate macro generalization by applying it to a novel "digit 5" image.
- Assert success (`SSIM > 0.9` or exact pixel match).

## Verification
- The new `ImageRenderer` prevents any modification or disruption to the existing `ASTRenderer` and list-processing tasks.
- `experiment_sp71_image_fewshot.py` executes successfully, finding a compositional macro via BFS and accurately predicting the transformed novel digit.