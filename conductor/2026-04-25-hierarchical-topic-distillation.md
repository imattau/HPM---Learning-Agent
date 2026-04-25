# Hierarchical Topic Distillation — Implementation Plan

**Date:** 2026-04-25
**Task:** Combine hierarchical 3-level simulation and topic-based pattern library creation into a unified experiment.

---

## Objective

Create a standalone experiment script `hpm_ai_v4/simulations/experiment_topic_distillation.py` that automates the transition from a broad Wikipedia topic to a distilled, hierarchical pattern library.

## Key Components

1.  **Topic Fetcher:** Integrate Wikipedia API search and fetch logic (from Task 3).
2.  **3-Level Stack:** Implement the L1 -> L2 -> L3 observer stack (from the 10,000 step simulation).
3.  **Distillation Engine:** Filter all three populations for high-weight "winner" patterns.
4.  **Multi-Level Serializer:** Save winner patterns with level-specific metadata.
5.  **Automated Analytics:** Generate loss curves across all three hierarchical levels.

## Implementation Steps

### 1. Create `hpm_ai_v4/simulations/experiment_topic_distillation.py`

The script will:
- Accept `--topic`, `--steps`, and `--output`.
- Use `fetch_wikipedia_content` to build the corpus.
- Initialize `_make_population` for L1 (obs_dim=CharClassAdapter.obs_dim), L2 (obs_dim=L1.latent_dim), and L3 (obs_dim=L2.latent_dim).
- Run the `_update_level` loop, piping latent states upwards.
- Apply `compute_density` to all levels.
- Save patterns with a dynamic `level` attribute to the library.

### 2. Verification

#### Manual Run
```bash
PYTHONPATH=. python3 hpm_ai_v4/simulations/experiment_topic_distillation.py \
    --topic "General Relativity" \
    --steps 2000 \
    --output relativity_library.pkl
```

Expected:
- [wiki] Fetching content for 'General Relativity'.
- [init] Stacking 3 levels...
- [step 500] ... [step 1000] ... [step 1500] ...
- [done] L1: Distilled X winner patterns.
- [done] L2: Distilled Y winner patterns.
- [done] L3: Distilled Z winner patterns.
- [save] Multi-level library saved to: relativity_library.pkl
- [plot] Analysis saved to distill_General_Relativity.png

## Impact

- Provides a "One-Click" pipeline for training HPM hierarchies on arbitrary subjects.
- Directly supports the HPM goal of "Discovery and Distillation" of knowledge.
- Creates multi-level innate priors that can be loaded by agents for specialized reasoning.
