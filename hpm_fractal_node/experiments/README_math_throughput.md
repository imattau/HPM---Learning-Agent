# `experiment_math_throughput.py`

## Summary
Benchmarks the math Observer loop as a throughput test rather than a discovery test. It sweeps arithmetic observations across multiple sample sizes and compares full sigma storage against diagonal sigma storage while measuring observations per second, pass-to-pass slowdown, coverage, and learned-node growth.

## What It Does
- Builds the existing math world model with its protected 306-prior library.
- Runs the same arithmetic stream in two storage modes:
  - `node_use_diag=False` for full sigma storage
  - `node_use_diag=True` for diagonal sigma storage
- Sweeps multiple sample sizes and runs two passes per size so the benchmark can see warm-pass throughput as the forest grows.
- Reports:
  - total throughput
  - pass 1 throughput
  - pass 2 throughput
  - peak RSS delta
  - learned-node survival
  - category purity for learned nodes

## What It Aims To Achieve
- Measure how fast HFN can actually process observations under a real prior library.
- See whether throughput degrades as the forest gets denser.
- Compare full vs diagonal sigma storage under the same math workload.
- Provide a practical speed benchmark for the system, not just a structural one.

## Observed Results
Full run with sample sizes `50`, `100`, and `250` and `N_PASSES=2`:

- `50` samples
  - full: `24.3` obs/s total, `38.4` obs/s pass 1, `17.8` obs/s pass 2
  - diag: `22.5` obs/s total, `33.8` obs/s pass 1, `16.8` obs/s pass 2
- `100` samples
  - full: `16.4` obs/s total, `22.2` obs/s pass 1, `13.1` obs/s pass 2
  - diag: `16.7` obs/s total, `24.1` obs/s pass 1, `12.7` obs/s pass 2
- `250` samples
  - full: `9.7` obs/s total, `13.9` obs/s pass 1, `7.5` obs/s pass 2
  - diag: `10.2` obs/s total, `14.3` obs/s pass 1, `7.9` obs/s pass 2

Other full-run signals:
- coverage stayed at `100%` in every run
- learned-node counts increased with sample size: `26 -> 34 -> 84`
- pass 2 was consistently slower than pass 1, with a slowdown of about `41%` to `54%`
- diagonal sigma was not uniformly faster, but it was slightly ahead at the largest sample size
- peak RSS grew as the forest densified, especially in the full-sigma runs

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math_throughput.py
```

## Main Signals
- Observations per second across sample sizes.
- First-pass versus second-pass throughput.
- Throughput degradation as the forest grows.
- Full vs diagonal sigma storage speed difference.
- Peak RSS growth under the same math workload.

## Insights
- HFN throughput drops sharply as the same math stream becomes denser in memory.
- The second pass is much slower than the first, which suggests that growth, replay, or cache pressure materially affects speed.
- Diagonal sigma is not a guaranteed throughput win in this benchmark, but it does remain competitive and slightly ahead at the largest sample size.
- This is a useful practical benchmark because it shows the system’s cost curve, not just its accuracy or coverage.

## Issues / Limits
- The benchmark uses only the math stream, so it measures arithmetic throughput rather than general-world throughput.
- It is still a single-process benchmark, so it does not include any external controller or orchestration overhead.
- The diagonal-vs-full comparison reflects learned-node storage behavior under this workload; it is not a universal speed guarantee.
- Larger sample sizes are substantially slower, so the default sweep is intentionally kept modest to remain runnable.

## Notes
- Uses the same protected math priors as `experiment_math.py` and `experiment_sigma_diag_scaling.py`.
- Best read as a substrate benchmark: how much observation volume HFN can process as memory pressure rises.
- If you want the next step, the natural extension is to test the same benchmark under explicit cache pressure or with a second stream.
