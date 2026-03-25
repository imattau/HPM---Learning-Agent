---
execution_mode: unattended
auto_continue: true
created: 2026-03-23
---

# Session State: Update README + Create benchmarks/README.md

## Current Objective
1. Update root README.md benchmarks section with SP4-SP10 results
2. Create benchmarks/README.md with thorough analysis of all benchmarks

## Key Results to Include

### All benchmark results (verified):

**Early benchmarks (pre-SP):**
- Reber Grammar: single agent AUROC=0.934, 3-agent AUROC=0.955
- Structural Immunity: T_rec ≤ 100, IMMUNE
- ARC (flat multi-agent): 65.5% (+45.5% vs chance)
- Substrate Efficiency: HPM on Pareto frontier

**SP4 — Structured ARC (hierarchical encoders):**
- flat=63.2%, l1_only=63.2%, l2_only=46.5%, full=69.0% (+5.8pp over flat)
- L4_only=88.6% (+19.6pp over full), L4+L5=88.6%

**SP5 — Structured Math (algebraic transformation families):**
- flat=10.6%, l1_only=10.6%, l2_only=66.7%, l3_only=97.8%, l2_l3=96.7%, full=96.7%
- Key finding: L3 alone achieves 97.8% (decisive abstraction level for symbolic reasoning)

**SP6 — Math L4/L5:**
- l2l3=96.7%, l4_only=98.3% (+1.6pp), l4l5_full=98.3%

**SP7 — PhyRE Physics:**
- flat=22.5%, l2l3=62.5% (+40pp), l4_only=61.7%, l4l5_full=61.7%
- 4 families: Projectile, Bounce, Slide, Collision
- 240 tasks (60/family)

**SP8 — Cross-task L4 (PhyRE):**
- cross_task_l4=58.3% (ties l2l3, no gain from cross-task training)

**SP9 — Naive cross-domain (zero-padding):**
- Math+PhyRE→ARC: l2l3=80.0%, cross_domain=26.7% (NEGATIVE)
- Math+ARC→PhyRE: l2l3=58.3%, cross_domain=16.7% (NEGATIVE)
- PhyRE+ARC→Math: l2l3=100%, cross_domain=22.2% (NEGATIVE)

**SP10 — Delta alignment (Procrustes):**
- Math+PhyRE→ARC: l2l3=80.0%, delta_align=80.0% (ties!)
- Math+ARC→PhyRE: l2l3=63.3%, delta_align=63.3% (ties!)
- PhyRE+ARC→Math: l2l3=97.8%, delta_align=57.8% (partial)
- PARTIAL verdict: beats SP9 3/3, ties l2l3 2/3

## Files to Update/Create

### 1. Update /home/mattthomson/workspace/HPM---Learning-Agent/README.md
- Find the "## Benchmarks" section and expand it to include all SP4-SP10 results
- Keep the existing early benchmark entries, add new sections for SP4-SP10
- Add a "Key Findings" section summarizing architectural insights

### 2. Create /home/mattthomson/workspace/HPM---Learning-Agent/benchmarks/README.md
Write a comprehensive analysis document covering:

**Structure:**
1. Overview — what the benchmarks test and why
2. Benchmark catalogue — one section per benchmark file with:
   - What it tests
   - How it works (task format, scoring)
   - Results table
   - Findings and interpretation
3. Cross-benchmark analysis — patterns across domains
4. Architectural insights — what the results reveal about HPM
5. How to run — commands for each benchmark

**Benchmarks to cover:**
- arc_benchmark.py / multi_agent_arc.py — original ARC flat baseline
- structured_arc.py — hierarchical ARC with L1/L2/L3/L4/L5
- structured_math.py — math transformation families
- structured_math_l4l5.py — math with L4/L5
- structured_phyre.py — physics reasoning
- phyre_l4_sweep.py — L4 training pairs sweep
- phyre_cross_task_l4.py — cross-task L4 (SP8)
- phyre_cross_domain_l4.py — naive cross-domain (SP9)
- phyre_delta_alignment.py — Procrustes delta alignment (SP10)
- reber_grammar.py / multi_agent_reber_grammar.py — sequence learning
- structural_immunity.py — noise resilience

**Key analytical themes to develop:**
1. Why L3 is decisive for math (97.8% solo) but L4 is decisive for ARC (+19.6pp)
2. Why L4 helps ARC but not PhyRE: ARC's L2→L3 mapping (object anatomy → transformation rule) is tight and learnable; PhyRE's L2→L3 depends on configuration not just materials
3. Cross-domain transfer: naive zero-padding fails (SP9), relational delta alignment partially succeeds (SP10 ties l2l3 on 2/3 rotations)
4. The role of epistemic threading: L1→L2→L3 information flow
5. L5 behavior: stays at gamma=1.0 when L4 is reliable (ARC, Math), has nothing to gate when L4 doesn't help (PhyRE)

## Commit/Push
After writing both files:
```bash
git add README.md benchmarks/README.md
git commit -m "docs: update README and add benchmarks/README.md with SP4-SP10 analysis"
git push origin master
```

## Working directory
/home/mattthomson/workspace/HPM---Learning-Agent
