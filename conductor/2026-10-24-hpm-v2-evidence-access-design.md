# HPM v2 Evidence Access over the Structured Store

**Status:** gate and baselines recorded before any selector exists (this section is committed first on purpose). Implementation notes follow below as they are built.
**Parent:** [text storage](2026-10-24-hpm-v2-text-storage-design.md), [chat design](2026-10-23-hpm-v2-generation-chat-design.md).

## 1. Problem

Chat answers cite the trainer's top sentence (one for questions, three for summaries). Complete answers need the sentences a question actually requires: process questions need the mechanism that follows the definition, compound questions need both halves, cross-source questions need a sentence from each source. The structured store now holds the relations that make this possible: reading order (`next` edges), shared concepts (term nodes) and stem-to-sentence lookup.

## 2. Measurement (built first)

`hpm_ai_v2/training/coverage.py` and `hpm_ai_v2/experiments/run_coverage_eval.py` score a response as the set of source sentences it cites against a labelled **support set**: the minimal sentences a complete answer needs. Per question: recall, precision, and complete (recall = 1). Reported per type (specific, process, compound, cross, unanswerable) with counts and Wilson intervals for completeness, never as one aggregate. Summaries have no gold support set; they are reported descriptively as idf-weighted source-stem coverage and width.

Eval set: `hpm_ai_v2/training/eval_sets/coverage_v1.jsonl`. Six short, two-paragraph sources (8–9 sentences each); 24 dev and 24 held-out questions (6 specific, 6 process, 5 compound, 4 cross-source, 3 unanswerable per split). The held-out split is scored only with `--include-heldout`, and design and tuning use dev only.

Caveats to keep in view:
- One annotator wrote both the sources and the labels, which are a judgement of a minimal sufficient set.
- n is small (21 answerable per split). Differences of one or two questions are not evidence.
- Sources are short, which favours fixed-k baselines (top-3 of 8 sentences covers a lot by chance). Absolute numbers will not transfer to long sources; direction and the comparison to fixed-k at several widths are what this tests.
- Unanswerable questions have zero word overlap with the sources, so they only check that expansion does not change abstention (the trainer here is uncalibrated).

## 3. Baselines (before any change)

All answerable, recall / precision / complete (of 21) / mean width:

| System | dev | held-out |
|---|---|---|
| current chat (top-1) | 0.33 / 0.52 / 4 / 1.0 | 0.52 / 0.76 / 6 / 1.0 |
| naive top-3 | 0.63 / 0.51 / 9 / 2.5 | 0.75 / 0.50 / 14 / 2.5 |

(Top-2 and top-4 are in the harness output.) Per type, the current chat completes 0 of 6 process, 0 of 5 compound and 0 of 4 cross-source questions on both splits. Summaries: width 3.0, source stem coverage 0.39.

## 4. Gate (fixed now, applied once to held-out after tuning on dev)

The expanded planner becomes the chat default only if **all** hold on held-out:

1. **vs current:** all-answerable recall at least 0.20 higher and strictly more complete answers.
2. **vs fixed-k:** all-answerable F1 higher than naive top-2, top-3 and top-4. (Returning more sentences is not the contribution.)
3. **Single-fact questions do not regress:** specific recall no lower than current, precision no more than 0.10 lower.
4. **Abstention unchanged:** the same questions abstain as with the current chat.
5. **Summaries:** source stem coverage at least 0.10 higher than current at width at most 7.
6. **No regressions** in the existing chat, study, reader and structure tests.

If any fails, the selector stays opt-in and the report says which and why. Tuning is limited to a handful of named parameters and each is reported.
