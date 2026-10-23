# HPM v2 Evidence Access over the Structured Store

**Status:** selector built (`hpm_ai_v2/chat/evidence.py`, opt-in via `ExpandingPlanner`). **It did not pass the gate on held-out, so it is not the chat default.** Sections 1–4 were committed before the selector existed; sections 5–7 are results.
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

## 5. The selector (what was built)

`EvidenceSelector` grows the trainer's chosen sentence (the seed) into an evidence set; `ExpandingPlanner` wraps `GroundedPlanner` and replaces the claims. Rules are fixed and hand-set (`AccessParams`), not learned:

- **Coverage.** Query content stems the set does not mention yet are looked up through the stem map (`release` finds `released`), across sources; the sentence covering most new query weight (idf) is added while it covers at least `theta_cov` of the total. A query word counts only if a *relevant* source says it, where relevant means the source ranks within `source_ratio` of the best, so a stray "work" in an unrelated source does nothing.
- **First mention.** "What is X" and "how does X work" start at the sentence where X is introduced.
- **Structure (process questions only).** Reading-order neighbours (forward weighted over backward) and sentences sharing concepts with the set, with gain falling by `decay` per addition, stopped below `theta_struct`.
- **Summaries, outlines, flashcards.** Greedy idf-weighted coverage of the source's own stems; between 3 and 6 sentences.
- Unchanged: abstain, clarify and compare plans; a user's correction as the seed; scope to one source; legacy (version 1) storage falls back to the trainer's own evidence.

Tuning: 64 cells over six parameters, on dev only; the declared rule (best dev F1, narrower on ties) chose `decay 0.8, theta_struct 0.1, link_weight 0.7, source_ratio 0.5, theta_cov 0.35`. Two of those sit at the grid's edge, and the top cells were within about 0.03 F1 of each other, which at 21 questions is noise.

## 6. Results

All answerable, recall / precision / complete (of 21) / mean width. Held-out was scored on a pinned clean export of the committed code with the parameters frozen above; a later re-run on the commit that added relation nodes gave identical numbers.

| System | dev | held-out |
|---|---|---|
| current chat (top-1) | 0.33 / 0.52 / 4 / 1.0 | 0.52 / 0.76 / 6 / 1.0 |
| naive top-2 | 0.57 / 0.57 / 7 / 1.9 | 0.75 / 0.67 / 14 / 1.8 |
| naive top-3 | 0.63 / 0.51 / 9 / 2.5 | 0.75 / 0.50 / 14 / 2.5 |
| naive top-4 | 0.68 / 0.48 / 10 / 3.1 | 0.75 / 0.44 / 14 / 3.0 |
| **expanded** | 0.77 / 0.65 / 13 / 3.0 | 0.64 / 0.58 / 9 / 2.8 |

Held-out per type, expanded versus current and naive top-2 (recall / precision / complete / width):

| Type (n) | current | naive top-2 | expanded |
|---|---|---|---|
| specific (6) | 1.00 / 1.00 / 6 / 1.0 | 1.00 / 0.67 / 6 / 1.7 | 0.83 / 0.83 / 5 / 1.0 |
| process (6) | 0.25 / 0.50 / 0 / 1.0 | 0.56 / 0.58 / 3 / 1.7 | 0.50 / 0.42 / 1 / 4.0 |
| compound (5) | 0.50 / 1.00 / 0 / 1.0 | 1.00 / 1.00 / 5 / 2.0 | 0.80 / 0.77 / 3 / 2.6 |
| cross-source (4) | 0.25 / 0.50 / 0 / 1.0 | 0.38 / 0.38 / 0 / 2.0 | 0.38 / 0.21 / 0 / 3.8 |

Summaries: source stem coverage 0.81 at width 6.0, against 0.39 at width 3.0 for the current chat and 0.25 to 0.35 for naive top-k. This is the one place the metric is partly circular, because the selector maximises the same stem coverage the metric reports.

## 7. Gate verdict (held-out, fixed in advance)

| # | Criterion | Result |
|---|---|---|
| 1 | recall at least +0.20 over current, more complete answers | recall +0.12 (fail); complete 9 vs 6 (pass) → **fail** |
| 2 | F1 above naive top-2, top-3 and top-4 | 0.61 vs 0.71 / 0.60 / 0.55: beats top-3 by 0.01 and top-4, **not top-2 → fail** |
| 3 | single-fact questions do not regress | recall 0.83 vs 1.00, precision 0.83 vs 1.00 → **fail** |
| 4 | abstention unchanged | same questions abstain (expansion only runs on answers) → pass |
| 5 | summaries: coverage +0.10 at width at most 7 | +0.42 at width 6 → pass (circular, see above) |
| 6 | existing tests unaffected | the planner is opt-in, so existing behaviour is unchanged → pass |

**Result: the gate is not met. The selector stays opt-in and `ChatSession` still defaults to `GroundedPlanner`.**

What the held-out failures say (looked at after the verdict, not used to change anything):

- **The dev gain did not transfer.** Dev recall 0.77 and process complete 4 of 6 became 0.64 and 1 of 6. The grid chose a loose regime that suited the six dev process questions.
- **Structural growth is mostly cap-bound.** With `decay 0.8` and `theta_struct 0.1` the gain rarely falls below the threshold, so process questions grow to the cap of 6 (H7, H8, H13, H18, H20). Width by marginal gain operates for coverage and for summaries but not, as tuned, for structure.
- **A wrong seed is amplified.** H9 ("how does rainfall form") seeded in the wrong source and grew six sentences around it.
- **Over-triggering.** A compound question containing "how do vaccines use them" took the process path.
- **The seed swap can hurt.** On H1 it replaced a correct sentence with a wrong one.
- **Short sources favour fixed-k.** Top-2 of 8 sentences completes most compound questions by chance, and it wins held-out F1. These sources will not stay that short, so this comparison is the least transferable part of the result.

## 8. What would change the verdict

None of these were tried, because the held-out split has now been seen and any fix needs a fresh one: a steeper decay or a structural cap tied to the question's own stems; restricting process growth to questions whose topic sits in one source; letting the seed swap fire only on a large margin; and a seed-confidence check before expanding at all. The next evaluation needs a new held-out set, longer sources, and annotation by someone other than the author of the selector.
