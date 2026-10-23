# HPM v2 Evidence Access over the Structured Store

**Status:** round 1 failed the gate; after round 2 (fresh longer-source held-out, four failure-mode fixes, protocol fixed in advance) **the gate is met and evidence expansion is the chat default** (`ChatSession(expand_evidence=False)` turns it off). The margin on one criterion is small and process questions are still not solved; see sections 11 and 12.
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

## 9. Round 2: protocol fixed before the selector changes

The v1 held-out split has been seen and the gate failed, so it can no longer test anything. Round 2 uses a new set and a stated procedure, committed here before the selector is touched.

**New set:** `hpm_ai_v2/training/eval_sets/coverage_v2.jsonl`. Six new topics (circulatory system, DNA to protein, life cycle of stars, supply and demand, how data crosses the internet, the French Revolution), each 18–20 sentences in four paragraphs, about twice the length of v1's. 24 dev and 24 held-out questions, same type mix (6 specific, 6 process, 5 compound, 4 cross-source, 3 unanswerable per split), each labelled with a minimal support set.

**What this set does not fix.** The same person (me) wrote the sources, the questions and the labels, and wrote them knowing which failure modes v1 exposed, so question types lean towards them. Labels are one annotator's judgement. n is still 21 answerable per split. An independent annotator would be better and was not available.

**Harness change (made before any selector run).** `build_trainer` capped the TF-IDF vocabulary at 300 words, but v2's sources use 617 distinct words, so every system was blind to many questions. The cap is now the number of distinct words in the set (never below 300). v1 has 298, so its numbers are unchanged; this was checked.

**v2 dev baselines** (all answerable; recall / precision / complete of 21 / width): current chat 0.33 / 0.48 / 4 / 1.0; naive top-2 0.52 / 0.48 / 9 / 1.8; top-3 0.56 / 0.44 / 10 / 2.5; top-4 0.56 / 0.39 / 10 / 3.1. On long sources process and cross-source questions are much harder for fixed top-k than on v1 (top-3 process recall 0.28, cross 0.12). Held-out baselines are recorded in the single scoring below, not before.

**Changes planned for the selector** (one per failure mode, each a named parameter):

1. *Cap-bound growth:* structural growth gets its own budget (`max_structural`) in addition to the gain threshold, and an addition must share at least one concept with the evidence so far, so adjacency alone cannot chain.
2. *Wrong-seed amplification:* no structural growth when the seed covers too little of the question (`seed_min_share`), or when at least `unseen_ratio` of the question's content words appear in no source at all (the question is probably out of scope; return the seed alone).
3. *Over-triggering:* a question with a second clause ("… and how …") is a compound question and gets coverage growth only (`compound_blocks_process`).
4. *Seed swap:* the seed is replaced only when it mentions none of the question's words (`swap_seed_share_max = 0`), not when another sentence merely mentions more.

**Tuning:** only on data already seen: v1 dev, v1 held-out, v2 dev. A small grid declared in advance over `max_structural`, `decay`, `theta_struct`, `seed_min_share`, `link_weight`, `theta_cov`. Selection rule: keep cells where, on each of those three splits, specific-question recall is at least the current chat's, specific precision is within 0.10 of it, and the same questions abstain; among those, choose the highest mean F1 over the three splits, narrower width on ties. Then freeze and commit the parameters.

**Scoring:** the v2 held-out split is scored **once**, on a pinned export of the committed code with frozen parameters, against the same six criteria as section 4 (recall at least +0.20 over current and more complete answers; F1 above naive top-2, top-3 and top-4; no single-fact regression; abstention unchanged; summary stem coverage at least +0.10 at width at most 7; no regressions in existing tests). If the gate is met the selector becomes the chat default; otherwise it stays opt-in. Nothing is changed after that scoring.

## 10. Round 2: selector changes and tuning record (committed before the v2 held-out is scored)

The four planned fixes are in `evidence.py` as parameters (`max_structural`, `require_link`, `seed_min_share`, `unseen_ratio`, `compound_blocks_process`, `swap_seed_share_max`). Summary selection is unchanged and untuned.

Tuning used only seen data: v1 dev, v1 held-out and v2 dev (63 questions, 21 answerable each). The declared grid had 96 cells (`max_structural` 1/2/3, `decay` 0.6/0.8, `theta_struct` 0.1/0.2, `seed_min_share` 0/0.34, `link_weight` 0.5/0.7, `theta_cov` 0.25/0.35). Results:

- All 96 cells met the single-fact constraints (specific recall at least the current chat's, precision within 0.10) on all three splits, so the constraints did not bind.
- Mean F1 over the three splits ranged only from 0.62 to 0.67 (median 0.645). The grid is not a sensitive instrument: the result does not hinge on the chosen cell.
- The top cells tied exactly (mean F1 0.665, width 1.83 to 1.84). The declared rule leaves a tie on `seed_min_share` (0 versus 0.34); I added one tie-break at this point, **keep the safeguard on (0.34)**, because the guard is part of the planned fix and the data cannot separate the two. That tie-break was chosen before any v2 held-out result existed.

Frozen parameters: `max_structural 3, decay 0.6, theta_struct 0.1, seed_min_share 0.34, link_weight 0.5 (adjacency_weight 0.5), theta_cov 0.25, unseen_ratio 0.5, swap_seed_share_max 0, compound_blocks_process on, require_link on`, others as in section 5.

On the tuning splits (F1 / complete of 21): v1 dev 0.65 / 9, v1 held-out 0.68 / 10, v2 dev 0.67 / 13, against current chat 0.41, 0.62, 0.39. These are tuning-set numbers and say nothing about generalisation; the v1 held-out figure in particular is no longer held out.

Known weakness left as is, to keep the protocol clean: the structural `link` term divides by the candidate's own shared-concept weight, so it favours sentences with few concept links over richly linked ones. For "how does photosynthesis work" it picks the glucose-storage and algae sentences over the light-reaction and Calvin-cycle ones. It was not changed because it is not one of the four declared fixes.

## 11. Round 2 result: v2 held-out, scored once

Scored once on a pinned clean export of commit `0038417` (selector and parameters frozen in section 10), all systems on the same trainer. Nothing about the selector was changed after seeing these numbers.

All answerable (21), recall / precision / F1 / complete / mean width:

| System | recall | precision | F1 | complete | width |
|---|---|---|---|---|---|
| current chat (top-1) | 0.480 | 0.762 | 0.589 | 5 | 1.0 |
| naive top-2 | 0.663 | 0.619 | 0.640 | 10 | 1.9 |
| naive top-3 | 0.714 | 0.532 | 0.610 | 12 | 2.6 |
| naive top-4 | 0.714 | 0.468 | 0.566 | 12 | 3.3 |
| **expanded** | 0.687 | 0.698 | **0.692** | 10 | 1.9 |

Per type (recall / precision / complete):

| Type (n) | current | naive top-2 | naive top-3 | expanded |
|---|---|---|---|---|
| specific (6) | 0.83 / 0.83 / 5 | 1.00 / 0.67 / 6 | 1.00 / 0.58 / 6 | 1.00 / 1.00 / 6 |
| process (6) | 0.26 / 0.67 / 0 | 0.40 / 0.58 / 0 | 0.58 / 0.61 / 2 | 0.40 / 0.42 / 0 |
| compound (5) | 0.50 / 1.00 / 0 | 0.90 / 0.90 / 4 | 0.90 / 0.67 / 4 | 0.90 / 0.83 / 4 |
| cross-source (4) | 0.25 / 0.50 / 0 | 0.25 / 0.25 / 0 | 0.25 / 0.17 / 0 | 0.38 / 0.50 / 0 |

Summaries: source stem coverage 0.51 at width 5.8, against 0.19 at width 3.0 for the current chat and 0.12 to 0.19 for naive top-k.

### Gate (section 4 criteria, unchanged)

| # | Criterion | Result |
|---|---|---|
| 1 | recall at least +0.20 over current; more complete answers | +0.206 (pass, by 0.006); complete 10 vs 5 → **pass** |
| 2 | F1 above naive top-2, top-3, top-4 | 0.692 vs 0.640 / 0.610 / 0.566 → **pass** |
| 3 | single-fact questions do not regress | recall 1.00 vs 0.83, precision 1.00 vs 0.83 → **pass** |
| 4 | abstention unchanged | the same two questions abstain (F22, F23) → **pass** |
| 5 | summaries: coverage at least +0.10 at width at most 7 | +0.32 at width 5.8 → **pass** (partly circular, see round 1) |
| 6 | existing tests unaffected | chat, study and structure tests pass on the pinned commit → **pass** |

**Result: the gate is met, so by the rule fixed in advance expansion became the chat default.**

## 12. How to read that result

- **The margin on criterion 1 is 0.006.** At 21 questions, one question moving by a tenth of a point changes it. Criterion 2 has more room (0.05 over naive top-2).
- **Per question:** recall was higher than the current chat on 8 of 21 questions and lower on none. Against naive top-2, expanded did better on recall plus precision on 7 and worse on 3.
- **Process questions are not solved.** 0 of 6 complete. Recall 0.40 equals naive top-2, and naive top-3 does better (0.58, 2 complete). The aggregate gate passes because the gains come from single-fact precision, compound questions and cross-source precision. On process questions expanded is no better than returning the top two or three sentences.
- **Cross-source questions are complete for no system** (0 of 4 everywhere); expanded only improves recall slightly and precision a lot.
- **The structural link term is still the weak part** (section 10): it favours sentences with few concept links.
- **Dev versus held-out:** v2 dev (0.67 recall, 13 complete) against held-out (0.69, 10 complete) is a much smaller gap than round 1 (0.77 to 0.64), which is some evidence the fixes reduced overfitting to the tuning questions. It is one split of 21.
- **A later trainer commit moved the frozen selector slightly.** With the trainer committed afterwards, the same selector and parameters score precision 0.687 on v2 held-out, against 0.698 at the scoring commit (recall unchanged at 0.687). The naive baselines and the gate were not re-scored on that trainer.
- **Same limits as before:** one author for sources, questions and labels, and the author knew round 1's failure modes. This is a direction check, not an accuracy measurement.

### Changes after the scoring (none affects the scored numbers)

- **Default flipped** to `ExpandingPlanner`, the declared consequence of meeting the gate. `ChatSession(..., expand_evidence=False)` restores the trainer's own evidence.
- **A user's correction is returned as chosen, with no widening.** Found when the default flip made the correction test fail: coverage expansion re-added the sentence the user had just corrected against. The eval sets contain no corrections, and the dev numbers were re-run and are unchanged.
- **The eval harness names the old behaviour explicitly** (`current_chat` uses `expand_evidence=False`), so the baseline cannot silently become the new default.
- **The C3 isolation test now compares another user's answer before and after a correction**, instead of pinning an absolute answer, because expansion legitimately changed every user's baseline.

### What the repo's own tests showed

Two structure tests (`test_structure_survives_a_checkpoint_round_trip_and_keeps_growing` and `test_proposition_nodes_have_deterministic_ids_and_survive_a_checkpoint`) fail in the shared working tree with "passage has no paragraphs" after a reload. They pass on a clean export of the committed code, and `hfn/tiered_forest.py` (uncommitted, not part of this work) was modified during the session, so the cause is in that uncommitted change, not here.

## 13. Next

Process questions: the structural term needs a different relevance signal (it cannot tell the sentences that elaborate a process from sentences that merely share its vocabulary). That, a larger and independently annotated set, and a per-type gate (so an aggregate cannot hide a type that did not improve) are the next measured steps.
