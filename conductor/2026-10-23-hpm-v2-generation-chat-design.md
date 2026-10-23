# HPM v2 Generation and Chat: Design

**Status:** active implementation; C0–C2 partial slice implemented
**Parent:** [HPM Training Regime: First Target at 9B-Model Usability](2026-10-23-hpm-9b-usability-training-regime.md)  
**Scope:** the user-facing conversational layer of `hpm_ai_v2` for the study-assistant task boundary (grounded QA, explain/compare/define/summarize, study aids, preferences, corrections).

## 1. Problem

The training regime defines what the assistant must do but has no interface through which a user talks to it. What exists today:

| Piece | State | Limit for chat |
|---|---|---|
| `SupervisedStudyTrainer.predict` | Returns the best source sentence, offsets, confidence, or abstains | Single-turn, extractive only, no dialogue state |
| `WriterMixin.answer_natural` | Wraps `answer_question_hierarchical` in "The answer is …" | Passes through whatever the Reader returns, ungrounded, no citation, unconditional "I don't know" |
| `WriterMixin.generate_summary` | Ranks sentences by summed concept mass | No evidence check, no level control, node ids as fallback text |
| `WriterMixin.generate_sentence` | SRL → template sentence | Toy grammar; useful as a realization primitive only |
| `ReaderAgent.query` / `WriterAgent.request_knowledge` | Debug `print`s, live web acquisition | Web acquisition is explicitly out of the first target |

The gap is not "a better text generator". It is a loop that decides what to do with a turn, grounds it, says it, and learns from the reaction.

## 2. Central design decision: plan, then realize

Split every response into two stages with different owners.

1. **Content plan (HPM-native).** A `ResponsePlan`: the dialogue act chosen, an ordered list of `Claim`s, each pointing to source spans (or flagged `inference`), the requested format, and a confidence. Everything learned lives here: which evidence is relevant, which procedure applies, whether to answer or abstain.
2. **Realization (pluggable, verified).** Turns the plan into surface text. It may not add claims. A verifier checks the output against the plan before it is shown.

This keeps the HPM contribution measurable (the plan) and keeps fluency from silently supplying answers, which the parent doc forbids: *a frozen language component must not provide answers or hidden labels to the learner*.

Realizers, in build order:

- **ExtractiveRealizer**: verbatim spans with offsets. Already effectively exists. Always available as the fallback.
- **SchemaRealizer**: L4 procedure nodes (define, compare, outline, flashcards, quiz) that fill slots from claims. Slot structure is a learned/registered pattern, not free text.
- **VerbalizerRealizer (optional)**: a frozen LM (e.g. the pinned 8–10B model via the existing `OllamaChatBaseline` adapter) rewrites the claim list into fluent prose under a "use only these claims" prompt. Its config is recorded. The verifier rejects any output sentence not entailed by a plan claim and falls back to the SchemaRealizer or ExtractiveRealizer.

## 3. HPM role mapping

| HPM role | Chat implementation |
|---|---|
| Substrate | New typed nodes beside passage and `answer_sentence` nodes: `claim` (source-span-backed), `turn`, `preference`, `procedure`, `correction`. Provenance edges from every claim to spans and every turn to the claims it used. Source text and learned abstractions stay separately addressable. |
| Dynamics | Per-turn feedback (thumbs, edit, "that's wrong because …") feeds `Observer.boost_id` / `penalize_id` on the claims, passages, and procedures involved. Corrections are stored as scoped evidence, never as unconditional rewrites. Repeated successful plan shapes are proposed as procedures, promoted only after held-out success. |
| Evaluators | Grounding (every output claim traces to a span), calibration (confidence vs correctness), format adherence, correction uptake, latency, turns-to-resolution. No fluency score can promote a pattern. |
| Fields | `ConversationState`: user id, active source collection, recent turns, salient entities (for pronouns and "the second one"), active preferences. Retrieval and procedure choice are conditioned on it. Private user patterns stay separate from shareable ones. |

HPM levels in a turn: L1 spans/entities/question, L2 definitions and stable preferences, L3 cross-document relations used for compare, L4 the procedure that builds the plan, L5 the act controller below.

## 4. Turn pipeline

```
user turn
  -> Interpret      resolve references using ConversationState; detect task family
  -> Act controller (L5)  choose: answer | clarify | retrieve_more | abstain | apply_correction | set_preference
  -> Plan           retrieve evidence (study trainer ranking), pick procedure, build ResponsePlan
  -> Realize        Extractive / Schema / Verbalizer
  -> Verify         grounding check; downgrade realizer or abstain on failure
  -> Respond        text + citations + visible confidence/uncertainty
  -> Log            replayable TurnEvent (same JSONL event log style as training)
  -> Learn          later: feedback event updates weights, corrections, preferences
```

The act controller starts as a thresholded rule over retrieval confidence and plan coverage, reusing the validation-calibrated abstention threshold, and becomes a learned L5 pattern only once there are enough logged turns to hold out. Clarification is chosen only when two or more distinct evidence clusters score within a margin, since the parent doc requires asking only when expected value exceeds the interruption cost.

## 5. Core types (sketch)

```python
@dataclass(frozen=True)
class Citation:      source_id: str; start: int; end: int; text: str
@dataclass(frozen=True)
class Claim:         claim_id: str; text: str; citations: tuple[Citation, ...]; kind: Literal["source","inference"]
@dataclass(frozen=True)
class ResponsePlan:  act: str; task_family: str; claims: tuple[Claim, ...]; fmt: str; confidence: float; procedure_id: str | None
@dataclass(frozen=True)
class Reply:         turn_id: str; text: str; plan: ResponsePlan; realizer: str; verified: bool; abstained: bool
@dataclass
class ConversationState:  user_id; session_id; source_ids; turns; salient_entities; preferences
```

`ChatSession.send(user_text) -> Reply` and `ChatSession.feedback(turn_id, signal, correction=None)` are the entire public surface. `ChatSession.save()/load()` persist state and the event log so cross-session correction tests are possible.

## 6. Module layout

```
hpm_ai_v2/chat/
  schema.py       # types above, TurnEvent
  state.py        # ConversationState, reference resolution
  controller.py   # act selection (L5)
  planner.py      # evidence -> ResponsePlan; procedure lookup
  realizers.py    # Extractive, Schema, Verbalizer (all implement Realizer protocol)
  verifier.py     # claim grounding check
  session.py      # ChatSession: the loop, logging, persistence
  procedures.py   # define/compare/outline/flashcards as registered L4 patterns
hpm_ai_v2/experiments/run_chat_eval.py   # multi-turn held-out episodes vs Ollama baseline
```

`planner.py` wraps `SupervisedStudyTrainer` rather than re-implementing retrieval. `WriterAgent` is not extended: its web-acquisition and debug-print behaviour is out of the first target, and `WriterMixin.generate_sentence` is kept only as a possible SchemaRealizer primitive.

## 7. Build slices

| Slice | Deliverable | Gate |
|---|---|---|
| C0 | `schema.py`, `ChatSession` with Extractive realizer, one-turn grounded QA and abstention, event log | Matches current study-trainer results through the chat interface; every reply cites resolvable spans |
| C1 | `ConversationState`, multi-turn reference resolution, act controller with clarify/abstain | Multi-turn fixture passes; clarification rate reported |
| C2 | Procedures + SchemaRealizer: define, summarize, outline, flashcards | Each procedure validated on held-out sources before registration; claims all traced |
| C3 | Feedback, scoped corrections, preferences, save/load | Correction applied to paraphrase after restart with no change to unrelated answers |
| C4 | Optional VerbalizerRealizer + verifier | Ungrounded-claim rate not above extractive baseline; config recorded |
| C5 | Multi-turn eval runner vs pinned 9B and retrieval baselines | Per-family results, abstention quality, latency; no aggregate-only score |

C0–C2 are in the first implementation tranche; C2 currently uses fixed response schemas and extractive source sentences, not learned procedures.

## 8. Risks

- **TF-IDF-style representation.** Planning quality is bounded by `TextDomainConfig`'s vocabulary vector. Reference resolution and compare need a semantic-encoder control (parent doc item 5) before results are interpreted.
- **Verifier strength.** A lexical-overlap grounding check will pass paraphrased falsehoods. C4 must state which entailment check is used, and the check must not be the same LM that produced the text.
- **Evaluation leakage.** Multi-turn episodes must keep the held-out documents out of procedure and controller tuning, as with the single-turn fixture.
- **Scope creep toward open chat.** Anything outside the study task boundary should abstain or decline, not be answered from the verbalizer's parametric knowledge.

## 9. Decisions needed

1. **Is the optional frozen-LM verbalizer (C4) in scope for the first target?** Recommended: yes, behind the verifier, with the extractive and schema realizers as the primary measured path and the LM-realized variant reported as a separate configuration.
2. **Interface:** Python API plus a terminal REPL first (recommended); a web UI is deferred.
3. **Reference resolution:** start rule-based (last-mentioned entity/source) and measure, rather than learn it in C1.

## 10. Implementation evidence (2026-10-23)

`hpm_ai_v2/chat/` now contains the C0–C2 partial path: typed response plans and turn events, `ConversationState`, a rule-based answer/abstain controller, a planner over `SupervisedStudyTrainer`, exact-span citation verification, extractive response schemas for define/summarize/outline/flashcards, and `ChatSession.send/save/load`. Follow-up questions with simple reference words carry forward the prior resolved question. The chat layer does not train or modify HFN weights itself; supervised learning remains in the trainer. The trainer now creates source-linked fact patterns from labeled answer references and validates/promotes them only through held-out answer and provenance checks; C2's response schemas themselves remain hand-authored.

Tests cover grounded answer and abstention behavior, citation validity, event logging, follow-up resolution after session restart, and fact-pattern support/promotion/checkpoint restoration. The supervised trainer now supports correction episodes and the runner verifies checkpoint reload predictions; `ChatSession.feedback` and conversational correction capture remain unimplemented. Three of three source-linked facts were created in the latest small procedure smoke run; none promoted because its validation episodes lack answer references. The planner now emits a `clarify` act, listing the candidate sources, when the runner-up source scores within a 2% relative margin of the best; it is a fixed rule, not a learned one, and clarification is not yet resolved across turns (the user must rephrase). Learned procedure promotion, preferences, optional verbalization, and multi-turn comparison against the pinned baseline remain unimplemented.

C3 is implemented in `hpm_ai_v2/chat/feedback.py` with a browser frontend (see [HPM v2 Chat Frontend: Design](2026-10-24-hpm-v2-chat-frontend-design.md), section 14): thumbs, span-picked corrections and the `summary_format` preference are recorded in an append-only ledger replayed onto the unchanged base checkpoint. Learned correction behaviour still depends on the trainer's correction-transfer gate, which is uncommitted work.

### Abstaining without calibration (2026-10-24)

A pretrained checkpoint has no validation episodes, so its abstention threshold is 0.0 and only a question with no word overlap abstains. On a pretrained corpus of 76 design documents the chat answered "what is the capital of France?" with a sentence about querying an LLM, and offered a clarification between two irrelevant sources for "how many moons does Jupiter have?".

**A confidence floor does not fix this.** On uncalibrated TF-IDF checkpoints retrieval confidence overlaps between good and junk matches. Correct answers ranged 0.04 to 0.18 (eval sets and pretrained corpus, median about 0.13); junk answers ranged 0.035 to 0.141. A label-free null (queries of unrelated words drawn from the corpus) had a 95th percentile of about 0.21, above the median correct answer, so a floor at that level would have abstained on about 80% of correct answers. Cosine scales also differ across encoders and vocabulary sizes, so a constant would not carry over. `GroundedPlanner(min_confidence=...)` exists as an explicit knob and is off by default.

**What is implemented instead:** `GroundedPlanner` abstains (`reason: no_content_support`) when an answer's cited sentences, or a clarification's offered sources, share no content word with the question. Question words (what, how, why, ...) are removed first, since a match resting only on them says nothing about subject matter. It applies to any lexical encoder (`require_content_support="auto"`), calibrated or not; a semantic encoder can match without a shared word, so it is left out. It was first limited to uncalibrated checkpoints; the calibration measurements below showed it still adds to a calibrated threshold without costing in-scope answers, so it now stays on. It judges the final evidence, after the selector has had a chance to replace a seed that mentions none of the question's words.

**Measured.** Eval sets (96 questions): exactly one question changed per set, both unanswerable ones now abstaining; every answerable metric is identical to three decimals. Pretrained corpus (10 in-scope, 12 out-of-scope questions): in-scope answered 9 of 10 before and after; out-of-scope answered or clarified 9 of 12 before and 5 of 12 after. **Correction:** those 22 were informal probes, not labelled questions, and two of the "in-scope" ones ("summarize the usability regime", "how does the chat plan a response") are about material that is not in the pretraining corpus (that document is dated after the 60 files it was built from, and the word "chat" occurs in none of them). Do not rely on the 9 of 10; the labelled held-out sets in the next section are the evidence.

**What it cannot do.** Junk that shares one word with the corpus still gets an answer ("when did the Roman empire fall", "how many moons does Jupiter have", "what is the boiling point of mercury"), and a weakly related sentence that shares a word with an in-scope question passes ("how does the chat plan a response", where "chat" is nowhere in the corpus). Closing that gap needs real calibration from validation episodes, not a heuristic. The question lists are small and written by the same author as the check, so these counts show direction, not a rate.

### Calibrating a pretrained checkpoint (2026-10-24)

Closing the one-shared-word gap needs real calibration. The study trainer already has it: `SupervisedStudyTrainer.fit` with validation episodes sets `abstain_threshold` by choosing the confidence that gives the most correct evidence choices. What a pretrained checkpoint lacked was validation episodes. `hpm_ai_v2/chat/calibrate.py` supplies the step:

```
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.chat.calibrate CHECKPOINT \
    --validation questions.jsonl --test held_out.jsonl --output calibration.json
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.chat.server CHECKPOINT --data-dir DATA --calibration calibration.json
```

A question row is `{"id", "question", "answer_contains": "quoted text" | null}`. The quote is a snippet of the answering sentence; every source containing it is a right source, which keeps labels valid when a corpus is re-chunked. `null` marks a question the corpus cannot answer. The threshold is the trainer's own, stored in a small JSON file bound to the checkpoint's fingerprint (a calibration for another checkpoint is refused), and the server applies it when loading. The checkpoint itself is never written: saving a loaded checkpoint to a new directory would move the base's node files, and a pretrained checkpoint may be large.

**Question sets.** `hpm_ai_v2/training/eval_sets/pretrained_calibration_{validation,test}.jsonl`: 24 in-scope and 24 out-of-scope questions each, disjoint. In-scope questions are paraphrases of single sentences in the corpus. Half of the out-of-scope questions share ordinary words with the project vocabulary ("plan", "test", "system", "node", "agent", "structure"). The corpus is the first 60 documents of `conductor/` by name (`pretrained_corpus_files.txt`), prepared with `prepare_pretraining_corpus --source-type text` and pretrained with `pretrain_hfn_corpus --max-vocab 3000`: 76 sources, 4,046 sentences. One author wrote the sources' questions and the labels.

**What I compared on the held-out test set** (24 in-scope, 24 out-of-scope; validation fitted only):

| Configuration | in-scope: right source / wrong source / abstained | out-of-scope abstained | balanced |
|---|---|---|---|
| word-support check only (no calibration) | 16 / 8 / 0 | 7 of 24 | 0.479 |
| trainer calibration (confidence only), no check | 15 / 7 / 2 | 18 of 24 | 0.688 |
| **trainer calibration + word-support check (shipped)** | 15 / 7 / 2 | **19 of 24** | **0.708** |
| my own two-signal fit (confidence and an idf-weighted word-support floor) | 13 / 7 / 4 | 19 of 24 | 0.667 |

The threshold fitted was 0.0875 (validation balanced accuracy 0.583 to 0.854). The two-signal fit caught the same number of junk questions on the test set while losing two more in-scope answers, so it was dropped; the trainer's own calibration is the mechanism, with the word-support check kept alongside it. Earlier I argued that confidence alone cannot separate junk from good answers; on a larger set it separates most of them (75 to 79% of junk), with the remainder overlapping.

**What it does not do.**
- Five out-of-scope test questions are still answered or clarified (Hamlet's author, the offside rule, what a real estate agent does, when the Roman empire fell, how a thermostat works), at confidences 0.09 to 0.15, inside the range of correct answers (0.08 to 0.19).
- Nine of 24 in-scope questions get no right answer: seven are answered from a wrong source and two abstain. This is retrieval quality with a 3,000-word TF-IDF vocabulary, which calibration cannot fix.
- **Summary requests do not lose out; an earlier version of this note said they did.** It rested on "summarize the usability regime" abstaining, but that request is about a document that is not in the corpus, so abstaining was correct. A labelled test of every request type follows.
- 24 plus 24 questions per split, one author: these counts show direction, not a rate.

### Request types and per-type thresholds (2026-10-24)

**Does one threshold serve every request type?** Cosine confidence differs between request types ("summarize X" has few content words), so a threshold per type was the natural worry. I tested it on held-out requests. `pretrained_calibration_requests_{validation,test}.jsonl` hold 24 in-scope and 24 out-of-scope requests each: summarize, outline, flashcards and compare, six in-scope and six out-of-scope of each. In-scope requests are labelled by topic (`title_contains`; a list means the request needs every topic, as a comparison does) and name documents of the pretraining corpus; validation and test topics are disjoint (a test checks this). Out-of-scope requests share ordinary words with the project vocabulary ("plan", "structure", "system"), and a comparison pairs an in-scope topic with an out-of-scope one, which must still abstain.

**Result.** The shipped single threshold (0.0875, fitted on validation *questions only*) on held-out questions plus requests (48 in-scope, 48 out-of-scope), by request type as the planner classified it, in-scope right source before to after, and out-of-scope abstained before to after:

| Request type | in-scope right source | out-of-scope abstained |
|---|---|---|
| question (question and define) | 15 to 14 of 23 | 7 to 19 of 24 |
| summarize | 5 to 5 of 7 | 4 to 6 of 6 |
| outline | 6 to 6 of 6 | 2 to 6 of 6 |
| flashcards | 4 to 4 of 6 | 0 to 6 of 6 |
| compare | 5 to 5 of 6 | 0 to 5 of 6 |
| **all** | **35 to 34 of 48** (wrong source 13 to 12, abstained 0 to 2) | **13 to 42 of 48** |

**Per-type thresholds were not better, so they are not shipped.** I built them (a floor fitted for each type on its six in-scope and six out-of-scope validation requests, the trainer left at the lowest floor and the planner enforcing each type's own) and compared three calibrations on the same held-out set, all fitted on validation only:

| Calibration | in-scope right | out-of-scope abstained |
|---|---|---|
| single threshold, validation questions only (shipped) | 34 of 48 | 42 of 48 |
| one pooled threshold, questions and requests together (0.0986) | 31 of 48 | 44 of 48 |
| per-type thresholds (summarize 0.080, outline 0.159, flashcards 0.093, compare 0.075; default 0.0875) | 32 of 48 | 41 of 48 |

The per-type fit overfitted its small samples: the outline floor (0.159) cost two in-scope outlines, and the compare floor missed one more out-of-scope comparison. The differences are one to three requests, so this does not show that per-type thresholds can never help, only that at this size they did not, and the extra mechanism was not worth keeping. Sets of about 24 per kind, one author.

