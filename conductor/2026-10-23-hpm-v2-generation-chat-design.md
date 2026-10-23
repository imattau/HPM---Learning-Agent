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
