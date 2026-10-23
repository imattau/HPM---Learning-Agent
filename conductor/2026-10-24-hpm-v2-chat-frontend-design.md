# HPM v2 Chat Frontend: Design

**Status:** W0 and W1 implemented and checked in the built-in browser; W2 (SSE stage progress, edit button, search) and W3 (feedback, after C3) not built
**Parent:** [HPM v2 Generation and Chat: Design](2026-10-23-hpm-v2-generation-chat-design.md). This supersedes section 9, decision 2 ("terminal REPL first, web UI deferred"): the frontend is a modern LLM-style chat app. A terminal client is no longer the first deliverable.

## 1. Goal

Talking to the HPM study assistant should feel like using any current LLM chat product: a conversation list, a message thread, a composer, fast feedback, readable formatting. It should also show what those products hide: where each statement came from, and when the system declines to answer.

The frontend is a thin presentation layer over `ChatSession`. It adds no planning, retrieval or grounding logic. Anything it shows is what the tested API returned.

Non-goals for this tranche: adding or uploading sources, accounts and multi-user hosting, token streaming from an LM (there is none), voice, image input, mobile apps, learning from feedback (C3).

## 2. What users expect, and how each maps to HPM

| Familiar feature | In this app | Notes |
|---|---|---|
| Sidebar of past chats, "New chat", rename, delete, search | Same. One `ChatSession` per conversation, saved server-side | Auto-title from the first question |
| Message thread, user right / assistant left, markdown | Same | Rendered from structured blocks, not by parsing text (section 5) |
| Composer: Enter sends, Shift+Enter newline, auto-growing | Same | Disabled while a turn is in flight |
| Streaming text | Real stage progress ("Searching sources", "Planning", "Checking citations"), then the reply | Replies are computed whole. Typing text out character by character would fake a generation process that does not exist, so it is off by default |
| Suggested prompts on an empty chat | Generated from the loaded sources: "Summarize *Solar energy*", "Flashcards for *Plants*", "What is …" | Teaches the supported task families |
| Citations / sources | Numbered chips `[1]` on each claim. Click opens a source panel scrolled to the cited span, highlighted | The differentiator. Spans are verified offsets, so the highlight is exact |
| Thumbs up/down, correct, copy | Copy now. Thumbs and "correct this" appear only once `ChatSession.feedback` exists (C3). No dead buttons | |
| Regenerate | Omitted. Planning is deterministic for a given state, so regenerating returns the same reply | "Edit" on the last user message replaces it and re-runs |
| Model picker | Replaced by a read-only "configuration" line: checkpoint, representation (lexical or encoder name), realizer | Matters for interpreting results (parent doc risk on TF-IDF) |
| "I can't help with that" | Abstain card: "I couldn't find this in your sources", with the sources searched | Not an apology wall. Abstaining is a measured behaviour |
| Clarifying question | Reply with clickable option chips, one per candidate source | Clicking resolves the ambiguity (section 6) |
| Dark mode, responsive, keyboard | Follows OS theme, works at phone width, focus order and ARIA live region for new replies | |

## 3. Architecture

```
browser (static SPA)  <--JSON/HTTP-->  hpm_ai_v2/chat/server.py  -->  ChatService  -->  ChatSession  -->  trainer
```

- **Server:** Python standard library `http.server.ThreadingHTTPServer`. No new dependencies: FastAPI, Flask and Starlette are not installed, and nothing else in this repo needs them. If streaming or auth ever needs a framework, only `server.py` changes.
- **ChatService** (`hpm_ai_v2/chat/service.py`): framework-free. Owns the loaded trainer, the conversation store, and a lock around `send` because the trainer is shared and not thread-safe. Everything testable lives here.
- **Frontend:** one static `index.html`, one JS file, one CSS file under `hpm_ai_v2/chat/web/`. Vanilla JS and no build step. No CDN calls, so it works offline and the repo has no Node toolchain to maintain.
- **Run:** `python -m hpm_ai_v2.chat.server <checkpoint> --data-dir DIR [--user NAME] [--port 8765]`. Binds `127.0.0.1` only.

Loading mirrors the trained-checkpoint path (`ReaderAgent.load_agent` then `load_checkpoint_state`). The checkpoint already contains the sources, so no curriculum is needed. The server never trains and never writes into the checkpoint.

## 4. HTTP API

| Method and path | Purpose |
|---|---|
| `GET /api/config` | Checkpoint, representation, realizer name, user id, source count |
| `GET /api/sources` / `GET /api/sources/{id}` | List, or full text for the source panel |
| `GET /api/conversations` | `[{id, title, updated_at}]`, newest first |
| `POST /api/conversations` | Create, returns id |
| `GET /api/conversations/{id}` | Messages with blocks, for reload |
| `PATCH /api/conversations/{id}` | Rename |
| `DELETE /api/conversations/{id}` | Delete (confirm in the UI) |
| `POST /api/conversations/{id}/messages` | `{text, scope_source_id?}` returns the assistant message |
| `POST /api/conversations/{id}/messages/{turn}/edit` | Truncate from that turn and re-send |
| `GET /api/conversations/{id}/events` | Server-sent events for stage progress during a send (optional, phase W2) |

JSON errors use `{error, message}` with a status code. The browser never sees a Python traceback.

## 5. Presentation layer: structured, not parsed

`Reply.text` is built by `SchemaRealizer` for a terminal. It includes "Source: … characters a–b" lines, which a chat UI should render as citation chips instead. Parsing that text back out would be fragile, so a pure function builds blocks from the plan:

```
to_blocks(plan) -> list[Block]
  Block = paragraph | bullet_list | numbered_list | flashcard | clarify | abstain
  each text item carries claim_id and citation refs [{n, source_id, start, end}]
```

- It lives in `hpm_ai_v2/chat/presentation.py` and reads only `ResponsePlan` (`act`, `fmt`, `format_style`, `claims`, `options`). It adds no words. Block text is exactly claim text, plus fixed labels ("Question", "Answer", "Sources searched").
- `Reply.text` is still returned and logged unchanged, as the plain-text fallback and for the event log.
- The same function backs a future terminal client, which keeps one source of truth for layout.

Assistant message JSON:

```json
{
  "turn_id": "…", "role": "assistant", "act": "answer", "task_family": "define",
  "confidence": 0.81, "verified": true,
  "blocks": [ {"type": "paragraph", "text": "…", "citations": [1]} ],
  "citations": [ {"n": 1, "source_id": "plants", "title": "Plants", "start": 0, "end": 57, "text": "…"} ],
  "why": { "reason": "supported_by_source", "procedure_id": null, "effective_question": "…" },
  "text": "plain-text reply"
}
```

All text from sources is untrusted. The client inserts it with `textContent` and a small whitelisted renderer. No `innerHTML` with source text. Markdown beyond lists and emphasis is not rendered.

## 6. Clarification that works

Today a clarify reply ends the turn and the user must rephrase. In a chat UI, option chips that do nothing would be a bug. So this tranche adds one backend change:

- `ChatSession.send(text, *, scope_source_id=None)`. When set, the planner restricts evidence ranking to that source (filter `ranked_evidence`, skip the ambiguity check) and records it in the event log.
- Clicking a chip calls `send` with the original question and the chosen `source_id`. In the thread it appears as a small user turn: "Using *Plants*".
- This is a hand-authored rule, not learned, and the doc says so. Resolving an answer typed as free text ("the second one") stays unsolved. The chip is the supported path.

This touches `session.py` and `planner.py`, where another writer has been active. It is done as a small, separate commit before the frontend depends on it, and the existing 20 tests must still pass.

## 7. Conversation persistence

- Store: one JSON file per conversation under `--data-dir/conversations/`, written atomically (temp file then rename). Contents: `ChatSession.save()` state plus the rendered messages, so reload does not re-run planning.
- The event log (`--data-dir/events.jsonl`) is shared and append-only, as today.
- Messages keep their `turn_id`, which is what C3 feedback will attach to. No schema change is needed later.
- Deleting a conversation removes its file. Its rows in the event log stay, because the log is the evaluation record. The UI says so in the delete dialog.

## 8. Honesty rules for the interface

These constrain the design more than the visuals do.

1. A reply with `verified == false` never shows as a normal answer. It is downgraded to abstain by the session already, and the UI displays whatever act it receives.
2. Confidence is a quiet label (for example "Confidence 0.81"), not a green tick. Its meaning ("calibrated against validation episodes") is in a tooltip.
3. Every claim has at least one citation chip, and a claim without one cannot be rendered. The renderer refuses it and shows an error block, since that state means the verifier was bypassed.
4. The configuration line is always visible, including the fact that a lexical TF-IDF representation is in use when it is.
5. No avatar, name or tone suggests general intelligence. The assistant is labelled "HPM study assistant". Out-of-scope questions get the abstain card, not an improvised answer.

## 9. Tests

Backend, in `tests/test_hpm_ai_v2_chat_service.py`, no browser needed:

1. `to_blocks` for answer, abstain, clarify, summarize (bullets), outline (numbered) and flashcards. Block text equals claim text, and every citation ref resolves in the source.
2. `ChatService` create, send, reload: messages round-trip and a follow-up after reload resolves its reference.
3. `scope_source_id` restricts evidence to the chosen source and logs it. Unscoped behaviour is unchanged.
4. Two concurrent sends to different conversations do not corrupt state (the lock).
5. HTTP layer on an ephemeral port using `httpx`, which is already installed: config, sources, conversation CRUD, send, error shapes, and refusal of non-localhost `Host`.
6. Edit-and-resend truncates later turns and keeps earlier ones.

Frontend: a smoke script drives the built-in browser against the running server: load, send, open a citation, see the highlighted span, answer a clarify chip, switch conversations, check dark mode and 375px width. It is a manual checklist in the doc, not CI.

## 10. Security

- Bind `127.0.0.1`. Reject requests whose `Host` is not localhost, and require `Origin` to match on writes, which blocks DNS rebinding and cross-site POSTs to the local port.
- No authentication in this tranche, which is acceptable only because of the bind. Exposing it beyond localhost is out of scope and needs auth first.
- Source text and user text are never executed or inserted as HTML.
- The data directory holds user text. The start banner prints its path.

## 11. Build slices

| Slice | Deliverable | Gate |
|---|---|---|
| W0 | `scope_source_id` in `send`/planner (own commit), `presentation.to_blocks`, `ChatService` with conversation store | Tests 1–3, 6; existing 20 tests still pass |
| W1 | `server.py` with the JSON API, static frontend: thread, composer, sidebar, citation chips and source panel, abstain and clarify rendering, config line | Test 5; manual browser checklist passes |
| W2 | SSE stage progress, suggested prompts, edit-last-message, search, dark mode and mobile polish | Checklist at 375px and in both themes |
| W3 (after C3) | Thumbs, "correct this", preferences panel wired to `ChatSession.feedback` | C3 gate |

## 12. Risks and open points

- **Looks more capable than it is.** A polished UI invites open-ended questions. The abstain card and configuration line are the counterweight, and the blinded-usefulness runner should be pointed at this UI's transcripts, not only API calls.
- **Single shared trainer.** Serialising `send` is fine for one local user and wrong for hosting. Hosting needs per-user trainer state and a real server, which is a later design.
- **Clarification stays rule-based.** Scoped resend works for source ambiguity only. Other ambiguity (which task, which entity) has no UI path yet.
- **Edit semantics.** Truncating later turns also truncates the state used for reference resolution. This is intentional, but the event log keeps the discarded turns, so evaluation code must not treat every logged turn as part of a live conversation. Edited-away turns get an `edited: true` marker in a follow-up event.
- **Another active writer** in `hpm_ai_v2/chat/`. Everything new is a new module except W0's small `send`/planner change.

## 13. Decisions (confirmed 2026-10-24)

1. **Stack:** stdlib server and vanilla JS, no build step.
2. **Text reveal animation:** included as a user setting ("Animate replies") in the UI, stored in the browser. Replies are still computed whole; the animation is cosmetic and the setting says so. Default on, since users expect it, with the stage progress shown either way.
3. **`scope_source_id` change to `send`/planner:** accepted and committed first (`4200c82`).
4. **Terminal client:** dropped from this tranche. `ChatService` and `to_blocks` keep one later possible.

## 14. Implementation evidence

W0: `ChatSession.send(text, scope_source_id=...)` restricts ranking to one source and is logged on the turn event; `hpm_ai_v2/chat/presentation.py` builds blocks for define, summarize, outline, flashcards, compare, abstain and clarify, and refuses a claim with no citation; `hpm_ai_v2/chat/service.py` stores conversations as atomic JSON files, serialises sends behind a lock, reloads follow-up state, supports rename, delete and edit-and-resend (dropped turns are logged), and rejects non-hex conversation ids. `tests/test_hpm_ai_v2_chat_service.py` has 7 tests.

W1: `hpm_ai_v2/chat/server.py` (stdlib `ThreadingHTTPServer`, localhost only, Host and Origin checks, JSON content-type required on writes, CSP, 64 KB body cap, no tracebacks to the client) and the static app in `hpm_ai_v2/chat/web/` (vanilla JS, `textContent` only). `tests/test_hpm_ai_v2_chat_server.py` has 6 tests over real HTTP. Checked by hand in the built-in browser: ask, citation chip opens the source with the exact span highlighted, clarify chips resolve through a scoped resend, abstain card, flashcards, rename and delete, dark and light themes, 375px width (bottom-sheet source panel, no horizontal scroll), no console errors. Not yet built: SSE stage progress (the typing line is static), an edit button on the last user message (the API exists), conversation search.

To try it:

```
python -m hpm_ai_v2.chat.demo_checkpoint /tmp/hpm-demo
python -m hpm_ai_v2.chat.server /tmp/hpm-demo/checkpoint --data-dir /tmp/hpm-demo/data --port 8765
```

For real use, pass a checkpoint written by `run_supervised_study.py --output-dir`.
