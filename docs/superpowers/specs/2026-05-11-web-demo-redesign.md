# HPM v6 Web Demo Redesign

**Date:** 2026-05-11
**Scope:** Rework `hpm_ai_v6/web/web_demo.py` HTML_TEMPLATE — visual overhaul + intuitive workflow layout. All Python routes/logic unchanged.

## Goals

- Dark-tech aesthetic (research terminal feel)
- Workflow-ordered layout: Train → Generate → Reason
- AJAX for Generate and Reason (no full page reload)
- Live Gutenberg cycle status ribbon

## Approach

Rework-in-place: replace `HTML_TEMPLATE` string only. No new files, no Flask config changes.

## Visual Language

**Palette:**
- Background: `#0d0f14`
- Panel: `#131720`
- Border: `#1e2435`
- Accent: `#00e5a0`
- Text primary: `#e2e8f0`
- Text muted: `#6b7a99`
- Error: `#ff5c5c`

**Typography:**
- Monospace: `JetBrains Mono` (Google Fonts) — labels, badges, output, step numbers
- Sans-serif: `Inter` (Google Fonts) — body, descriptions

## Layout Structure

```
┌─────────────────────────────────────────────┐
│  HEADER BAR  [HPM v6]          [corpus] [●] │  sticky
├─────────────────────────────────────────────┤
│  STATUS RIBBON  ● RUNNING | book 84 | ...   │  hidden when idle
├─────────────────────────────────────────────┤
│  01 — TRAIN                             ▼   │  accordion header
│  ┌─────────────────────────────────────┐    │
│  │  [Web Links] [Gutenberg]  pill tabs │    │
│  │  form fields...                     │    │
│  │  [Train] button                     │    │
│  │  inline success/error card          │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│  02 — GENERATE                          ▼   │
│  ┌─────────────────────────────────────┐    │
│  │  seed input                         │    │
│  │  max tokens slider                  │    │
│  │  [Generate ▶] → spinner → output   │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│  03 — REASON                            ▼   │
│  ┌─────────────────────────────────────┐    │
│  │  question textarea                  │    │
│  │  [Reason ▶] → spinner → output     │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Components

### Header bar (sticky)
- Left: `HPM v6` wordmark in monospace
- Right: corpus badge + reader status dot (green = ready, grey = loading)

### Status ribbon
- Polls `/api/gutenberg_cycle_status` every 4s
- Format: `● RUNNING | book {id} | chapter {n} | {phase} | {n} books processed`
- `display:none` when `active === false`

### Accordion sections
- Clickable header bar: step number (accent, monospace) + title + chevron + status badge
- Body animates open/close via `max-height` CSS transition
- `localStorage` persists open/closed state per section key

### Train section (01)
- Two pill toggles: "Web Links" / "Gutenberg" — show/hide sub-forms
- Web Links sub-form: URLs textarea, RSS URL, top_k, min_score, POST submit
- Gutenberg sub-form: book IDs, top_k, min_score, POST submit + curated cycle controls
- Curated cycle: "Run Curated Cycle" (accent) + "Stop" (danger outline) buttons — POST
- Result card: inline below form, dismissible, green (success) or red (error)

### Generate section (02)
- Seed text input
- Max tokens range slider (1–200, value displayed live)
- Generate button: on click → `fetch('/api/generate', POST JSON)` → spinner → result card
- Result card: monospace output, accent left border, copy button

### Reason section (03)
- Question textarea
- Reason button: same AJAX pattern as Generate → `/api/reason`
- Result card: same style as Generate output

### Error handling
- AJAX errors: dismissible red banner below the relevant section
- POST errors: existing inline panel pattern retained

## Interaction Model

| Action | Method | Notes |
|--------|--------|-------|
| Generate | AJAX POST `/api/generate` | No page reload |
| Reason | AJAX POST `/api/reason` | No page reload |
| Train links | Form POST | Side-effectful, keep as POST |
| Train Gutenberg | Form POST | Side-effectful, keep as POST |
| Start/Stop cycle | Form POST | Side-effectful, keep as POST |
| Cycle status | AJAX GET poll | Every 4s, updates ribbon |

## Files Changed

- `hpm_ai_v6/web/web_demo.py` — `HTML_TEMPLATE` constant only

## Files Unchanged

- All Python routes, logic, agents, evaluators
