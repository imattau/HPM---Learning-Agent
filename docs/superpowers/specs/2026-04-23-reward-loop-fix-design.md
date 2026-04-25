# Reward Loop Fix Design Spec
Date: 2026-04-23
Branch: hpm-ai-v3-dev

## Problem Statement

Three structural defects prevent episode rewards from influencing pattern selection and parameter learning. The population replicator cannot select for task-correct patterns because fitness evaluators never receive task outcome data, the LM cannot adapt online, and the log_prob heuristic actively miscredits incorrect responses.

---

## Blocker 1: Episode reward never reaches pattern weights

### Root cause

In `base_discovery.py` `act()`, after the binary reward is computed (`reward = 1.0 if is_valid else -0.5`), the `obs` dict passed to `population.step()` contains `reward`, `outcome`, `input`, and `context_features`. The `action_pattern` that produced the result has evaluators updated via `evaluator_mgr.update_epistemic` and `evaluator_mgr.update_affective`, but **`action_pattern.accuracy` is never updated via EMA from the binary task reward**.

In `population.step()` (lines 50-90), `evaluator_mgr.update_epistemic(p, observations)` is called for ALL patterns, not only the one that acted. Because `action_pattern.accuracy` starts at `-10.0` and is never updated from task outcomes, the replicator cannot distinguish successful from unsuccessful patterns.

### Required fix

After computing `reward` in `act()` and before calling `population.step()`, apply an EMA update to `action_pattern.accuracy` and `action_pattern.loss_ema`:

```python
EMA_ALPHA = 0.1
# Clamp accuracy to [0.0, 1.0]; initial sentinel -10.0 becomes 0.0 on first update
prev_acc = max(0.0, action_pattern.accuracy)
action_pattern.accuracy = (1 - EMA_ALPHA) * prev_acc + EMA_ALPHA * max(0.0, reward)
loss_val = 0.0 if reward > 0 else 1.0
if action_pattern.loss_ema is None:
    action_pattern.loss_ema = loss_val
else:
    action_pattern.loss_ema = (1 - EMA_ALPHA) * action_pattern.loss_ema + EMA_ALPHA * loss_val
```

### Invariants

- `action_pattern.accuracy` lies in `[0.0, 1.0]` after first update.
- Initial sentinel -10.0 is clamped to 0.0 on first update via `max(0.0, ...)`.
- EMA alpha = 0.1 (matches existing stickiness dynamics).
- Only the pattern that acted gets updated here; `population.step()` updates all others via evaluators as before.

---

## Blocker 2: LanguageModelPattern.update_parameters() is a no-op

### Root cause

`neural_lm_pattern.py:250-252` explicitly documents and implements a no-op. The LM pretrain() works offline but the LM never adapts during episodes. Task-specific patterns that emerge in episode text are discarded.

### Required fix

Implement online fine-tuning when `observations` contains a `text` key and `reward > 0`. Perform one gradient step on next-token prediction, weighted by the reward signal.

```python
def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 1e-4):
    reward_tensor = observations.get("reward", None)
    if reward_tensor is None:
        return
    reward_val = float(reward_tensor.item() if hasattr(reward_tensor, 'item') else reward_tensor)
    if reward_val <= 0.0:
        return  # only learn from positive outcomes
    text = observations.get("text", None)
    if not isinstance(text, str) or len(text.strip()) == 0:
        return
    if self.model is None or self.tokenizer is None:
        return
    tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].to(self._device)
    if input_ids.shape[1] < 2:
        return
    self.model.train()
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    outputs = self.model(input_ids, labels=input_ids)
    loss = outputs.loss * reward_val
    loss.backward()
    optimizer.step()
    self.model.train(False)  # back to eval mode
    alpha = 0.1
    loss_val = loss.item()
    if self.loss_ema is None:
        self.loss_ema = loss_val
    else:
        self.loss_ema = (1 - alpha) * self.loss_ema + alpha * loss_val
```

### Invariants

- Only triggered when `reward > 0`.
- Default `learning_rate` = 1e-4 (conservative relative to offline pretrain).
- Model returns to eval mode after each step.
- If model or tokenizer is None, return silently.
- `loss_ema` updated to reflect online adaptation quality.

---

## Blocker 3: log_prob heuristic corrupts fitness for list tools

### Root cause

`neural_lm_pattern.py:238-240`: when `observations` does not contain `"reward"`, the method falls through to a heuristic that rewards any non-empty list proportional to its length (`min(1.0, len(result) * 0.1)`). A tool returning `[1.0, 2.0, 3.0]` for an answer of `42` gets `log_prob = 0.3` instead of `0.0`, inflating fitness of wrong-answer patterns.

### Required fix

Remove heuristic branches. Return `0.0` for any unverified result; only return the explicit reward when `observations["reward"]` is present:

```python
def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "reward" in observations:
        r = observations["reward"]
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(float(r), device=self._device)
        return r.to(self._device)
    return torch.tensor(0.0, device=self._device)
```

### Invariants

- No heuristic branches on result type or length.
- Absent `"reward"` key returns exactly `0.0` (neutral).
- Negative rewards (e.g. `-0.5`) are permitted and correctly penalise the pattern.
- Backward-compatible: callers already passing `observations["reward"]` are unaffected.

---

## Integration Contract

After all three fixes:

1. `act()` computes binary reward → EMA-updates `action_pattern.accuracy` + `action_pattern.loss_ema` → calls `population.step()` → replicator sees correct fitness.
2. `population.step()` calls `p.update_parameters(obs)` for all patterns → LM fine-tunes on positive-reward steps.
3. `log_prob()` returns verified reward only → clean fitness signal.

## Type Contracts

| Field | Type | Range | Owner |
|---|---|---|---|
| `HPMPattern.accuracy` | `float` | `[0.0, 1.0]` after first update | `act()` sets via EMA |
| `HPMPattern.loss_ema` | `Optional[float]` | `>= 0.0` | `act()` sets; None until first update |
| `observations["reward"]` | `torch.Tensor` shape `(1,)` | `[-1.0, 1.0]` | `act()` sets |
| `observations["text"]` | `str` | non-empty | caller sets for text tasks |
