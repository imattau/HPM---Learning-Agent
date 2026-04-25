# LanguageModelPattern Design
Date: 2026-04-22

## Purpose
Add a self-supervised neural linguistic substrate to HPM v3. The agent learns
linguistic structure from raw text via next-character prediction, then exposes
tokenization, number extraction, and embedding as tools. Once stable, the neural
pattern distils into symbolic rules via the existing SubstrateCompiler.

## Architecture Position

```
InnateCognitiveSubstrate  (arg wiring — always on)
    ↓
Population (ActionPatterns + LanguageModelPattern)
    ↓
ToolRegistry → language_model tool
    ↓
SubstrateCompiler → SymbolicPattern (post-distillation)
```

The `LanguageModelPattern` sits in the **learned population** — not the innate
layer. It competes with other patterns via replicator dynamics. The innate
substrate handles argument resolution for it just like any other tool.

## Relationship to InnateCognitiveSubstrate
- Substrate Group C already provides `extract_numbers(text)` as a regex fallback
- The LM provides a *learnable* version of the same capability plus embedding
- Post-distillation, the LM's learned behaviour replaces the substrate's regex
  fallback with a derived symbolic pattern — this is substrate shifting in HPM terms

## Neural Architecture
`CharLevelLSTM` — lightweight, no external deps beyond PyTorch:
- `vocab_size=128` (printable ASCII)
- `embed_dim=64`, `hidden_dim=128`, `n_layers=2`
- Input: character index sequence. Output: next-character logits + hidden state

## LanguageModelPattern (HPMPattern subclass)
File: `hpm_ai_v3/neural_lm_pattern.py`

Inherits `HPMPattern`. Implements:
- `pretrain(corpus_file, epochs, device)` — self-supervised next-char prediction
- `sample(context)` — dispatches on `context["action"]`:
  - `"tokenize"` → whitespace split (upgradeable)
  - `"extract_numbers"` → LM-guided span detection (regex fallback pre-training)
  - `"embed"` → final LSTM hidden state as float list
- `log_prob(obs)` → returns reward signal
- `update_parameters(obs)` → no-op (pretraining is offline)
- `save(path)` / `load(path)` — checkpoint model + vocab
- `structural_distance(other)` → 0.0 if same hidden_dim, 0.5 otherwise, 1.0 if not LM

## Tool Registration
`register_language_tool(lm_pattern)` registers `"language_model"` in ToolRegistry:
- `input_keys=["action", "text"]`
- `output_key="result"`
- `cost=0.05`

## Substrate Shifting (Distillation)
Once `lm_pattern.accuracy >= 0.9` on number extraction:
`compile_lm_to_symbolic(lm_pattern)` uses existing `SubstrateCompiler` to produce
a `SymbolicPattern` wrapping the derived regex. This symbolic pattern is added to
the population with high weight, replacing the neural pattern over time via
replicator dynamics.

## Embedding Integration
`train_cold_start.py` currently uses bag-of-chars embeddings for vector memory.
Replace with `lm_pattern._embed(text)` once trained, giving semantic retrieval.

## Curriculum Integration
Linguistic Analysis phase (phase 0) tasks can now use `language_model` tool:
- `action="tokenize"` for word-count tasks
- `action="extract_numbers"` for number extraction tasks
- `action="embed"` used internally for vector memory (not a curriculum task)

No curriculum JSON changes required — the agent discovers the tool via demo
absorption or population dynamics.

## Files
- **CREATE**: `hpm_ai_v3/neural_lm_pattern.py` — `CharLevelLSTM`, `LanguageModelPattern`, `register_language_tool`, `compile_lm_to_symbolic`
- **CREATE**: `hpm_ai_v3/data/lm_corpus/sample.txt` — small training corpus (1000 lines of simple text)
- **MODIFY**: `hpm_ai_v3/task8/train_cold_start.py` — optionally use LM embeddings for vector memory
- **CREATE**: `hpm_ai_v3/tests/test_lm_pattern.py` — unit + integration tests

## Success Criteria
- `CharLevelLSTM` trains on sample corpus, loss < 2.0 after 5 epochs
- `sample({"action": "extract_numbers", "text": "Speed is 20.5 m/s"})` returns `[20.5]`
- `sample({"action": "embed", "text": "hello"})` returns list of 128 floats
- `compile_lm_to_symbolic` returns a `SymbolicPattern` that passes same extraction tests
- Pattern registers in ToolRegistry and is callable by population via `resolve_call`
