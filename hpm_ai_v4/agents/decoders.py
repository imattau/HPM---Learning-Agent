"""Task-specific decoders layered on top of the reasoner."""
from collections import Counter, defaultdict
from typing import Optional

import numpy as np


class WordDecoder:
    """Corpus-guided word decoder with optional target bias."""

    def decode(
        self,
        agent,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
    ) -> str:
        if mode not in {"decode", "target", "hybrid"}:
            raise ValueError(f"Unsupported generation mode: {mode!r}")

        corpus_text = "".join(chr(v + 32) for v in agent._raw_history)
        if seed_text:
            seed_tokens = agent._tokenize_words(seed_text)
        else:
            seed_tokens = agent._tokenize_words(corpus_text[-200:])

        target_tokens = agent._tokenize_words(target_text) if target_text else []
        target_bias = 0.0
        if mode == "target":
            target_bias = 2.0
        elif mode == "hybrid":
            target_bias = 1.0

        tokens = agent._tokenize_words(corpus_text)
        if not tokens:
            return seed_text.strip() if seed_text else "the"

        followers = defaultdict(Counter)
        starts = Counter()
        for i, tok in enumerate(tokens):
            if i == 0:
                starts[tok] += 1
            else:
                followers[tokens[i - 1]][tok] += 1

        output_tokens = list(seed_tokens[-3:]) if (include_seed and seed_tokens) else []
        prev_token = seed_tokens[-1] if seed_tokens else max(starts, key=starts.get)
        generated_start = len(output_tokens)

        for _ in range(steps):
            context_raw = [ord(ch) - 32 for ch in " ".join(output_tokens)[-20:]]
            pred = agent.predict_next_chars(context_raw, top_k=5)
            pred_probs = {name: prob for name, prob in pred}
            gen_idx = max(0, len(output_tokens) - generated_start)
            expected_token = target_tokens[min(gen_idx, len(target_tokens) - 1)] if target_tokens else None

            candidates = followers.get(prev_token, Counter())
            if not candidates:
                candidates = starts

            best_tok = None
            best_score = -np.inf
            total = sum(candidates.values()) + 1e-12
            for tok, count in candidates.items():
                base = np.log(count / total)
                class_name = agent._token_class_name(tok)
                base += 1.5 * np.log(pred_probs.get(class_name, 1e-6) + 1e-12)
                if expected_token:
                    if tok == expected_token:
                        base += target_bias
                    elif tok.lower() == expected_token.lower():
                        base += target_bias * 0.8
                    elif agent._token_class_name(expected_token) == class_name:
                        base += target_bias * 0.25
                    else:
                        base -= target_bias * 0.15
                if agent.dictionary:
                    if agent.dictionary.contains(tok.lower()):
                        base += 0.5
                    elif agent.dictionary.is_prefix(tok.lower()):
                        base += 0.15
                    else:
                        base -= 0.1
                if agent.grammar and output_tokens:
                    prev_word = output_tokens[-1]
                    if prev_word and tok:
                        if agent.grammar.is_valid_transition(prev_word, tok):
                            base += 0.3
                        else:
                            base -= 0.05
                if tok == prev_token:
                    base -= 0.15
                if base > best_score:
                    best_score = base
                    best_tok = tok

            if best_tok is None:
                best_tok = max(candidates, key=candidates.get)

            output_tokens.append(best_tok)
            prev_token = best_tok

        text = agent._detokenize_words(output_tokens)
        if feedback:
            agent._feed_text_back(text)
        return text.strip()


class CharDecoder:
    """Character-level decoder that renders class predictions as printable text."""

    def decode(
        self,
        agent,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
    ) -> str:
        if mode not in {"decode", "target", "hybrid"}:
            raise ValueError(f"Unsupported generation mode: {mode!r}")

        corpus_text = "".join(chr(v + 32) for v in agent._raw_history)
        seed_chars = list(seed_text[-20:]) if seed_text else list(corpus_text[-20:])
        target_chars = list(target_text) if target_text else []

        target_bias = 0.0
        if mode == "target":
            target_bias = 2.0
        elif mode == "hybrid":
            target_bias = 1.0

        if not corpus_text:
            return (seed_text or "")[:steps]

        counts = Counter(corpus_text)
        transitions = defaultdict(Counter)
        for i in range(1, len(corpus_text)):
            transitions[corpus_text[i - 1]][corpus_text[i]] += 1

        output_chars = list(seed_chars if include_seed else [])
        prev_char = output_chars[-1] if output_chars else (corpus_text[-1] if corpus_text else " ")

        for _ in range(steps):
            context_raw = [ord(ch) - 32 for ch in output_chars[-20:]]
            pred = agent.predict_next_chars(context_raw, top_k=5)
            pred_probs = {name: prob for name, prob in pred}
            gen_idx = len(output_chars) if not include_seed else max(0, len(output_chars) - len(seed_chars))
            expected_char = target_chars[min(gen_idx, len(target_chars) - 1)] if target_chars else None

            candidates = transitions.get(prev_char, Counter())
            if not candidates:
                candidates = counts

            best_char = None
            best_score = -np.inf
            total = sum(candidates.values()) + 1e-12
            for ch, count in candidates.items():
                score = np.log(count / total)
                class_name = agent._token_class_name(ch if ch != "\n" else " ")
                score += 1.2 * np.log(pred_probs.get(class_name, 1e-6) + 1e-12)
                if expected_char:
                    if ch == expected_char:
                        score += target_bias
                    elif ch.lower() == expected_char.lower():
                        score += target_bias * 0.8
                    elif agent._token_class_name(expected_char) == class_name:
                        score += target_bias * 0.25
                    else:
                        score -= target_bias * 0.15
                if ch == prev_char:
                    score -= 0.15
                if score > best_score:
                    best_score = score
                    best_char = ch

            if best_char is None:
                best_char = max(candidates, key=candidates.get)

            output_chars.append(best_char)
            prev_char = best_char

        text = "".join(output_chars)
        if feedback:
            agent._feed_text_back(text)
        return text.strip()


class TargetDecoder:
    """Target-conditioned continuation decoder based on the reasoner."""

    def decode(
        self,
        agent,
        target_text: str,
        seed_text: str | None = None,
        horizon: Optional[int] = None,
        strategy: str = "beam",
        lookback: Optional[int] = None,
        feature_pack: Optional[dict] = None,
    ) -> str:
        target_classes = [agent._adapter.encode(ord(ch) - 32) for ch in target_text if 32 <= ord(ch) <= 126]
        if not target_classes:
            return ""

        planned_classes = agent.l1.reasoner.plan_sequence(
            target_sequence=target_classes,
            horizon=horizon if horizon is not None else len(target_classes),
            strategy=strategy,
            lookback=lookback,
            target_weight=2.5,
            feature_pack=feature_pack,
        )

        target_chars = [ch for ch in target_text if 32 <= ord(ch) <= 126]
        chars = []
        prev_char: str | None = None
        for idx, class_id in enumerate(planned_classes):
            preferred_char = target_chars[min(idx, len(target_chars) - 1)] if target_chars else None
            if preferred_char is not None and agent._adapter.encode_char(preferred_char) == class_id:
                ch = preferred_char
            else:
                ch = agent._choose_char_for_class(class_id, prev_char, preferred_char=preferred_char)
            chars.append(ch)
            prev_char = ch
        return "".join(chars).strip()


class ConstrainedDecoder(WordDecoder):
    """Word decoder with extra lexical constraints."""

    def decode(
        self,
        agent,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
        allowed_words: Optional[set[str]] = None,
        strict_dictionary: bool = True,
        strict_grammar: bool = True,
    ) -> str:
        if mode not in {"decode", "target", "hybrid"}:
            raise ValueError(f"Unsupported generation mode: {mode!r}")

        corpus_text = "".join(chr(v + 32) for v in agent._raw_history)
        if seed_text:
            seed_tokens = agent._tokenize_words(seed_text)
        else:
            seed_tokens = agent._tokenize_words(corpus_text[-200:])
        target_tokens = agent._tokenize_words(target_text) if target_text else []
        target_bias = 2.0 if mode == "target" else 1.0 if mode == "hybrid" else 0.0

        tokens = agent._tokenize_words(corpus_text)
        if not tokens:
            return seed_text.strip() if seed_text else "the"

        followers = defaultdict(Counter)
        starts = Counter()
        for i, tok in enumerate(tokens):
            if i == 0:
                starts[tok] += 1
            else:
                followers[tokens[i - 1]][tok] += 1

        output_tokens = list(seed_tokens[-3:]) if (include_seed and seed_tokens) else []
        prev_token = seed_tokens[-1] if seed_tokens else max(starts, key=starts.get)
        generated_start = len(output_tokens)

        for _ in range(steps):
            context_raw = [ord(ch) - 32 for ch in " ".join(output_tokens)[-20:]]
            pred = agent.predict_next_chars(context_raw, top_k=5)
            pred_probs = {name: prob for name, prob in pred}
            gen_idx = max(0, len(output_tokens) - generated_start)
            expected_token = target_tokens[min(gen_idx, len(target_tokens) - 1)] if target_tokens else None

            candidates = followers.get(prev_token, Counter())
            if not candidates:
                candidates = starts

            best_tok = None
            best_score = -np.inf
            total = sum(candidates.values()) + 1e-12
            for tok, count in candidates.items():
                score = np.log(count / total)
                class_name = agent._token_class_name(tok)
                score += 1.5 * np.log(pred_probs.get(class_name, 1e-6) + 1e-12)
                if expected_token:
                    if tok == expected_token:
                        score += target_bias
                    elif tok.lower() == expected_token.lower():
                        score += target_bias * 0.8
                    elif agent._token_class_name(expected_token) == class_name:
                        score += target_bias * 0.25
                    else:
                        score -= target_bias * 0.15
                if allowed_words is not None:
                    lower = tok.lower()
                    if lower not in allowed_words and not any(w.startswith(lower) for w in allowed_words):
                        score -= 0.75
                if strict_dictionary and agent.dictionary:
                    if agent.dictionary.contains(tok.lower()):
                        score += 0.5
                    elif agent.dictionary.is_prefix(tok.lower()):
                        score += 0.15
                    else:
                        score -= 0.15
                if strict_grammar and agent.grammar and output_tokens:
                    prev_word = output_tokens[-1]
                    if prev_word and tok:
                        if agent.grammar.is_valid_transition(prev_word, tok):
                            score += 0.3
                        else:
                            score -= 0.1
                if tok == prev_token:
                    score -= 0.15
                if score > best_score:
                    best_score = score
                    best_tok = tok

            if best_tok is None:
                best_tok = max(candidates, key=candidates.get)

            output_tokens.append(best_tok)
            prev_token = best_tok

        text = agent._detokenize_words(output_tokens)
        if feedback:
            agent._feed_text_back(text)
        return text.strip()


class AdaptiveDecoder:
    """Arbitrate among existing decoders using the observed output quality."""

    def decode(
        self,
        agent,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
        allowed_words: Optional[set[str]] = None,
        strict_dictionary: bool = True,
        strict_grammar: bool = True,
        use_word: bool = True,
        use_char: bool = True,
        use_target: bool = True,
        use_constrained: bool = True,
    ) -> str:
        candidates: list[tuple[str, str]] = []

        if use_word and "word" in agent.decoders:
            candidates.append((
                "word",
                agent.decoders["word"].decode(
                    agent,
                    steps=steps,
                    seed_text=seed_text,
                    target_text=target_text,
                    mode=mode,
                    include_seed=include_seed,
                    feedback=False,
                ),
            ))

        if use_char and "char" in agent.decoders:
            candidates.append((
                "char",
                agent.decoders["char"].decode(
                    agent,
                    steps=steps,
                    seed_text=seed_text,
                    target_text=target_text,
                    mode=mode,
                    include_seed=include_seed,
                    feedback=False,
                ),
            ))

        if use_constrained and "constrained" in agent.decoders and (agent.dictionary or agent.grammar):
            candidates.append((
                "constrained",
                agent.decoders["constrained"].decode(
                    agent,
                    steps=steps,
                    seed_text=seed_text,
                    target_text=target_text,
                    mode=mode,
                    include_seed=include_seed,
                    feedback=False,
                    allowed_words=allowed_words,
                    strict_dictionary=strict_dictionary,
                    strict_grammar=strict_grammar,
                ),
            ))

        if use_target and target_text and "target" in agent.decoders:
            candidates.append((
                "target",
                agent.decoders["target"].decode(
                    agent,
                    target_text=target_text,
                    seed_text=seed_text,
                    horizon=steps,
                    strategy="beam",
                    lookback=agent.l1.reasoner.context_window,
                ),
            ))

        if not candidates:
            fallback = agent.decoders["word"].decode(
                agent,
                steps=steps,
                seed_text=seed_text,
                target_text=target_text,
                mode=mode,
                include_seed=include_seed,
                feedback=False,
            )
            agent._last_decoder_choice = "word"
            if feedback:
                agent._feed_text_back(fallback)
            return fallback

        best_name = candidates[0][0]
        best_text = candidates[0][1]
        best_score = self._score(agent, best_text, target_text)

        for name, text in candidates[1:]:
            score = self._score(agent, text, target_text)
            if score > best_score:
                best_name = name
                best_text = text
                best_score = score

        agent._last_decoder_choice = best_name
        if feedback:
            agent._feed_text_back(best_text)
        return best_text

    def _score(self, agent, text: str, target_text: str | None) -> tuple[float, float, int]:
        if target_text:
            stats = agent.evaluate_generated_text(text, target_text)
            return (
                float(stats["token_agreement"]),
                float(stats["plausibility"]),
                -len(text),
            )
        return (
            float(agent._text_plausibility(text)),
            float(agent._text_agreement(text, text)),
            -len(text),
        )


class ExplanationDecoder:
    """Human-readable rendering of the stacked latent state."""

    def decode(self, agent, steps: int = 20) -> str:
        future = agent.l3.reasoner.simulate_future(steps=steps, top_k=3, lookback=agent.l3.reasoner.context_window)
        return " ".join(f"L3:{v}" for v in future)
