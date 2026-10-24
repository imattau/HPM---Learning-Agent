"""L5 prequential mixture calibration methods for the HFN language model."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterable

from hpm_ai_v2.training.language_constants import BOUNDARY
from hpm_ai_v2.training.language_corpus import split_sentences, tokenize
from hpm_ai_v2.training.l5_language_meta import PrequentialL5Mixer

class LanguageL5Mixin:
    def _l5_order_expert_probabilities(
        self, context: tuple[str, ...], target: str, *, order_alpha: float = 0.1,
    ) -> dict[str, float]:
        """Score target under the fixed Observer and raw order experts."""
        vocabulary_size = max(1, len(self.vocab))
        result = {"observer": self._probability(context, target)}
        for order in range(self.max_order):
            suffix = context[-order:] if order else ()
            counts = self.counts.get(suffix, Counter())
            total = sum(counts.values())
            result[f"word_order_{order}"] = (
                counts.get(target, 0) + order_alpha
            ) / (total + order_alpha * vocabulary_size)
        return result
    def _new_l5_mixer(self, eta: float, fixed_share: float = 0.0) -> PrequentialL5Mixer:
        names = ("observer", *(f"word_order_{order}" for order in range(self.max_order)))
        mixer = PrequentialL5Mixer(names, eta=eta, fixed_share=fixed_share)
        # Keep most initial mass on the already calibrated predictor; L5 must
        # earn a shift toward raw order experts from prequential evidence.
        mixer.log_weights["observer"] = math.log(0.8)
        raw_prior = 0.2 / self.max_order
        for name in names[1:]:
            mixer.log_weights[name] = math.log(raw_prior)
        return mixer
    def fit_l5_meta(
        self, validation_documents: Iterable[str], *, eta: float = 0.1,
        order_alpha: float = 0.1, fixed_share: float = 0.0,
    ) -> dict:
        """Calibrate L5 over word-order experts on validation, prequentially.

        Each event is scored before its target updates the meta-weights. The
        base HFN count tables remain frozen; this only calibrates which context
        order is reliable for later predictions.
        """
        if not math.isfinite(order_alpha) or order_alpha <= 0:
            raise ValueError("order_alpha must be finite and positive")
        mixer = self._new_l5_mixer(eta, fixed_share=fixed_share)
        nll = count = 0
        documents = list(validation_documents)
        if not any(tokenize(sentence) for text in documents for sentence in split_sentences(text)):
            raise ValueError("L5 validation calibration requires at least one lexical event")
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    probabilities = self._l5_order_expert_probabilities(
                        context, target, order_alpha=order_alpha,
                    )
                    probability = mixer.probability(probabilities)
                    nll -= math.log(max(probability, 1e-300))
                    mixer.update_after_score(probabilities)
                    count += 1
        self.l5_language_state = {**mixer.state_dict(), "order_alpha": float(order_alpha)}
        self.l5_calibration_document_hashes = {
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents
        }
        mean_loss = nll / max(count, 1)
        return {"events": count, "prequential_cross_entropy_nats": mean_loss,
                "prequential_perplexity": math.exp(min(mean_loss, 700)),
                "weights": mixer.weights(), "state": self.l5_language_state,
                "calibration_split": "validation"}
    def continue_l5_meta(self, new_validation_documents: Iterable[str]) -> dict:
        """Continue L5 reliability learning from exact-text-new validation data.

        This carries forward the selected L5 weights and learning rate. It is
        intended for a later data round after HFN training has incorporated the
        new training split; validation targets update L5 only after scoring.
        """
        if self.l5_language_state is None:
            raise RuntimeError("fit or tune L5 before continuing its calibration")
        documents = list(new_validation_documents)
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents]
        if not documents or len(set(hashes)) != len(hashes):
            raise ValueError("new L5 calibration documents must be nonempty and unique")
        if self.l5_calibration_document_hashes.intersection(hashes):
            raise ValueError("L5 calibration rejects exact-text replay from prior rounds")
        mixer = PrequentialL5Mixer.from_state_dict(self.l5_language_state)
        order_alpha = float(self.l5_language_state.get("order_alpha", 0.1))
        nll = count = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    probabilities = self._l5_order_expert_probabilities(
                        context, target, order_alpha=order_alpha,
                    )
                    probability = mixer.probability(probabilities)
                    nll -= math.log(max(probability, 1e-300))
                    mixer.update_after_score(probabilities)
                    count += 1
        if count == 0:
            raise ValueError("new L5 calibration data must contain a lexical event")
        previous_updates = int(self.l5_language_state["updates"])
        self.l5_language_state = {**mixer.state_dict(), "order_alpha": order_alpha}
        self.l5_calibration_document_hashes.update(hashes)
        mean_loss = nll / count
        return {"events": count, "previous_updates": previous_updates,
                "cumulative_updates": mixer.updates,
                "prequential_cross_entropy_nats": mean_loss,
                "weights": mixer.weights(), "state": self.l5_language_state,
                "new_validation_documents": len(documents)}
    def tune_l5_meta(
        self, validation_documents: Iterable[str], *,
        eta_candidates: Iterable[float] = (0.0001, 0.0003, 0.001, 0.003, 0.01),
        order_alpha_candidates: Iterable[float] = (0.1, 1.0, 10.0),
        fixed_share_candidates: Iterable[float] = (0.0, 0.02),
    ) -> dict:
        """Select L5 update settings by validation prequential log loss only.

        The fixed-share candidate set includes the legacy no-share behavior and
        a small share rate. Direct mixer construction and ``fit_l5_meta`` keep
        fixed share disabled unless it is explicitly requested.
        """
        documents = list(validation_documents)
        candidates = tuple(sorted(set(float(value) for value in eta_candidates)))
        alpha_candidates = tuple(sorted(set(float(value) for value in order_alpha_candidates)))
        share_candidates = tuple(sorted(set(float(value) for value in fixed_share_candidates)))
        if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
            raise ValueError("eta_candidates must contain finite positive values")
        if not alpha_candidates or any(not math.isfinite(value) or value <= 0 for value in alpha_candidates):
            raise ValueError("order_alpha_candidates must contain finite positive values")
        if not share_candidates or any(not math.isfinite(value) or not 0.0 <= value < 1.0
                                       for value in share_candidates):
            raise ValueError("fixed_share_candidates must contain finite values in [0, 1)")
        fixed_loss = self.score_cross_entropy(documents)
        sweep = []
        best = None
        for eta in candidates:
            for order_alpha in alpha_candidates:
                for fixed_share in share_candidates:
                    result = self.fit_l5_meta(documents, eta=eta, order_alpha=order_alpha,
                                              fixed_share=fixed_share)
                    item = {"eta": eta, "order_alpha": order_alpha,
                        "fixed_share": fixed_share,
                        "prequential_cross_entropy_nats": result["prequential_cross_entropy_nats"],
                        "weights": result["weights"]}
                    sweep.append(item)
                    if best is None or item["prequential_cross_entropy_nats"] < best["prequential_cross_entropy_nats"]:
                        best = item
                        selected_state = result["state"]
        self.l5_language_state = selected_state
        return {"fixed_observer_validation_cross_entropy_nats": fixed_loss,
                "selected_eta": best["eta"],
                "selected_order_alpha": best["order_alpha"],
                "selected_fixed_share": best["fixed_share"],
                "selected_prequential_cross_entropy_nats": best["prequential_cross_entropy_nats"],
                "sweep": sweep, "selected_state": selected_state,
                "selection_split": "validation"}
    def evaluate_l5_prequential(
        self, documents: Iterable[str], *, state: dict | None = None,
    ) -> dict:
        """Score a stream with L5 weights, updating only after each prediction.

        The supplied or checkpoint state is copied into a local mixer. Returned
        state can be carried to a later stream, while evaluation never mutates
        the fitted checkpoint implicitly.
        """
        if self.l5_language_state is None and state is None:
            raise RuntimeError("fit L5 on validation data or provide a saved L5 state first")
        mixer = PrequentialL5Mixer.from_state_dict(
            state if state is not None else self.l5_language_state
        )
        order_alpha = float((state if state is not None else self.l5_language_state).get("order_alpha", 0.1))
        nll = count = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    probabilities = self._l5_order_expert_probabilities(
                        context, target, order_alpha=order_alpha,
                    )
                    probability = mixer.probability(probabilities)
                    nll -= math.log(max(probability, 1e-300))
                    mixer.update_after_score(probabilities)
                    count += 1
        mean_loss = nll / max(count, 1)
        return {"events": count, "prequential_cross_entropy_nats": mean_loss,
                "prequential_perplexity": math.exp(min(mean_loss, 700)),
                "weights": mixer.weights(), "state": mixer.state_dict(),
                "starting_updates": int((state if state is not None else self.l5_language_state)["updates"]),
                "ending_updates": mixer.updates}
