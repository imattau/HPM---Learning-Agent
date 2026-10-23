"""Supervised grounded-study loop for ReaderAgent.

This is the first training slice: source passages become L1 patterns; labeled
question/evidence episodes reinforce or penalize passage utility; validation
episodes calibrate abstention; and test episodes are read-only evaluations.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.domains.text_domain import tokenise


Split = Literal["train", "validation", "test"]
TASK_TYPES = {"grounded_qa", "abstain", "correction", "define", "summarize", "outline", "flashcards"}
MULTI_SENTENCE_TASKS = {"summarize", "outline", "flashcards"}


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    text: str
    title: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.text.strip():
            raise ValueError("source text must not be empty")


@dataclass(frozen=True)
class StudyEpisode:
    """One supervised retrieval or abstention episode.

    ``evidence_id`` names a SourceDocument. Use ``None`` for an unanswerable
    question. ``expected_answer`` is retained as a human-readable reference;
    this first trainer evaluates evidence selection, not answer generation.
    """

    episode_id: str
    split: Split
    task_type: str
    question: str
    evidence_id: Optional[str]
    expected_answer: Optional[str] = None
    user_id: str = "default"
    correction_of: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task_type: {self.task_type}")
        if self.task_type == "abstain" and self.evidence_id is not None:
            raise ValueError("abstain episodes must not name evidence")
        if self.task_type != "abstain" and self.evidence_id is None:
            raise ValueError("answerable episodes must name evidence")


@dataclass(frozen=True)
class StudyPrediction:
    episode_id: str
    selected_evidence_id: Optional[str]
    confidence: float
    ranked_evidence: tuple[tuple[str, float], ...]
    abstained: bool
    answer: Optional[str] = None
    citation: Optional[dict[str, Any]] = None
    citations: tuple[dict[str, Any], ...] = ()
    abstention_reason: Optional[str] = None


def load_curriculum_jsonl(path: str | Path) -> tuple[list[SourceDocument], list[StudyEpisode]]:
    """Load a UTF-8 JSONL curriculum containing ``source`` and ``episode`` rows."""
    sources: list[SourceDocument] = []
    episodes: list[StudyEpisode] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                kind = row.pop("kind")
                if kind == "source":
                    sources.append(SourceDocument(**row))
                elif kind == "episode":
                    episodes.append(StudyEpisode(**row))
                else:
                    raise ValueError(f"unknown row kind {kind!r}")
            except Exception as exc:
                raise ValueError(f"invalid curriculum row at {path}:{line_number}: {exc}") from exc
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("curriculum contains duplicate source_id values")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("curriculum contains duplicate episode_id values")
    return sources, episodes


class SupervisedStudyTrainer:
    """Train and evaluate evidence retrieval using ReaderAgent HFN weights."""

    def __init__(
        self,
        agent: Any,
        *,
        event_log: Optional[str | Path] = None,
        feedback_gain: float = 1.0,
        feedback_penalty: float = 0.25,
        correction_gain: Optional[float] = None,
        correction_penalty: Optional[float] = None,
        semantic_lexical_bonus: float = 0.05,
        semantic_lexical_rescue_threshold: float = 0.5,
        semantic_lexical_rescue_margin: float = 0.0,
        abstention_similarity_threshold: float = 0.8,
    ) -> None:
        if not hasattr(agent, "observe_passage") or not hasattr(agent, "query_scored_nodes"):
            raise TypeError("agent must provide ReaderAgent passage and retrieval methods")
        self.agent = agent
        self.event_log = Path(event_log) if event_log else None
        self.feedback_gain = feedback_gain
        self.feedback_penalty = feedback_penalty
        self.correction_gain = (
            float(correction_gain) if correction_gain is not None else feedback_gain * 2.0
        )
        self.correction_penalty = (
            float(correction_penalty)
            if correction_penalty is not None
            else min(feedback_penalty * 2.0, 0.75)
        )
        if not 0.0 <= semantic_lexical_bonus <= 1.0:
            raise ValueError("semantic_lexical_bonus must be between 0 and 1")
        if not 0.0 <= semantic_lexical_rescue_threshold <= 1.0:
            raise ValueError("semantic_lexical_rescue_threshold must be between 0 and 1")
        if not 0.0 <= semantic_lexical_rescue_margin <= 1.0:
            raise ValueError("semantic_lexical_rescue_margin must be between 0 and 1")
        if not 0.0 <= abstention_similarity_threshold <= 1.0:
            raise ValueError("abstention_similarity_threshold must be between 0 and 1")
        self.semantic_lexical_bonus = float(semantic_lexical_bonus)
        self.semantic_lexical_rescue_threshold = float(semantic_lexical_rescue_threshold)
        self.semantic_lexical_rescue_margin = float(semantic_lexical_rescue_margin)
        self.abstention_similarity_threshold = float(abstention_similarity_threshold)
        self.abstain_threshold = 0.0
        # 2: sentences are hierarchy nodes with offsets, linked by next edges and
        # shared term nodes. 1 (older checkpoints): a flat answer_sentence copy.
        self.storage_version = 2
        self.sources: dict[str, SourceDocument] = {}
        self.source_to_node: dict[str, str] = {}
        self.node_to_source: dict[str, str] = {}
        self.source_to_answer_nodes: dict[str, list[str]] = {}
        self.answer_node_data: dict[str, dict[str, Any]] = {}
        self.fact_key_to_node: dict[str, str] = {}
        self.fact_node_data: dict[str, dict[str, Any]] = {}
        self.source_to_fact_nodes: dict[str, list[str]] = {}
        self.abstention_node_data: dict[str, dict[str, Any]] = {}
        self._event_count = 0
        self._fit_episode_ids: set[str] = set()

    def ingest(self, sources: Iterable[SourceDocument]) -> None:
        """Register each source once as a passage pattern in the HFN forest."""
        for source in sources:
            if source.source_id in self.sources:
                if self.sources[source.source_id] != source:
                    raise ValueError(f"source_id reused with different content: {source.source_id}")
                continue
            # Start evidence patterns below saturation so correct feedback can
            # strengthen them as well as incorrect feedback weakening distractors.
            spans = _sentences_with_offsets(source.text)
            index = self.agent.observe_passage(
                source.text,
                initial_weight=0.5,
                sentence_spans=spans,
                sentence_ids=[self._sentence_node_id(source.source_id, s, start) for s, start, _ in spans],
                sentence_weight=0.5,
                source_id=source.source_id,
            )
            node_id = f"passage_{index}"
            self.sources[source.source_id] = source
            self.source_to_node[source.source_id] = node_id
            self.node_to_source[node_id] = source.source_id
            self._index_answer_sentences(source)
            self._log({"event": "source_ingested", "source_id": source.source_id, "node_id": node_id})

    @staticmethod
    def _sentence_node_id(source_id: str, sentence: str, start: int) -> str:
        # The "answer_sentence_" prefix is legacy: learned facts and checkpoints
        # already refer to sentences by these ids, so the scheme is kept.
        digest = hashlib.sha1(f"{source_id}:{start}:{sentence}".encode()).hexdigest()[:12]
        return f"answer_sentence_{digest}"

    def _index_answer_sentences(self, source: SourceDocument) -> None:
        """Record the source's sentence nodes, built by the document hierarchy.

        The sentence nodes are the ReaderAgent hierarchy's own (passage ->
        paragraph -> sentence), so there is one node per sentence, not a flat
        parallel copy. This only maps them to their text and offsets.
        """
        self.source_to_answer_nodes[source.source_id] = []
        for sentence, start, end in _sentences_with_offsets(source.text):
            node_id = self._sentence_node_id(source.source_id, sentence, start)
            if node_id not in self.agent.patterns:
                raise RuntimeError(f"sentence node was not created for {source.source_id}:{start}")
            if node_id not in self.answer_node_data:
                self.answer_node_data[node_id] = {
                    "source_id": source.source_id,
                    "text": sentence,
                    "start": start,
                    "end": end,
                }
            self.source_to_answer_nodes[source.source_id].append(node_id)

    def _rank_answers(self, question: str, source_id: str) -> list[tuple[str, float]]:
        query = self.agent.config.encode_passage(question)
        s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
        query_vec = query[s_dim : s_dim + dim]
        ranked = []
        for node_id in self.source_to_answer_nodes.get(source_id, []):
            node = self.agent.forest.get(node_id)
            if node is None:
                continue
            passage_vec = node.mu[s_dim : s_dim + dim]
            denom = float(np.linalg.norm(query_vec) * np.linalg.norm(passage_vec) + 1e-9)
            similarity = float(np.dot(query_vec, passage_vec) / denom)
            if similarity <= 0:
                continue
            weight = float(self.agent.observer.get_weight(node_id))
            fact_utility = self._fact_utility(source_id, node_id)
            ranked.append((
                node_id,
                similarity * max(0.0, min(1.0, weight)) * fact_utility,
            ))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _gold_answer_node(self, source_id: str, expected_answer: str) -> Optional[str]:
        """Map a reference answer to the best source sentence for supervision."""
        candidates = self.source_to_answer_nodes.get(source_id, [])
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda node_id: (
                -_token_f1(expected_answer, self.answer_node_data[node_id]["text"]),
                node_id,
            ),
        )
        best = ranked[0]
        return best if _token_f1(expected_answer, self.answer_node_data[best]["text"]) > 0 else None

    def _rank(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        """Rank evidence using both source passages and learned fact patterns."""
        scores_by_source: dict[str, float] = {}
        passage_count = len(getattr(self.agent.config, "_passage_vecs", ()))
        for node_id, _text, similarity in self.agent.query_scored_nodes(
            question, k=max(k, len(self.sources), passage_count)
        ):
            # A zero lexical overlap is absence of retrievable evidence, not a
            # zero-confidence match that may win by deterministic tie-break.
            if similarity <= 0.0:
                continue
            source_id = self.node_to_source.get(node_id)
            if source_id is None:
                continue
            weight = float(self.agent.observer.get_weight(node_id))
            # A neutral weight of 1 leaves the representation score unchanged;
            # penalties shift future retrieval away from repeatedly wrong nodes.
            score = float(similarity) * max(0.0, min(1.0, weight))
            scores_by_source[source_id] = max(scores_by_source.get(source_id, 0.0), score)

        # Fact patterns are source-backed L2 knowledge created from supervised
        # answers. Let them retrieve their supporting sources directly; without
        # this path, fact nodes only affected answer sentence utility after the
        # passage-level retriever had already chosen a source.
        if self.fact_node_data:
            query = self.agent.config.encode_passage(question)
            s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
            query_vec = query[s_dim : s_dim + dim]
            query_norm = float(np.linalg.norm(query_vec))
            if query_norm > 0.0:
                for fact_id, data in self.fact_node_data.items():
                    fact_node = self.agent.forest.get(fact_id)
                    if fact_node is None:
                        continue
                    fact_vec = fact_node.mu[s_dim : s_dim + dim]
                    denom = query_norm * float(np.linalg.norm(fact_vec)) + 1e-9
                    similarity = float(np.dot(query_vec, fact_vec) / denom)
                    if similarity <= 0.0:
                        continue
                    status_factor = {
                        "candidate": 0.95,
                        "validated": 1.0,
                        "promoted": 1.05,
                    }.get(data.get("status", "candidate"), 0.9)
                    weight = max(0.0, min(1.0, float(self.agent.observer.get_weight(fact_id))))
                    fact_support = similarity * weight * status_factor
                    for source_id in data.get("source_ids", ()):
                        if source_id in self.source_to_node:
                            # Keep the corpus match as the primary signal. A
                            # small learned-fact bonus lets validated knowledge
                            # break close retrieval ties without erasing source
                            # specificity when the same fact has multiple proofs.
                            base_score = scores_by_source.get(source_id, 0.0)
                            scores_by_source[source_id] = min(
                                1.0, base_score + 0.1 * fact_support
                            )

        # Dense embeddings can rank a semantically related distractor above a
        # source that contains a distinctive query term (for example,
        # "distance" in a bee-dance question). Add a small BM25 corroboration
        # signal for the frozen semantic encoder, then let validation calibrate
        # abstention over the fused score. TF-IDF already supplies this signal.
        if hasattr(self.agent.config, "embedding_model") and self.semantic_lexical_bonus:
            lexical_scores = self._bm25_source_scores(question)
            for source_id, lexical_score in lexical_scores.items():
                normalized = lexical_score / (lexical_score + 2.0)
                scores_by_source[source_id] = min(
                    1.0,
                    scores_by_source.get(source_id, 0.0)
                    + self.semantic_lexical_bonus * normalized,
                )

        ranked = list(scores_by_source.items())
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def _bm25_source_scores(self, question: str) -> dict[str, float]:
        """Return standard BM25 scores for the registered source corpus."""
        query_terms = set(tokenise(question))
        if not query_terms or not self.sources:
            return {}
        token_rows = {
            source_id: tokenise(source.text)
            for source_id, source in self.sources.items()
        }
        average_length = sum(map(len, token_rows.values())) / len(token_rows) or 1.0
        document_frequency = {
            term: sum(term in tokens for tokens in token_rows.values())
            for term in query_terms
        }
        k1, b = 1.5, 0.75
        scores = {}
        for source_id, tokens in token_rows.items():
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = document_frequency[term]
                idf = math.log1p((len(token_rows) - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1.0 - b + b * length / average_length)
                score += idf * tf * (k1 + 1.0) / denominator
            if score > 0.0:
                scores[source_id] = score
        return scores

    def predict(self, episode: StudyEpisode) -> StudyPrediction:
        return self.predict_question(
            episode.question, episode.episode_id, episode.task_type, user_id=episode.user_id
        )

    def predict_question(
        self,
        question: str,
        episode_id: str = "ad-hoc",
        task_type: str = "grounded_qa",
        *,
        user_id: str = "default",
        scope_source_id: Optional[str] = None,
    ) -> StudyPrediction:
        """Read-only evidence selection for a free-form question (no labels needed)."""
        if task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task_type: {task_type}")
        ranked = self._rank(question)
        if scope_source_id is not None:
            if scope_source_id not in self.sources:
                raise ValueError(f"unknown scope_source_id: {scope_source_id}")
            ranked = [item for item in ranked if item[0] == scope_source_id]
        confidence = ranked[0][1] if ranked else 0.0
        selected = ranked[0][0] if ranked and confidence >= self.abstain_threshold else None
        abstention_reason = "calibrated_threshold" if ranked and selected is None else None
        matched_abstention = self._matching_abstention_pattern(question, user_id)
        if matched_abstention is not None:
            selected = None
            abstention_reason = "learned_negative_pattern"
        if (
            selected is None
            and ranked
            and matched_abstention is None
            and hasattr(self.agent.config, "embedding_model")
            and confidence >= self.abstain_threshold - self.semantic_lexical_rescue_margin
        ):
            lexical_scores = self._bm25_source_scores(question)
            lexical_best = max(lexical_scores, key=lexical_scores.get) if lexical_scores else None
            lexical_confidence = (
                lexical_scores[lexical_best] / (lexical_scores[lexical_best] + 2.0)
                if lexical_best is not None else 0.0
            )
            # Rescue only a near-threshold answer when independent, strong
            # lexical support agrees with the semantic top source.
            if (
                lexical_best == ranked[0][0]
                and lexical_confidence >= self.semantic_lexical_rescue_threshold
            ):
                selected = ranked[0][0]
        answer = None
        citation = None
        citations: tuple[dict[str, Any], ...] = ()
        if selected is not None:
            sentence_rank = self._rank_answers(question, selected)
            if task_type in MULTI_SENTENCE_TASKS:
                ranked_ids = {node_id: score for node_id, score in sentence_rank}
                for node_id in self.source_to_answer_nodes.get(selected, []):
                    ranked_ids.setdefault(node_id, 0.0)
                sentence_rank = sorted(
                    ranked_ids.items(),
                    key=lambda item: (-item[1], self.answer_node_data[item[0]]["start"]),
                )
            if sentence_rank:
                source = self.sources[selected]
                limit = 3 if task_type in MULTI_SENTENCE_TASKS else 1
                selected_answers = sentence_rank[:limit]
                # Keep extractive multi-sentence outputs in their source order.
                if task_type in MULTI_SENTENCE_TASKS:
                    selected_answers = sorted(
                        selected_answers,
                        key=lambda item: self.answer_node_data[item[0]]["start"],
                    )
                citation_rows = []
                for node_id, _score in selected_answers:
                    answer_data = self.answer_node_data[node_id]
                    citation_rows.append({
                        "source_id": selected,
                        "title": source.title,
                        "start": answer_data["start"],
                        "end": answer_data["end"],
                        "text": source.text[answer_data["start"] : answer_data["end"]],
                    })
                citations = tuple(citation_rows)
                citation = citations[0]
                answer = "\n".join(item["text"] for item in citations)
        return StudyPrediction(
            episode_id=episode_id,
            selected_evidence_id=selected,
            confidence=confidence,
            ranked_evidence=tuple(ranked),
            abstained=selected is None,
            answer=answer,
            citation=citation,
            citations=citations,
            abstention_reason=abstention_reason,
        )

    def _matching_abstention_pattern(
        self, question: str, user_id: str
    ) -> Optional[tuple[str, float]]:
        """Find a sufficiently similar supervised negative example for this user."""
        if not self.abstention_node_data:
            return None
        query = self.agent.config.encode_passage(question)
        s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
        query_vec = query[s_dim : s_dim + dim]
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm <= 0.0:
            return None
        best: Optional[tuple[str, float]] = None
        for node_id, data in self.abstention_node_data.items():
            if data.get("user_id", "default") != user_id:
                continue
            node = self.agent.forest.get(node_id)
            if node is None:
                continue
            pattern_vec = node.mu[s_dim : s_dim + dim]
            denom = query_norm * float(np.linalg.norm(pattern_vec)) + 1e-9
            similarity = float(np.dot(query_vec, pattern_vec) / denom)
            if best is None or similarity > best[1]:
                best = (node_id, similarity)
        if best and best[1] >= self.abstention_similarity_threshold:
            return best
        return None

    def fit(
        self,
        sources: Iterable[SourceDocument],
        train_episodes: Iterable[StudyEpisode],
        validation_episodes: Iterable[StudyEpisode] = (),
    ) -> dict[str, Any]:
        """Ingest the training corpus, apply labeled feedback, and calibrate abstention.

        Validation episodes tune only the abstention threshold. They do not
        update HFN pattern weights. Test episodes must never be passed here.
        """
        train = list(train_episodes)
        validation = list(validation_episodes)
        self._validate_split(train, "train")
        self._validate_split(validation, "validation")
        overlap = {ep.episode_id for ep in train} & {ep.episode_id for ep in validation}
        overlap |= ({ep.episode_id for ep in train + validation} & self._fit_episode_ids)
        if overlap:
            raise ValueError(f"episode IDs reused across training/validation: {sorted(overlap)[:3]}")
        self.ingest(sources)
        facts_before = set(self.fact_node_data)
        support_edges_before = sum(len(data["answer_sentence_ids"]) for data in self.fact_node_data.values())
        correction_links_before = sum(len(data.get("corrects_node_ids", ())) for data in self.fact_node_data.values())
        abstentions_before = set(self.abstention_node_data)
        for episode in train:
            self._validate_evidence(episode)
            ranked = self._rank(episode.question)
            predicted = ranked[0][0] if ranked else None
            correct = predicted == episode.evidence_id
            is_correction = episode.task_type == "correction"
            reinforcement_gain = self.correction_gain if is_correction else self.feedback_gain
            rejection_penalty = self.correction_penalty if is_correction else self.feedback_penalty
            target_answer_node = None
            fact_id = None
            fact_created = False
            support_added = False
            correction_added = False
            abstention_node_id = None
            abstention_pattern_created = False
            predicted_answer_node = None
            if predicted is not None:
                predicted_answers = self._rank_answers(episode.question, predicted)
                if predicted_answers:
                    predicted_answer_node = predicted_answers[0][0]
            correction_target_id = None
            if episode.task_type == "correction" and predicted is not None:
                if predicted != episode.evidence_id:
                    correction_target_id = self.source_to_node[predicted]
                elif predicted_answer_node is not None:
                    correction_target_id = predicted_answer_node
            if episode.evidence_id is not None:
                gold_node = self.source_to_node[episode.evidence_id]
                self.agent.observer.boost_id(gold_node, gain=reinforcement_gain)
                if episode.expected_answer:
                    target_answer_node = self._gold_answer_node(
                        episode.evidence_id, episode.expected_answer
                    )
                    if target_answer_node is not None:
                        self.agent.observer.boost_id(target_answer_node, gain=reinforcement_gain)
                        fact_id, fact_created, support_added, correction_added = self._learn_fact(
                            episode, target_answer_node,
                            correction_target_id=correction_target_id,
                        )
                        self.agent.observer.boost_id(fact_id, gain=reinforcement_gain)
                        if predicted_answer_node is not None and predicted_answer_node != target_answer_node:
                            self.agent.observer.penalize_id(
                                predicted_answer_node, penalty=rejection_penalty
                            )
            else:
                abstention_node_id, abstention_pattern_created = self._learn_abstention(episode)
                self.agent.observer.boost_id(abstention_node_id, gain=reinforcement_gain)
            if predicted is not None and predicted != episode.evidence_id:
                # Abstention is query-scoped evidence. Do not reduce a source's
                # global utility because one unsupported question matched it.
                if episode.evidence_id is not None:
                    self.agent.observer.penalize_id(
                        self.source_to_node[predicted], penalty=rejection_penalty
                    )
            self._log({
                "event": "supervision",
                "episode_id": episode.episode_id,
                "split": episode.split,
                "predicted_evidence_id": predicted,
                "gold_evidence_id": episode.evidence_id,
                "correct": correct,
                "correction_of": episode.correction_of,
                "correction_target_node_id": correction_target_id,
                "correction_link_added": correction_added,
                "reinforcement_gain": reinforcement_gain,
                "rejection_penalty": rejection_penalty,
                "user_id": episode.user_id,
                "fact_node_id": fact_id if episode.evidence_id is not None and episode.expected_answer and target_answer_node is not None else None,
                "fact_node_created": bool(fact_created) if episode.evidence_id is not None and episode.expected_answer and target_answer_node is not None else False,
                "fact_support_added": bool(support_added) if episode.evidence_id is not None and episode.expected_answer and target_answer_node is not None else False,
                "abstention_pattern_node_id": abstention_node_id,
                "abstention_pattern_created": abstention_pattern_created,
            })

        if validation:
            self.abstain_threshold = self._calibrate_threshold(validation)
        validation_metrics = self._evaluate_validation_fit(validation)
        validation_promotions = self._validate_fact_patterns(validation)
        self._fit_episode_ids.update(ep.episode_id for ep in train + validation)
        knowledge_growth = {
            "new_fact_nodes": len(set(self.fact_node_data) - facts_before),
            "total_fact_nodes": len(self.fact_node_data),
            "source_support_edges_added": (
                sum(len(data["answer_sentence_ids"]) for data in self.fact_node_data.values())
                - support_edges_before
            ),
            "correction_links_added": (
                sum(len(data.get("corrects_node_ids", ())) for data in self.fact_node_data.values())
                - correction_links_before
            ),
            "validated_fact_nodes": sum(data["status"] == "validated" for data in self.fact_node_data.values()),
            "promoted_cross_source_fact_nodes": sum(data["status"] == "promoted" for data in self.fact_node_data.values()),
            "validation_promotions": validation_promotions,
            "new_abstention_patterns": len(set(self.abstention_node_data) - abstentions_before),
            "total_abstention_patterns": len(self.abstention_node_data),
        }
        return {
            "train_episodes": len(train),
            "validation_episodes": len(validation),
            "source_count": len(self.sources),
            "abstain_threshold": self.abstain_threshold,
            "abstention_similarity_threshold": self.abstention_similarity_threshold,
            "validation_metrics": validation_metrics,
            "knowledge_growth": knowledge_growth,
        }

    def _evaluate_validation_fit(self, episodes: list[StudyEpisode]) -> dict[str, Any]:
        """Report calibration-set behavior separately from held-out evaluation.

        These are resubstitution diagnostics because the same validation labels
        choose the abstention threshold. They expose calibration failures but
        must not be presented as independent generalization metrics.
        """
        if not episodes:
            return {"episode_count": 0, "resubstitution": True}
        predictions = [
            self.predict_question(
                ep.question, ep.episode_id, ep.task_type, user_id=ep.user_id
            )
            for ep in episodes
        ]
        correct = sum(
            prediction.selected_evidence_id == episode.evidence_id
            for episode, prediction in zip(episodes, predictions)
        )
        answerable = [
            (episode, prediction)
            for episode, prediction in zip(episodes, predictions)
            if episode.evidence_id is not None
        ]
        unsupported = [
            prediction
            for episode, prediction in zip(episodes, predictions)
            if episode.evidence_id is None
        ]
        evidence_correct = sum(
            prediction.selected_evidence_id == episode.evidence_id
            for episode, prediction in answerable
        )
        return {
            "episode_count": len(episodes),
            "resubstitution": True,
            "accuracy": correct / len(episodes),
            "answerable_evidence_accuracy": (
                evidence_correct / len(answerable) if answerable else None
            ),
            "unsupported_rejection_rate": (
                sum(prediction.abstained for prediction in unsupported) / len(unsupported)
                if unsupported else None
            ),
            "false_abstention_rate": (
                sum(prediction.abstained for _episode, prediction in answerable) / len(answerable)
                if answerable else None
            ),
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "expected_evidence_id": episode.evidence_id,
                    "selected_evidence_id": prediction.selected_evidence_id,
                    "confidence": prediction.confidence,
                    "abstained": prediction.abstained,
                }
                for episode, prediction in zip(episodes, predictions)
            ],
        }

    def fit_phase(
        self,
        sources: Iterable[SourceDocument],
        train_episodes: Iterable[StudyEpisode],
        validation_episodes: Iterable[StudyEpisode] = (),
        *,
        retention_episodes: Iterable[StudyEpisode],
        max_accuracy_drop: float = 0.05,
    ) -> dict[str, Any]:
        """Train one curriculum phase and measure retained-task regression.

        The retention suite is a held-out ``test`` split and is scored both
        before and after this phase. Its labels are never used for updates or
        validation calibration. Call this once per curriculum phase, reusing
        the same suite to measure cumulative interference.
        """
        if not 0.0 <= max_accuracy_drop <= 1.0:
            raise ValueError("max_accuracy_drop must be between 0 and 1")
        retention = list(retention_episodes)
        self._validate_split(retention, "test")
        if not retention:
            raise ValueError("retention evaluation requires at least one episode")
        # Make new phase sources available before taking the pre-phase score.
        self.ingest(sources)
        before = self.evaluate(retention)
        fit_result = self.fit([], train_episodes, validation_episodes)
        after = self.evaluate(retention)
        accuracy_delta = after["accuracy"] - before["accuracy"]
        return {
            "fit": fit_result,
            "retention_before": before,
            "retention_after": after,
            "retention_accuracy_delta": accuracy_delta,
            "retention_max_accuracy_drop": max_accuracy_drop,
            "retention_gate_passed": accuracy_delta >= -max_accuracy_drop,
        }

    def fit_curriculum(
        self,
        phases: Iterable[dict[str, Any]],
        *,
        retention_episodes: Iterable[StudyEpisode],
        max_accuracy_drop: float = 0.05,
    ) -> dict[str, Any]:
        """Train an ordered, cumulative curriculum and audit retention per phase.

        Each phase is a mapping with ``name``, ``sources``, ``train_episodes``,
        and optional ``validation_episodes``. The same held-out test suite is
        evaluated before and after every phase; its labels never enter fitting.
        The trainer and HFN state are reused across phases, so later phases
        build on accumulated knowledge rather than starting from a fresh model.
        """
        phase_items = list(phases)
        if not phase_items:
            raise ValueError("curriculum requires at least one training phase")
        retention = list(retention_episodes)
        self._validate_split(retention, "test")
        if not retention:
            raise ValueError("retention evaluation requires at least one episode")
        if not 0.0 <= max_accuracy_drop <= 1.0:
            raise ValueError("max_accuracy_drop must be between 0 and 1")

        reports = []
        names: set[str] = set()
        for index, phase in enumerate(phase_items, start=1):
            name = str(phase.get("name") or f"phase-{index}")
            if name in names:
                raise ValueError(f"duplicate curriculum phase name: {name}")
            names.add(name)
            if "train_episodes" not in phase:
                raise ValueError(f"curriculum phase {name!r} is missing train_episodes")
            result = self.fit_phase(
                phase.get("sources", ()),
                phase["train_episodes"],
                phase.get("validation_episodes", ()),
                retention_episodes=retention,
                max_accuracy_drop=max_accuracy_drop,
            )
            reports.append({"name": name, **result})
            self._log({
                "event": "curriculum_phase_complete",
                "phase": name,
                "phase_index": index,
                "retention_gate_passed": result["retention_gate_passed"],
                "retention_accuracy_delta": result["retention_accuracy_delta"],
                "knowledge_growth": result["fit"]["knowledge_growth"],
            })

        return {
            "phase_count": len(reports),
            "phases": reports,
            "all_retention_gates_passed": all(
                report["retention_gate_passed"] for report in reports
            ),
            "retention_episode_count": len(retention),
            "total_fact_nodes": len(self.fact_node_data),
            "validated_fact_nodes": sum(
                data["status"] in {"validated", "promoted"}
                for data in self.fact_node_data.values()
            ),
        }

    def _learn_fact(
        self,
        episode: StudyEpisode,
        answer_node_id: str,
        *,
        correction_target_id: Optional[str] = None,
    ) -> tuple[str, bool, bool, bool]:
        """Create/reinforce a source-linked fact pattern from a supervised answer."""
        normalized = " ".join((episode.expected_answer or "").casefold().split())
        fact_key = f"{episode.user_id}\0{normalized}"
        fact_id = self.fact_key_to_node.get(fact_key)
        created = fact_id is None
        if created:
            fact_id = f"study_fact_{hashlib.sha256(fact_key.encode()).hexdigest()[:16]}"
            answer_node = self.agent.forest.get(answer_node_id)
            node = HFN(
                mu=answer_node.mu.copy(),
                sigma=np.ones(self.agent.m_dim) * 0.2,
                id=fact_id,
                use_diag=True,
            )
            node.relation_type = "study_fact"
            node.metadata = {
                "type": "study_fact",
                "text": episode.expected_answer,
                "user_id": episode.user_id,
                "status": "candidate",
                "correction_of": episode.correction_of,
            }
            self.agent.observer.register(node, protected=False, initial_weight=0.5)
            self.agent.patterns[fact_id] = node
            self.fact_key_to_node[fact_key] = fact_id
            self.fact_node_data[fact_id] = {
                "fact_key": fact_key,
                "text": episode.expected_answer,
                "user_id": episode.user_id,
                "status": "candidate",
                "source_ids": [],
                "answer_sentence_ids": [],
                "validation_episode_ids": [],
                "corrects_node_ids": [],
                "correction_episode_ids": [],
            }
        data = self.fact_node_data[fact_id]
        data.setdefault("corrects_node_ids", [])
        data.setdefault("correction_episode_ids", [])
        data.setdefault("source_ids", [])
        data.setdefault("answer_sentence_ids", [])
        data.setdefault("validation_episode_ids", [])
        support_added = answer_node_id not in data["answer_sentence_ids"]
        correction_added = False
        if support_added:
            answer_node = self.agent.forest.get(answer_node_id)
            node = self.agent.forest.get(fact_id)
            node.add_child(answer_node)
            node.add_edge(node, answer_node, "supported_by")
            data["answer_sentence_ids"].append(answer_node_id)
        if episode.evidence_id not in data["source_ids"]:
            data["source_ids"].append(episode.evidence_id)
        if correction_target_id is not None:
            if correction_target_id not in data["corrects_node_ids"]:
                node = self.agent.forest.get(fact_id)
                incorrect_node = self.agent.forest.get(correction_target_id)
                if incorrect_node is not None:
                    node.add_edge(node, incorrect_node, "corrects")
                    data["corrects_node_ids"].append(correction_target_id)
                    correction_added = True
            if episode.episode_id not in data["correction_episode_ids"]:
                data["correction_episode_ids"].append(episode.episode_id)
            node = self.agent.forest.get(fact_id)
            node.metadata["correction_of"] = episode.correction_of
            node.metadata["corrects_node_ids"] = list(data["corrects_node_ids"])
        self.source_to_fact_nodes.setdefault(episode.evidence_id, [])
        if fact_id not in self.source_to_fact_nodes[episode.evidence_id]:
            self.source_to_fact_nodes[episode.evidence_id].append(fact_id)
        return fact_id, created, support_added, correction_added

    def _learn_abstention(self, episode: StudyEpisode) -> tuple[str, bool]:
        """Store a user-scoped HFN pattern for a supervised unsupported request."""
        normalized = " ".join(episode.question.casefold().split())
        key = f"{episode.user_id}\0{normalized}"
        node_id = f"study_abstain_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
        created = node_id not in self.abstention_node_data
        if created:
            node = HFN(
                mu=self.agent.config.encode_passage(episode.question),
                sigma=np.ones(self.agent.m_dim) * 0.2,
                id=node_id,
                use_diag=True,
            )
            node.relation_type = "study_abstention"
            node.metadata = {
                "type": "study_abstention",
                "user_id": episode.user_id,
                "status": "candidate",
                "episode_ids": [],
            }
            self.agent.observer.register(node, protected=False, initial_weight=0.5)
            self.agent.patterns[node_id] = node
            self.abstention_node_data[node_id] = {
                "user_id": episode.user_id,
                "question": episode.question,
                "episode_ids": [],
            }
        data = self.abstention_node_data[node_id]
        if episode.episode_id not in data["episode_ids"]:
            data["episode_ids"].append(episode.episode_id)
        node = self.agent.forest.get(node_id)
        node.metadata["episode_ids"] = list(data["episode_ids"])
        node.metadata["status"] = "supervised_negative"
        return node_id, created

    def _validate_fact_patterns(self, episodes: list[StudyEpisode]) -> int:
        """Promote facts only after held-out answer and provenance checks pass."""
        promotions = 0
        for episode in episodes:
            if not episode.expected_answer or not episode.evidence_id:
                continue
            predicted = self.predict_question(
                episode.question, episode.episode_id, episode.task_type,
                user_id=episode.user_id,
            )
            if predicted.selected_evidence_id != episode.evidence_id or not predicted.answer:
                continue
            if _token_f1(episode.expected_answer, predicted.answer) < 0.8:
                continue
            normalized = " ".join(episode.expected_answer.casefold().split())
            fact_key = f"{episode.user_id}\0{normalized}"
            fact_id = self.fact_key_to_node.get(fact_key)
            if fact_id is None:
                continue
            data = self.fact_node_data[fact_id]
            if episode.episode_id not in data["validation_episode_ids"]:
                data["validation_episode_ids"].append(episode.episode_id)
            status = "promoted" if len(set(data["source_ids"])) >= 2 else "validated"
            if data["status"] != status:
                data["status"] = status
                node = self.agent.forest.get(fact_id)
                node.metadata["status"] = status
                promotions += 1
            self._log({
                "event": "fact_validation",
                "episode_id": episode.episode_id,
                "fact_node_id": fact_id,
                "status": data["status"],
                "answer_token_f1": _token_f1(episode.expected_answer, predicted.answer),
            })
        return promotions

    def _fact_utility(self, source_id: str, answer_node_id: str) -> float:
        for fact_id in self.source_to_fact_nodes.get(source_id, ()):
            data = self.fact_node_data[fact_id]
            if answer_node_id not in data["answer_sentence_ids"]:
                continue
            weight = float(self.agent.observer.get_weight(fact_id))
            level = 0.20 if data["status"] == "promoted" else 0.10 if data["status"] == "validated" else 0.05
            return 1.0 + level * weight
        return 1.0

    def _calibrate_threshold(self, episodes: list[StudyEpisode]) -> float:
        scored = [(self._rank(ep.question), ep) for ep in episodes]
        candidates = {0.0}
        candidates.update(rank[0][1] for rank, _ in scored if rank)
        candidates.update(math.nextafter(rank[0][1], math.inf) for rank, _ in scored if rank)
        best_threshold, best_accuracy = 0.0, -1.0
        for threshold in sorted(candidates):
            correct = 0
            for rank, episode in scored:
                confidence = rank[0][1] if rank else 0.0
                predicted = rank[0][0] if rank and confidence >= threshold else None
                correct += predicted == episode.evidence_id
            accuracy = correct / len(scored)
            # Candidates are ascending; keep the lower threshold on ties to
            # avoid needless abstention when validation gives no evidence that
            # the higher threshold improves correctness.
            if accuracy > best_accuracy:
                best_threshold, best_accuracy = threshold, accuracy
        return best_threshold

    def evaluate(self, episodes: Iterable[StudyEpisode]) -> dict[str, Any]:
        """Read-only evaluation. Returns accuracy, evidence quality, and per-task metrics."""
        items = list(episodes)
        self._validate_split(items, "test")
        overlap = {ep.episode_id for ep in items} & self._fit_episode_ids
        if overlap:
            raise ValueError(f"test episode IDs appeared during fit: {sorted(overlap)[:3]}")
        if not items:
            raise ValueError("evaluation requires at least one episode")
        metrics: dict[str, dict[str, int]] = {}
        predictions = []
        latencies_ms = []
        for episode in items:
            self._validate_evidence(episode)
            started = time.perf_counter()
            prediction = self.predict(episode)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            predictions.append(prediction)
            bucket = metrics.setdefault(episode.task_type, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(prediction.selected_evidence_id == episode.evidence_id)
        correct = sum(v["correct"] for v in metrics.values())
        answerable = [ep for ep in items if ep.evidence_id is not None]
        unsupported = [ep for ep in items if ep.evidence_id is None]
        evidence_correct = sum(
            p.selected_evidence_id == ep.evidence_id
            for ep, p in zip(items, predictions)
            if ep.evidence_id is not None
        )
        return {
            "accuracy": correct / len(items),
            "accuracy_ci95": _wilson_interval(correct, len(items)),
            "correct": correct,
            "total": len(items),
            "evidence_accuracy": evidence_correct / len(answerable) if answerable else None,
            "evidence_accuracy_ci95": (
                _wilson_interval(evidence_correct, len(answerable)) if answerable else None
            ),
            "answerable_total": len(answerable),
            "abstention_rate": sum(p.abstained for p in predictions) / len(items),
            "unsupported_rejection_rate": (
                sum(p.abstained for ep, p in zip(items, predictions) if ep.evidence_id is None)
                / len(unsupported) if unsupported else None
            ),
            "false_abstention_rate": (
                sum(p.abstained for ep, p in zip(items, predictions) if ep.evidence_id is not None)
                / len(answerable) if answerable else None
            ),
            "median_latency_ms": float(np.median(latencies_ms)),
            "answer_reference_count": sum(
                ep.expected_answer is not None and ep.evidence_id is not None for ep in items
            ),
            "answer_token_f1": _mean_answer_f1(items, predictions),
            "answer_exact_match": _answer_exact_match(items, predictions),
            "episode_results": [
                {
                    "episode_id": episode.episode_id,
                    "task_type": episode.task_type,
                    "expected_evidence_id": episode.evidence_id,
                    "selected_evidence_id": prediction.selected_evidence_id,
                    "confidence": prediction.confidence,
                    "abstained": prediction.abstained,
                    "answer": prediction.answer,
                    "answer_token_f1": (
                        _token_f1(episode.expected_answer, prediction.answer or "")
                        if episode.expected_answer is not None and episode.evidence_id is not None
                        else None
                    ),
                    "answer_exact_match": (
                        _answer_exact_match([episode], [prediction])
                        if episode.expected_answer is not None and episode.evidence_id is not None
                        else None
                    ),
                    "citation": prediction.citation,
                    "citations": prediction.citations,
                    "abstention_reason": prediction.abstention_reason,
                    "latency_ms": latency_ms,
                }
                for episode, prediction, latency_ms in zip(items, predictions, latencies_ms)
            ],
            "per_task_type": {
                task: _task_metrics(task, counts, items, predictions)
                for task, counts in sorted(metrics.items())
            },
        }

    def save_checkpoint(self, directory: str | Path) -> Path:
        """Persist ReaderAgent state and the trainer's source/evidence mapping."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.agent.save_agent(str(path))
        state = {
            "storage_version": self.storage_version,
            "abstain_threshold": self.abstain_threshold,
            "abstention_similarity_threshold": self.abstention_similarity_threshold,
            "sources": [asdict(source) for source in self.sources.values()],
            "source_to_node": self.source_to_node,
            "source_to_answer_nodes": self.source_to_answer_nodes,
            "answer_node_data": self.answer_node_data,
            "fact_key_to_node": self.fact_key_to_node,
            "fact_node_data": self.fact_node_data,
            "source_to_fact_nodes": self.source_to_fact_nodes,
            "abstention_node_data": self.abstention_node_data,
            "feedback_gain": self.feedback_gain,
            "feedback_penalty": self.feedback_penalty,
            "correction_gain": self.correction_gain,
            "correction_penalty": self.correction_penalty,
            "semantic_lexical_bonus": self.semantic_lexical_bonus,
            "semantic_lexical_rescue_threshold": self.semantic_lexical_rescue_threshold,
            "semantic_lexical_rescue_margin": self.semantic_lexical_rescue_margin,
            "observer_weights": {
                node_id: self.agent.observer.get_weight(node_id)
                for node_id in (
                    *self.node_to_source, *self.answer_node_data, *self.fact_node_data,
                    *self.abstention_node_data,
                )
            },
            "fit_episode_ids": sorted(self._fit_episode_ids),
        }
        state_path = path / "study_training_state.json"
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state_path

    def load_checkpoint_state(self, directory: str | Path) -> None:
        """Restore trainer metadata and supervised weights into a loaded ReaderAgent.

        Call this after ``ReaderAgent.load_agent(directory)`` so the forest and
        Observer nodes already exist.
        """
        state_path = Path(directory) / "study_training_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        sources = [SourceDocument(**row) for row in state["sources"]]
        if len(self.agent.config._passages) < len(sources):
            raise ValueError("loaded ReaderAgent is missing passages from trainer checkpoint")
        self.sources = {source.source_id: source for source in sources}
        self.source_to_node = dict(state["source_to_node"])
        self.node_to_source = {node: source for source, node in self.source_to_node.items()}
        self.source_to_answer_nodes = {
            source: list(nodes) for source, nodes in state.get("source_to_answer_nodes", {}).items()
        }
        self.answer_node_data = dict(state.get("answer_node_data", {}))
        self.fact_key_to_node = dict(state.get("fact_key_to_node", {}))
        self.fact_node_data = dict(state.get("fact_node_data", {}))
        self.source_to_fact_nodes = {
            source: list(nodes) for source, nodes in state.get("source_to_fact_nodes", {}).items()
        }
        self.abstention_node_data = dict(state.get("abstention_node_data", {}))
        if set(self.source_to_node) != set(self.sources):
            raise ValueError("checkpoint source map does not match stored sources")
        for node_id, weight in state.get("observer_weights", {}).items():
            self.agent.observer.set_weight(node_id, float(weight))
        self.feedback_gain = float(state.get("feedback_gain", self.feedback_gain))
        self.feedback_penalty = float(state.get("feedback_penalty", self.feedback_penalty))
        self.correction_gain = float(state.get("correction_gain", self.correction_gain))
        self.correction_penalty = float(state.get("correction_penalty", self.correction_penalty))
        self.semantic_lexical_bonus = float(
            state.get("semantic_lexical_bonus", self.semantic_lexical_bonus)
        )
        self.semantic_lexical_rescue_threshold = float(
            state.get("semantic_lexical_rescue_threshold", self.semantic_lexical_rescue_threshold)
        )
        self.semantic_lexical_rescue_margin = float(
            state.get("semantic_lexical_rescue_margin", self.semantic_lexical_rescue_margin)
        )
        self.abstention_similarity_threshold = float(
            state.get("abstention_similarity_threshold", self.abstention_similarity_threshold)
        )
        self.abstain_threshold = float(state.get("abstain_threshold", 0.0))
        self.storage_version = int(state.get("storage_version", 1))
        self._fit_episode_ids = set(state.get("fit_episode_ids", []))

    def _validate_evidence(self, episode: StudyEpisode) -> None:
        if episode.evidence_id is not None and episode.evidence_id not in self.source_to_node:
            raise ValueError(f"episode {episode.episode_id} references unknown evidence {episode.evidence_id}")

    @staticmethod
    def _validate_split(episodes: list[StudyEpisode], expected: Split) -> None:
        wrong = [ep.episode_id for ep in episodes if ep.split != expected]
        if wrong:
            raise ValueError(f"expected {expected} episodes, got mismatched splits: {wrong[:3]}")
        ids = [ep.episode_id for ep in episodes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate episode ids in {expected} split")

    def _log(self, event: dict[str, Any]) -> None:
        self._event_count += 1
        if self.event_log is None:
            return
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": time.time(), **event}
        with self.event_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split source text while retaining exact citation offsets."""
    result = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        sentence = raw.strip()
        if sentence:
            start = match.start() + left
            result.append((sentence, start, start + len(sentence)))
    if not result and text.strip():
        start = len(text) - len(text.lstrip())
        sentence = text.strip()
        result.append((sentence, start, start + len(sentence)))
    return result


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a binomial proportion, returned as [low, high]."""
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _token_f1(reference: str, candidate: str) -> float:
    expected = Counter(tokenise(reference))
    actual = Counter(tokenise(candidate))
    if not expected or not actual:
        return float(expected == actual)
    overlap = sum((expected & actual).values())
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean_answer_f1(episodes: list[StudyEpisode], predictions: list[StudyPrediction]) -> Optional[float]:
    pairs = [
        (episode.expected_answer, prediction.answer)
        for episode, prediction in zip(episodes, predictions)
        if episode.expected_answer is not None and episode.evidence_id is not None
    ]
    if not pairs:
        return None
    return sum(_token_f1(expected, answer or "") for expected, answer in pairs) / len(pairs)


def _task_metrics(task, counts, episodes, predictions):
    pairs = [
        (episode.expected_answer, prediction.answer)
        for episode, prediction in zip(episodes, predictions)
        if episode.task_type == task
        and episode.expected_answer is not None
        and episode.evidence_id is not None
    ]
    return {
        **counts,
        "accuracy": counts["correct"] / counts["total"],
        "accuracy_ci95": _wilson_interval(counts["correct"], counts["total"]),
        "answer_reference_count": len(pairs),
        "answer_token_f1": (
            sum(_token_f1(expected, answer or "") for expected, answer in pairs) / len(pairs)
            if pairs else None
        ),
    }


def _answer_exact_match(episodes: list[StudyEpisode], predictions: list[StudyPrediction]) -> Optional[float]:
    pairs = [
        (episode.expected_answer, prediction.answer)
        for episode, prediction in zip(episodes, predictions)
        if episode.expected_answer is not None and episode.evidence_id is not None
    ]
    if not pairs:
        return None
    normalize = lambda text: " ".join(tokenise(text))
    return sum(normalize(expected) == normalize(answer or "") for expected, answer in pairs) / len(pairs)
