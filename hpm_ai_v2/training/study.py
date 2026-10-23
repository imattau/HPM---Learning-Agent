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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.domains.text_domain import tokenise


Split = Literal["train", "validation", "test"]
TASK_TYPES = {"grounded_qa", "abstain", "correction", "define", "summarize", "outline", "flashcards", "preference", "procedure", "compare"}
MULTI_SENTENCE_TASKS = {"summarize", "outline", "flashcards"}
QUERY_SCOPED_FACT_TASKS = {"grounded_qa", "define", "summarize", "outline", "flashcards"}
CORRECTION_QUERY_SIMILARITY_THRESHOLD = 0.30
CORRECTION_ANSWER_SIMILARITY_THRESHOLD = 0.10
CORRECTION_TRANSFER_SCORE_THRESHOLD = 0.11
FACT_RETRIEVAL_SIMILARITY_THRESHOLD = 0.30
CONCEPT_RETRIEVAL_SIMILARITY_THRESHOLD = 0.30
SUPPORTED_PROCEDURES = {"define", "summarize", "outline", "flashcards"}


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

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
    preference_key: Optional[str] = None
    concept_key: Optional[str] = None
    workflow_key: Optional[str] = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task_type: {self.task_type}")
        if self.task_type == "compare":
            if self.evidence_id is not None:
                raise ValueError("compare episodes use evidence_ids, not evidence_id")
            if len(self.evidence_ids) < 2 or len(set(self.evidence_ids)) != len(self.evidence_ids):
                raise ValueError("compare episodes require at least two distinct evidence_ids")
        elif self.evidence_ids:
            raise ValueError("evidence_ids is only valid for compare episodes")
        if self.task_type in {"abstain", "preference", "procedure", "compare"} and self.evidence_id is not None:
            raise ValueError("abstain episodes must not name evidence")
        if self.task_type not in {"abstain", "preference", "procedure", "compare"} and self.evidence_id is None:
            raise ValueError("answerable episodes must name evidence")
        if self.task_type == "preference":
            if not self.preference_key or not self.preference_key.strip():
                raise ValueError("preference episodes require preference_key")
            if not self.expected_answer or not self.expected_answer.strip():
                raise ValueError("preference episodes require expected_answer as the preference value")
        elif self.preference_key is not None:
            raise ValueError("preference_key is only valid for preference episodes")
        if self.task_type == "procedure":
            if self.workflow_key not in SUPPORTED_PROCEDURES:
                raise ValueError(f"procedure workflow_key must be one of {sorted(SUPPORTED_PROCEDURES)}")
        elif self.workflow_key is not None:
            raise ValueError("workflow_key is only valid for procedure episodes")
        if self.concept_key is not None:
            if not self.concept_key.strip():
                raise ValueError("concept_key must not be empty")
            if self.evidence_id is None or not self.expected_answer:
                raise ValueError("concept supervision requires evidence_id and expected_answer")


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
    selected_evidence_ids: tuple[str, ...] = ()


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
        fact_retrieval_similarity_threshold: float = FACT_RETRIEVAL_SIMILARITY_THRESHOLD,
        concept_retrieval_similarity_threshold: float = CONCEPT_RETRIEVAL_SIMILARITY_THRESHOLD,
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
        if not 0.0 <= fact_retrieval_similarity_threshold <= 1.0:
            raise ValueError("fact_retrieval_similarity_threshold must be between 0 and 1")
        if not 0.0 <= concept_retrieval_similarity_threshold <= 1.0:
            raise ValueError("concept_retrieval_similarity_threshold must be between 0 and 1")
        self.semantic_lexical_bonus = float(semantic_lexical_bonus)
        self.semantic_lexical_rescue_threshold = float(semantic_lexical_rescue_threshold)
        self.semantic_lexical_rescue_margin = float(semantic_lexical_rescue_margin)
        self.abstention_similarity_threshold = float(abstention_similarity_threshold)
        self.fact_retrieval_similarity_threshold = float(fact_retrieval_similarity_threshold)
        self.concept_retrieval_similarity_threshold = float(concept_retrieval_similarity_threshold)
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
        self.concept_node_data: dict[str, dict[str, Any]] = {}
        self.concept_key_to_node: dict[str, str] = {}
        self.abstention_node_data: dict[str, dict[str, Any]] = {}
        self.preference_node_data: dict[str, dict[str, Any]] = {}
        self.preference_key_to_node: dict[str, str] = {}
        self.procedure_node_data: dict[str, dict[str, Any]] = {}
        self.procedure_key_to_node: dict[str, str] = {}
        self.procedure_similarity_threshold = 0.50
        self.correction_query_similarity_threshold = CORRECTION_QUERY_SIMILARITY_THRESHOLD
        self.correction_answer_similarity_threshold = CORRECTION_ANSWER_SIMILARITY_THRESHOLD
        self.correction_transfer_score_threshold = CORRECTION_TRANSFER_SCORE_THRESHOLD
        self._event_count = 0
        self._fit_episode_ids: set[str] = set()

    def ingest(self, sources: Iterable[SourceDocument]) -> None:
        """Register each source once as a passage pattern in the HFN forest."""
        sources = list(sources)
        forest = self.agent.forest
        bulk_registration = getattr(forest, "bulk_registration", None)
        if bulk_registration is not None:
            with bulk_registration():
                for source in sources:
                    self._ingest_source(source)
            meta_forest = self.agent.observer.meta_forest
            if hasattr(meta_forest, "rebuild_hierarchy_cache"):
                meta_forest.rebuild_hierarchy_cache()
            self._link_conversation_sources()
            return
        for source in sources:
            self._ingest_source(source)
        self._link_conversation_sources()

    def _link_conversation_sources(self) -> None:
        """Connect imported conversation messages by their preserved parent IDs."""
        row_sources: dict[tuple[str, str, str, str], str] = {}
        for source_id, source in self.sources.items():
            metadata = source.metadata
            row_id = metadata.get("huggingface_row_id")
            if not row_id:
                continue
            key = (
                str(metadata.get("huggingface_dataset_id", "")),
                str(metadata.get("huggingface_revision", "")),
                str(metadata.get("huggingface_split", "")),
                str(row_id),
            )
            row_sources[key] = source_id
        if not row_sources:
            return
        for source_id, source in self.sources.items():
            metadata = source.metadata
            parent_id = metadata.get("parent_id")
            if not parent_id or not metadata.get("huggingface_row_id"):
                continue
            parent_key = (
                str(metadata.get("huggingface_dataset_id", "")),
                str(metadata.get("huggingface_revision", "")),
                str(metadata.get("huggingface_split", "")),
                str(parent_id),
            )
            parent_source_id = row_sources.get(parent_key)
            if parent_source_id is None:
                continue
            child_node = self.agent.forest.get(self.source_to_node[source_id])
            parent_node = self.agent.forest.get(self.source_to_node[parent_source_id])
            if child_node is None or parent_node is None:
                continue
            if not any(
                edge.source.id == child_node.id and edge.target.id == parent_node.id
                and edge.relation == "conversation_reply_to"
                for edge in child_node.edges()
            ):
                child_node.add_edge(child_node, parent_node, "conversation_reply_to")

    def _ingest_source(self, source: SourceDocument) -> None:
        if source.source_id in self.sources:
            if self.sources[source.source_id] != source:
                raise ValueError(f"source_id reused with different content: {source.source_id}")
            return
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

    def _rank_answers(
        self, question: str, source_id: str, *, user_id: str = "default"
    ) -> list[tuple[str, float]]:
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
            fact_utility = self._fact_utility(
                source_id, node_id, question=question, user_id=user_id
            )
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

    def _rank(
        self, question: str, k: int = 10, *, user_id: str = "default"
    ) -> list[tuple[str, float]]:
        """Rank evidence using passages and the requesting user's learned facts."""
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
                    if data.get("user_id", "default") != user_id:
                        continue
                    fact_node = self.agent.forest.get(fact_id)
                    if fact_node is None:
                        continue
                    fact_vec = fact_node.mu[s_dim : s_dim + dim]
                    denom = query_norm * float(np.linalg.norm(fact_vec)) + 1e-9
                    similarity = float(np.dot(query_vec, fact_vec) / denom)
                    is_correction = bool(data.get("corrects_node_ids"))
                    if is_correction:
                        # Keep correction retrieval bound to both the taught
                        # query pattern and its corrected proposition.
                        similarity = self._correction_query_similarity(question, data)
                        answer_similarity = self._correction_answer_similarity(question, data)
                        if not self._correction_transfer_matches(similarity, answer_similarity):
                            continue
                    elif similarity < self.fact_retrieval_similarity_threshold:
                        # A single shared topic word can make a stored answer
                        # look relevant to a different fact (for example,
                        # "heat" in a latent-heat question and a conduction
                        # answer). Require a stronger match before a fact may
                        # move source rankings.
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
                            base_score = scores_by_source.get(source_id, 0.0)
                            scores_by_source[source_id] = min(
                                1.0,
                                # Correction nodes are query-scoped answer
                                # evidence. They may select a corrected answer
                                # after its source is retrieved, but must not
                                # dominate source retrieval: near-neighbor
                                # corrections otherwise pull later questions
                                # onto an adjacent source.
                                base_score + 0.1 * fact_support,
                            )
                    if is_correction:
                        # Apply correction edges only after the paired query
                        # and answer checks above pass. Keep the effect bounded
                        # so a correction cannot erase unrelated source signal.
                        penalty = min(0.5, 0.5 * fact_support)
                        for wrong_node_id in data.get("corrects_node_ids", ()):
                            wrong_source_id = self.node_to_source.get(wrong_node_id)
                            if wrong_source_id is None:
                                wrong_answer = self.answer_node_data.get(wrong_node_id)
                                if wrong_answer is not None:
                                    wrong_source_id = wrong_answer["source_id"]
                            if wrong_source_id is not None:
                                scores_by_source[wrong_source_id] = (
                                    scores_by_source.get(wrong_source_id, 0.0)
                                    * (1.0 - penalty)
                                )
                concept_bonuses = {
                    "candidate": 0.0,
                    "validated": 0.05,
                    "promoted": 0.08,
                }
                for concept_id, data in self.concept_node_data.items():
                    if data.get("user_id", "default") != user_id:
                        continue
                    concept_node = self.agent.forest.get(concept_id)
                    if concept_node is None:
                        continue
                    concept_vec = concept_node.mu[s_dim : s_dim + dim]
                    denom = query_norm * float(np.linalg.norm(concept_vec)) + 1e-9
                    similarity = float(np.dot(query_vec, concept_vec) / denom)
                    if similarity < self.concept_retrieval_similarity_threshold:
                        continue
                    weight = max(0.0, min(1.0, float(self.agent.observer.get_weight(concept_id))))
                    bonus = concept_bonuses.get(data.get("status", "candidate"), 0.02)
                    for source_id in data.get("source_ids", ()):
                        if source_id in self.source_to_node:
                            scores_by_source[source_id] = min(
                                1.0,
                                scores_by_source.get(source_id, 0.0)
                                + bonus * similarity * weight,
                            )

        # Add a small BM25 corroboration signal to both TF-IDF and dense
        # retrieval. Validation on the cumulative checkpoint showed that the
        # normalized lexical signal improves source selection over the sparse
        # cosine score alone; keeping the weight small preserves HFN ranking.
        if self.semantic_lexical_bonus:
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
        if episode.task_type == "compare":
            claims = self.predict_comparison_sources(
                episode.question, user_id=episode.user_id, limit=2
            )
            evidence_ids = tuple(claim["source_id"] for claim in claims)
            citations = tuple(claim["citation"] for claim in claims)
            return StudyPrediction(
                episode_id=episode.episode_id,
                selected_evidence_id=evidence_ids[0] if evidence_ids else None,
                selected_evidence_ids=evidence_ids,
                confidence=min((claim["confidence"] for claim in claims), default=0.0),
                ranked_evidence=tuple(self._rank(episode.question, user_id=episode.user_id)),
                abstained=len(evidence_ids) < 2,
                answer="\n".join(claim["text"] for claim in claims) if claims else None,
                citation=citations[0] if citations else None,
                citations=citations,
                abstention_reason=(
                    "comparison_requires_two_supported_sources" if len(evidence_ids) < 2 else None
                ),
            )
        if episode.task_type == "preference":
            value = self.get_preference(episode.user_id, episode.preference_key or "")
            return StudyPrediction(
                episode_id=episode.episode_id,
                selected_evidence_id=None,
                confidence=1.0 if value is not None else 0.0,
                ranked_evidence=(),
                abstained=value is None,
                answer=value,
                abstention_reason="preference_not_learned" if value is None else None,
            )
        if episode.task_type == "procedure":
            predicted = self.predict_workflow(episode.question, user_id=episode.user_id, include_candidates=True)
            return StudyPrediction(
                episode_id=episode.episode_id, selected_evidence_id=None,
                confidence=predicted[1] if predicted else 0.0, ranked_evidence=(),
                abstained=predicted is None, answer=predicted[0] if predicted else None,
                abstention_reason="workflow_not_learned" if predicted is None else None,
            )
        return self.predict_question(
            episode.question, episode.episode_id, episode.task_type, user_id=episode.user_id
        )

    def predict_workflow(self, question: str, *, user_id: str = "default", include_candidates: bool = False):
        """Route a phrase to a supervised, validated study workflow."""
        query = self._procedure_vector(question)
        qnorm = float(np.linalg.norm(query))
        if qnorm <= 0:
            return None
        matches = []
        query_tokens = _workflow_tokens(question)
        anchored = []
        for node_id, data in self.procedure_node_data.items():
            if data.get("user_id") != user_id or (not include_candidates and data.get("status") != "validated"):
                continue
            vectors = [np.asarray(v, dtype=float) for v in data.get("trigger_vectors", ())]
            if not vectors:
                vectors = [np.asarray(data.get("centroid", []), dtype=float)]
            score = max(
                (float(np.dot(query, vector) / (qnorm * np.linalg.norm(vector) + 1e-9))
                 for vector in vectors if vector.size),
                default=0.0,
            )
            matches.append((data["workflow_key"], score, node_id))
            if query_tokens & set(data.get("anchor_tokens", ())):
                anchored.append((data["workflow_key"], node_id))
        anchor_keys = {row[0] for row in anchored}
        if len(anchor_keys) == 1:
            key = next(iter(anchor_keys))
            node_id = next(node for workflow, node in anchored if workflow == key)
            return key, 1.0, node_id
        if not matches:
            return None
        matches.sort(key=lambda row: row[1], reverse=True)
        best = matches[0]
        if best[1] < self.procedure_similarity_threshold:
            return None
        if len(matches) > 1 and best[1] - matches[1][1] < 0.03:
            return None
        return best[0], best[1], best[2]

    def get_preference(self, user_id: str, preference_key: str) -> Optional[str]:
        """Return the latest explicitly supervised value for one user's key."""
        node_id = self.preference_key_to_node.get(f"{user_id}\0{preference_key}")
        if node_id is None:
            return None
        data = self.preference_node_data.get(node_id)
        return data.get("value") if data else None

    def _correction_query_similarity(self, question: str, data: dict[str, Any]) -> float:
        """Return the strongest aligned vector match to a stored correction query."""
        encoded = self.agent.config.encode_passage(question)
        s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
        query_vec = encoded[s_dim : s_dim + dim]
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm <= 0.0:
            return 0.0
        query_tokens = set(tokenise(question))
        similarities = []
        pattern_texts = data.get("correction_query_texts", ())
        pattern_vectors = data.get("correction_query_vectors", ())
        for index, pattern_text in enumerate(pattern_texts):
            pattern_tokens = set(tokenise(pattern_text))
            # TF-IDF vectors can put unrelated queries close together in a
            # small or sparse vocabulary. Require shared content terms too.
            if len(query_tokens & pattern_tokens) < 2:
                continue
            # Vectors saved in a checkpoint can predate later TF-IDF
            # vocabulary expansion. Re-encode their durable text here so a
            # stale vector cannot silently disable correction transfer.
            pattern_vec = self.agent.config.encode_passage(pattern_text)[
                s_dim : s_dim + dim
            ]
            denom = query_norm * float(np.linalg.norm(pattern_vec)) + 1e-9
            similarities.append(float(np.dot(query_vec, pattern_vec) / denom))
        # Older checkpoints may have vectors but no durable query text. Keep
        # compatible matching when their stored dimensions still agree.
        if not pattern_texts:
            for pattern in pattern_vectors:
                pattern_vec = np.asarray(pattern, dtype=float)
                if pattern_vec.size != dim:
                    continue
                denom = query_norm * float(np.linalg.norm(pattern_vec)) + 1e-9
                similarities.append(float(np.dot(query_vec, pattern_vec) / denom))
        return max(similarities, default=0.0)

    def _correction_answer_similarity(
        self,
        question: str,
        data: dict[str, Any],
        *,
        answer_node_ids: Iterable[str] | None = None,
    ) -> float:
        """Measure whether a query also matches the corrected proposition.

        Topic similarity alone can transfer one correction to a different
        question about the same source (for example, confusing what a
        seismograph records with what its trace is called). Requiring query
        alignment with the validated answer sentence keeps the correction
        scoped to the proposition that was actually taught.
        """
        encoded = self.agent.config.encode_passage(question)
        s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
        query_vec = encoded[s_dim : s_dim + dim]
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm <= 0.0:
            return 0.0
        candidates = answer_node_ids or data.get("answer_sentence_ids", ())
        similarities = []
        for node_id in candidates:
            if node_id not in self.answer_node_data:
                continue
            node = self.agent.forest.get(node_id)
            if node is None:
                continue
            answer_vec = node.mu[s_dim : s_dim + dim]
            denom = query_norm * float(np.linalg.norm(answer_vec)) + 1e-9
            similarities.append(float(np.dot(query_vec, answer_vec) / denom))
        return max(similarities, default=0.0)

    def _correction_transfer_matches(self, query_similarity: float, answer_similarity: float) -> bool:
        """Require a query match plus minimum alignment with the corrected claim.

        The answer-similarity floor catches neighboring questions about the
        same source. Keeping it modest, and requiring a combined score, allows
        paraphrases without letting topic overlap alone transfer a correction.
        """
        return (
            query_similarity >= self.correction_query_similarity_threshold
            and answer_similarity >= self.correction_answer_similarity_threshold
            and query_similarity * answer_similarity
            >= self.correction_transfer_score_threshold
        )

    def _matching_correction_answer(
        self, question: str, *, user_id: str
    ) -> Optional[tuple[str, str, float]]:
        """Find a validated corrected answer for this user's close paraphrase.

        The correction query is a scoped retrieval key, not a new source of
        evidence: the returned sentence remains attached to its original
        source and citation offsets.
        """
        if not self.fact_node_data:
            return None
        matches: list[tuple[float, str, str]] = []
        for fact_id, data in self.fact_node_data.items():
            if data.get("user_id", "default") != user_id:
                continue
            if not data.get("correction_episode_ids"):
                continue
            if data.get("status") not in {"validated", "promoted"}:
                continue
            answer_nodes = data.get("answer_sentence_ids", ())
            for source_id in data.get("source_ids", ()):
                if source_id not in self.source_to_node:
                    continue
                source_answers = [
                    node_id for node_id in answer_nodes
                    if self.answer_node_data.get(node_id, {}).get("source_id") == source_id
                ]
                if not source_answers:
                    continue
                similarity = self._correction_query_similarity(question, data)
                for answer_id in source_answers:
                    answer_similarity = self._correction_answer_similarity(
                        question, data, answer_node_ids=(answer_id,)
                    )
                    if self._correction_transfer_matches(similarity, answer_similarity):
                        matches.append((similarity * answer_similarity, source_id, answer_id))
        if not matches:
            return None
        similarity, source_id, answer_id = max(matches, key=lambda row: (row[0], row[1], row[2]))
        return source_id, answer_id, similarity

    def predict_comparison_sources(
        self, question: str, *, user_id: str = "default", limit: int = 2
    ) -> tuple[dict[str, Any], ...]:
        """Return exact cited claims from distinct sources for a comparison.

        The path requires at least two sources above the calibrated evidence
        threshold and respects user-scoped learned negative patterns.
        """
        if limit < 2:
            raise ValueError("comparison requires at least two candidate sources")
        if self._matching_abstention_pattern(question, user_id) is not None:
            return ()
        # When the user names documents or topics, treat those names as an
        # explicit comparison constraint. Generic lexical ranking can otherwise
        # fill the second slot with a merely related source (for example, a
        # memory document for "machine learning and neuroscience").
        question_tokens = tokenise(question)
        source_mentions: list[tuple[int, int, str]] = []
        for source_id, source in self.sources.items():
            aliases = (source_id.replace("_", " "), source.title)
            for alias in aliases:
                alias_tokens = tokenise(alias)
                if not alias_tokens:
                    continue
                width = len(alias_tokens)
                source_mentions.extend(
                    (index, width, source_id)
                    for index in range(len(question_tokens) - width + 1)
                    if question_tokens[index:index + width] == alias_tokens
                )
        # Prefer the longest title at an occurrence so "solar power" cannot
        # accidentally select both a "solar" source and a "solar_power" source.
        longest_by_start: dict[int, int] = {}
        for start, width, _source_id in source_mentions:
            longest_by_start[start] = max(longest_by_start.get(start, 0), width)
        named_sources = sorted({
            (start, source_id)
            for start, width, source_id in source_mentions
            if width == longest_by_start[start]
        })
        all_ranked = self._rank(question, k=len(self.sources), user_id=user_id)
        score_by_source = dict(all_ranked)
        explicit_ids = [source_id for _, source_id in named_sources]
        # Explicitly named sources are valid targets even when their title is
        # absent from the passage text and therefore has zero lexical score.
        # They remain grounded because the claim itself is still extracted
        # from, and cited to, that named source.
        for source_id in explicit_ids:
            score_by_source.setdefault(source_id, self.abstain_threshold)
        ranked_ids = [source_id for source_id, _ in all_ranked]
        ordered_ids = explicit_ids + [
            source_id for source_id in ranked_ids if source_id not in explicit_ids
        ]
        ranked = [(source_id, score_by_source[source_id]) for source_id in ordered_ids]
        selected = [
            (source_id, score) for source_id, score in ranked
            if source_id in explicit_ids or score >= self.abstain_threshold
        ][:limit]
        claims = []
        for source_id, score in selected:
            sentence_rows = self._rank_answers(question, source_id, user_id=user_id)
            if not sentence_rows and source_id in explicit_ids:
                # The user explicitly chose this source, but may describe it
                # with a title absent from its text. For a broad comparison,
                # cite its first indexed sentence instead of silently swapping
                # in an unrelated document.
                source_answer_ids = self.source_to_answer_nodes.get(source_id, [])
                if source_answer_ids:
                    sentence_rows = [(source_answer_ids[0], 0.0)]
            if not sentence_rows:
                continue
            node_id = sentence_rows[0][0]
            data = self.answer_node_data[node_id]
            source = self.sources[source_id]
            claims.append({
                "source_id": source_id,
                "confidence": score,
                "text": data["text"],
                "citation": {
                    "source_id": source_id,
                    "title": source.title,
                    "start": data["start"],
                    "end": data["end"],
                    "text": source.text[data["start"]:data["end"]],
                },
            })
        return tuple(claims) if len(claims) >= 2 else ()

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
        ranked = self._rank(question, user_id=user_id)
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
        correction_match = (
            self._matching_correction_answer(question, user_id=user_id)
            if task_type in {"grounded_qa", "correction"} and matched_abstention is None
            else None
        )
        if correction_match is not None and scope_source_id not in (None, correction_match[0]):
            correction_match = None
        if correction_match is not None:
            correction_source, _answer_node_id, correction_similarity = correction_match
            selected = correction_source
            confidence = max(confidence, correction_similarity)
        if selected is not None:
            sentence_rank = self._rank_answers(question, selected, user_id=user_id)
            if correction_match is not None and correction_match[0] == selected:
                corrected_answer_id = correction_match[1]
                sentence_rank = [(corrected_answer_id, correction_match[2])] + [
                    (node_id, score) for node_id, score in sentence_rank
                    if node_id != corrected_answer_id
                ]
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
        best: Optional[tuple[str, float]] = None
        for node_id, data in self.abstention_node_data.items():
            if data.get("user_id", "default") != user_id:
                continue
            node = self.agent.forest.get(node_id)
            if node is None:
                continue
            pattern_vec = node.mu[s_dim : s_dim + dim]
            denom = query_norm * float(np.linalg.norm(pattern_vec)) + 1e-9
            vector_similarity = (
                float(np.dot(query_vec, pattern_vec) / denom)
                if query_norm > 0.0 else 0.0
            )
            query_tokens = set(tokenise(question))
            pattern_tokens = set(data.get("anchor_tokens") or tokenise(data["question"]))
            shared = len(query_tokens & pattern_tokens)
            lexical_similarity = (
                shared / min(len(query_tokens), len(pattern_tokens))
                if shared and query_tokens and pattern_tokens else 0.0
            )
            similarity = max(vector_similarity, lexical_similarity)
            if best is None or similarity > best[1]:
                best = (node_id, similarity)
        if best and best[1] > 0.0 and best[1] >= self.abstention_similarity_threshold:
            return best
        return None

    def _calibrate_abstention_similarity_threshold(
        self, episodes: list[StudyEpisode]
    ) -> float:
        """Choose a negative-pattern threshold from accumulated validation labels."""
        scored: list[tuple[float, StudyEpisode]] = []
        s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
        for episode in episodes:
            query = self.agent.config.encode_passage(episode.question)
            query_vec = query[s_dim : s_dim + dim]
            query_norm = float(np.linalg.norm(query_vec))
            best_similarity = 0.0
            for node_id, data in self.abstention_node_data.items():
                if data.get("user_id", "default") != episode.user_id:
                    continue
                node = self.agent.forest.get(node_id)
                if node is None:
                    continue
                pattern_vec = node.mu[s_dim : s_dim + dim]
                vector_similarity = (
                    float(np.dot(query_vec, pattern_vec)
                          / (query_norm * float(np.linalg.norm(pattern_vec)) + 1e-9))
                    if query_norm > 0.0 else 0.0
                )
                query_tokens = set(tokenise(episode.question))
                pattern_tokens = set(
                    data.get("anchor_tokens") or tokenise(data["question"])
                )
                shared = len(query_tokens & pattern_tokens)
                lexical_similarity = (
                    shared / min(len(query_tokens), len(pattern_tokens))
                    if shared and query_tokens and pattern_tokens else 0.0
                )
                similarity = max(vector_similarity, lexical_similarity)
                best_similarity = max(best_similarity, similarity)
            scored.append((best_similarity, episode))
        if not scored or not any(episode.evidence_id is None for _, episode in scored):
            return self.abstention_similarity_threshold
        if not any(
            similarity > 0.0
            for similarity, episode in scored
            if episode.evidence_id is None
        ):
            return self.abstention_similarity_threshold
        candidates = {0.0, 1.0}
        candidates.update(score for score, _ in scored if score > 0.0)
        candidates.update(
            math.nextafter(score, math.inf) for score, _ in scored if score > 0.0
        )
        best_threshold, best_accuracy = self.abstention_similarity_threshold, -1.0
        # Prefer the stricter cutoff when validation accuracy ties, limiting
        # collateral abstention on unrelated answerable questions.
        for threshold in sorted(candidates, reverse=True):
            correct = 0
            for similarity, episode in scored:
                predicted_abstention = similarity > 0.0 and similarity >= threshold
                correct += predicted_abstention == (episode.evidence_id is None)
            accuracy = correct / len(scored)
            if accuracy > best_accuracy:
                best_threshold, best_accuracy = threshold, accuracy
        # A validation paraphrase shares more wording with its seed than a
        # later unseen paraphrase. Use the validated threshold as an anchor,
        # then allow half-strength overlap for cross-session transfer.
        return best_threshold * 0.5

    def fit(
        self,
        sources: Iterable[SourceDocument],
        train_episodes: Iterable[StudyEpisode],
        validation_episodes: Iterable[StudyEpisode] = (),
        *,
        calibration_episodes: Iterable[StudyEpisode] = (),
    ) -> dict[str, Any]:
        """Ingest the training corpus, apply labeled feedback, and calibrate abstention.

        Validation episodes calibrate abstention and validate source-grounded
        fact/concept patterns; they do not create supervised fact candidates or
        update HFN pattern weights. Test episodes must never be passed here.
        """
        train = list(train_episodes)
        validation = list(validation_episodes)
        calibration_history = list(calibration_episodes)
        self._validate_split(train, "train")
        self._validate_split(validation, "validation")
        self._validate_split(calibration_history, "validation")
        overlap = {ep.episode_id for ep in train} & {ep.episode_id for ep in validation}
        overlap |= ({ep.episode_id for ep in train + validation} & self._fit_episode_ids)
        if overlap:
            raise ValueError(f"episode IDs reused across training/validation: {sorted(overlap)[:3]}")
        train_fact_keys = {
            (ep.user_id, " ".join(ep.expected_answer.casefold().split()))
            for ep in train
            if ep.task_type in {"grounded_qa", "correction"}
            and ep.evidence_id is not None and ep.expected_answer
        }
        validation_fact_keys = {
            (ep.user_id, " ".join(ep.expected_answer.casefold().split()))
            for ep in validation
            if ep.task_type in {"grounded_qa", "correction"}
            and ep.evidence_id is not None and ep.expected_answer
        }
        matched_validation_facts = train_fact_keys & validation_fact_keys
        self.ingest(sources)
        facts_before = set(self.fact_node_data)
        concepts_before = set(self.concept_node_data)
        support_edges_before = sum(len(data["answer_sentence_ids"]) for data in self.fact_node_data.values())
        correction_links_before = sum(len(data.get("corrects_node_ids", ())) for data in self.fact_node_data.values())
        abstentions_before = set(self.abstention_node_data)
        preferences_before = {
            node_id: data.get("value") for node_id, data in self.preference_node_data.items()
        }
        procedures_before = set(self.procedure_node_data)
        for episode in train:
            self._validate_evidence(episode)
            if episode.task_type == "compare":
                ranked = self._rank(
                    episode.question, k=max(2, len(self.sources)), user_id=episode.user_id
                )
                predicted_ids = {source_id for source_id, score in ranked
                                 if score >= self.abstain_threshold}
                for source_id in episode.evidence_ids:
                    self.agent.observer.boost_id(
                        self.source_to_node[source_id], gain=self.feedback_gain
                    )
                    answer_rows = self._rank_answers(
                        episode.question, source_id, user_id=episode.user_id
                    )
                    if answer_rows:
                        self.agent.observer.boost_id(answer_rows[0][0], gain=self.feedback_gain)
                for source_id in predicted_ids - set(episode.evidence_ids):
                    self.agent.observer.penalize_id(
                        self.source_to_node[source_id], penalty=self.feedback_penalty
                    )
                self._log({
                    "event": "comparison_supervision",
                    "episode_id": episode.episode_id,
                    "user_id": episode.user_id,
                    "predicted_evidence_ids": sorted(predicted_ids),
                    "gold_evidence_ids": list(episode.evidence_ids),
                    "correct": set(episode.evidence_ids) <= predicted_ids,
                })
                continue
            if episode.task_type == "procedure":
                node_id, created = self._learn_procedure(episode)
                self.agent.observer.boost_id(node_id, gain=self.feedback_gain)
                self._log({"event": "procedure_learned", "episode_id": episode.episode_id,
                           "node_id": node_id, "workflow_key": episode.workflow_key,
                           "user_id": episode.user_id, "node_created": created})
                continue
            if episode.task_type == "preference":
                preference_id, created, updated, previous = self._learn_preference(episode)
                self.agent.observer.boost_id(preference_id, gain=self.feedback_gain)
                self._log({
                    "event": "preference_learned",
                    "episode_id": episode.episode_id,
                    "user_id": episode.user_id,
                    "preference_key": episode.preference_key,
                    "previous_value": previous,
                    "value": episode.expected_answer,
                    "node_id": preference_id,
                    "node_created": created,
                    "value_updated": updated,
                })
                continue
            ranked = self._rank(episode.question, user_id=episode.user_id)
            predicted = ranked[0][0] if ranked else None
            correct = predicted == episode.evidence_id
            is_correction = episode.task_type == "correction"
            reinforcement_gain = self.correction_gain if is_correction else self.feedback_gain
            rejection_penalty = self.correction_penalty if is_correction else self.feedback_penalty
            source_weight_feedback = episode.task_type not in QUERY_SCOPED_FACT_TASKS
            target_answer_node = None
            fact_id = None
            fact_created = False
            support_added = False
            correction_added = False
            concept_id = None
            concept_created = False
            concept_support_added = False
            abstention_node_id = None
            abstention_pattern_created = False
            predicted_answer_node = None
            if predicted is not None:
                predicted_answers = self._rank_answers(
                    episode.question, predicted, user_id=episode.user_id
                )
                if predicted_answers:
                    predicted_answer_node = predicted_answers[0][0]
            correction_target_id = None
            if episode.task_type == "correction" and predicted is not None:
                if predicted != episode.evidence_id:
                    correction_target_id = self.source_to_node[predicted]
            if episode.evidence_id is not None:
                gold_node = self.source_to_node[episode.evidence_id]
                if not is_correction and source_weight_feedback:
                    self.agent.observer.boost_id(gold_node, gain=reinforcement_gain)
                if episode.expected_answer:
                    target_answer_node = self._gold_answer_node(
                        episode.evidence_id, episode.expected_answer
                    )
                    if target_answer_node is not None:
                        if not is_correction and source_weight_feedback:
                            self.agent.observer.boost_id(target_answer_node, gain=reinforcement_gain)
                        fact_id, fact_created, support_added, correction_added = self._learn_fact(
                            episode, target_answer_node,
                            correction_target_id=correction_target_id,
                        )
                        self.agent.observer.boost_id(fact_id, gain=reinforcement_gain)
                        if episode.concept_key:
                            concept_id, concept_created, concept_support_added = self._learn_concept(
                                episode, fact_id
                            )
                            self.agent.observer.boost_id(concept_id, gain=reinforcement_gain)
                        if (
                            not is_correction and source_weight_feedback
                            and predicted_answer_node is not None
                            and predicted_answer_node != target_answer_node
                        ):
                            self.agent.observer.penalize_id(
                                predicted_answer_node, penalty=rejection_penalty
                            )
            else:
                abstention_node_id, abstention_pattern_created = self._learn_abstention(episode)
                self.agent.observer.boost_id(abstention_node_id, gain=reinforcement_gain)
            if predicted is not None and predicted != episode.evidence_id:
                # Abstention is query-scoped evidence. Do not reduce a source's
                # global utility because one unsupported question matched it.
                if episode.evidence_id is not None and not is_correction and source_weight_feedback:
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
                "concept_node_id": concept_id,
                "concept_node_created": concept_created,
                "concept_support_added": concept_support_added,
                "abstention_pattern_node_id": abstention_node_id,
                "abstention_pattern_created": abstention_pattern_created,
            })

        current_calibration = [
            ep for ep in validation if ep.task_type not in {"preference", "procedure", "compare"}
        ]
        prior_calibration = [
            ep for ep in calibration_history
            if ep.task_type not in {"preference", "procedure", "compare"}
        ]
        current_ids = {ep.episode_id for ep in current_calibration}
        if current_ids & {ep.episode_id for ep in prior_calibration}:
            raise ValueError("current and historical calibration episode IDs overlap")
        calibration_set = [*prior_calibration, *current_calibration]
        if calibration_set:
            self.abstain_threshold = self._calibrate_threshold(calibration_set)
            if any(episode.evidence_id is None for episode in calibration_set):
                self.abstention_similarity_threshold = (
                    self._calibrate_abstention_similarity_threshold(calibration_set)
                )
        validation_metrics = self._evaluate_validation_fit(current_calibration)
        calibration_metrics = self._evaluate_validation_fit(calibration_set)
        preference_validation = [ep for ep in validation if ep.task_type == "preference"]
        preference_validation_metrics = self._evaluate_preference_fit(preference_validation)
        procedure_validation = [ep for ep in validation if ep.task_type == "procedure"]
        procedure_validation_metrics = self._evaluate_procedure_fit(procedure_validation)
        validation_promotions = self._validate_fact_patterns(validation)
        concept_validation = self._validate_concept_patterns(validation)
        validated_training_fact_count = sum(
            (
                fact_id := self.fact_key_to_node.get(
                    f"{user_id}\0{normalized_answer}"
                )
            ) is not None
            and self.fact_node_data[fact_id]["status"] in {"validated", "promoted"}
            for user_id, normalized_answer in train_fact_keys
        )
        fact_validation_gate = {
            "training_fact_count": len(train_fact_keys),
            "matched_validation_fact_count": len(matched_validation_facts),
            "validated_training_fact_count": validated_training_fact_count,
            "passed": (
                None if not train_fact_keys else
                len(matched_validation_facts) == len(train_fact_keys)
                and validated_training_fact_count == len(train_fact_keys)
            ),
        }
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
            "new_concept_nodes": len(set(self.concept_node_data) - concepts_before),
            "total_concept_nodes": len(self.concept_node_data),
            "validated_concept_nodes": sum(data["status"] in {"validated", "promoted"} for data in self.concept_node_data.values()),
            "promoted_concept_nodes": sum(data["status"] == "promoted" for data in self.concept_node_data.values()),
            "concept_validation": concept_validation,
            "new_abstention_patterns": len(set(self.abstention_node_data) - abstentions_before),
            "total_abstention_patterns": len(self.abstention_node_data),
            "new_preferences": sum(
                1 for node_id, data in self.preference_node_data.items()
                if node_id not in preferences_before
            ),
            "updated_preferences": sum(
                1 for node_id, data in self.preference_node_data.items()
                if node_id in preferences_before and data.get("value") != preferences_before[node_id]
            ),
            "total_preferences": len(self.preference_node_data),
            "new_procedure_nodes": len(set(self.procedure_node_data) - procedures_before),
            "total_procedure_nodes": len(self.procedure_node_data),
            "validated_procedure_nodes": sum(d.get("status") == "validated" for d in self.procedure_node_data.values()),
        }
        return {
            "train_episodes": len(train),
            "validation_episodes": len(validation),
            "source_count": len(self.sources),
            "abstain_threshold": self.abstain_threshold,
            "abstention_similarity_threshold": self.abstention_similarity_threshold,
            "validation_metrics": validation_metrics,
            "calibration_metrics": calibration_metrics,
            "calibration_episode_count": len(calibration_set),
            "preference_validation_metrics": preference_validation_metrics,
            "procedure_validation_metrics": procedure_validation_metrics,
            "fact_validation_coverage": {
                "training_fact_count": len(train_fact_keys),
                "matched_validation_fact_count": len(matched_validation_facts),
                "fraction": (
                    len(matched_validation_facts) / len(train_fact_keys)
                    if train_fact_keys else None
                ),
            },
            "fact_validation_gate": fact_validation_gate,
            "knowledge_growth": knowledge_growth,
        }

    def _learn_procedure(self, episode: StudyEpisode) -> tuple[str, bool]:
        key = f"{episode.user_id}\0{episode.workflow_key}"
        node_id = self.procedure_key_to_node.get(key)
        created = node_id is None
        vector = np.asarray(self.agent.config.encode_passage(episode.question), dtype=float)
        if created:
            node_id = f"study_procedure_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            node = HFN(mu=vector, sigma=np.ones(self.agent.m_dim) * 0.2, id=node_id, use_diag=True)
            node.relation_type = "study_procedure"
            node.metadata = {"type": "study_procedure", "user_id": episode.user_id,
                             "workflow_key": episode.workflow_key, "episode_ids": [episode.episode_id],
                             "triggers": [episode.question], "status": "candidate"}
            self.agent.observer.register(node, protected=False, initial_weight=0.5)
            self.agent.patterns[node_id] = node
            self.procedure_key_to_node[key] = node_id
            self.procedure_node_data[node_id] = {"user_id": episode.user_id, "workflow_key": episode.workflow_key,
                "episode_ids": [episode.episode_id], "triggers": [episode.question], "status": "candidate",
                "validation_episode_ids": [], "centroid": self._procedure_vector(episode.question).tolist(),
                "trigger_vectors": [self._procedure_vector(episode.question).tolist()],
                "anchor_tokens": sorted(_workflow_tokens(episode.question))}
            return node_id, True
        data = self.procedure_node_data[node_id]
        if episode.question not in data["triggers"]:
            data["triggers"].append(episode.question)
            data["episode_ids"].append(episode.episode_id)
            data.setdefault("trigger_vectors", []).append(self._procedure_vector(episode.question).tolist())
            trigger_tokens = _workflow_tokens(episode.question)
            anchors = set(data.get("anchor_tokens", ()))
            data["anchor_tokens"] = sorted(anchors & trigger_tokens) if anchors else sorted(trigger_tokens)
            node = self.agent.forest.get(node_id)
            vectors = [np.asarray(self.agent.config.encode_passage(t), dtype=float) for t in data["triggers"]]
            centroid = np.mean([self._procedure_vector(t) for t in data["triggers"]], axis=0)
            data["centroid"] = centroid.tolist()
            node.mu = np.mean(vectors, axis=0)
            self._update_node_metadata(node_id, {
                "triggers": list(data["triggers"]),
                "episode_ids": list(data["episode_ids"]),
            }, defaults={"type": "study_procedure", "user_id": data.get("user_id"),
                         "workflow_key": data.get("workflow_key"), "status": data.get("status")})
        return node_id, False

    def _procedure_vector(self, text: str) -> np.ndarray:
        """Use the configured frozen semantic encoder when available."""
        if hasattr(self.agent.config, "embedding_model"):
            encoded = np.asarray(self.agent.config.encode_passage(text), dtype=float)
            start = self.agent.config.S_DIM
            vector = encoded[start:start + self.agent.config.DIM]
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm else vector
        return _workflow_vector(text)

    def _update_node_metadata(
        self,
        node_id: str,
        updates: dict[str, Any],
        *,
        defaults: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update metadata on the live forest node, restoring it when absent.

        Older/cold-loaded HFN nodes can exist in the forest without their
        dynamic ``metadata`` attribute. Training must still be able to promote
        or revise those nodes after checkpoint reload.
        """
        node = self.agent.forest.get(node_id)
        pattern_node = self.agent.patterns.get(node_id)
        if node is None:
            node = pattern_node
        if node is None:
            return
        metadata = getattr(node, "metadata", None)
        if not isinstance(metadata, dict):
            pattern_metadata = getattr(pattern_node, "metadata", None)
            metadata = dict(pattern_metadata) if isinstance(pattern_metadata, dict) else dict(defaults or {})
            node.metadata = metadata
        metadata.update(updates)
        if pattern_node is not None and pattern_node is not node:
            pattern_node.metadata = dict(metadata)

    def _evaluate_procedure_fit(self, episodes: list[StudyEpisode]) -> dict[str, Any]:
        rows = []
        for episode in episodes:
            match = self.predict_workflow(episode.question, user_id=episode.user_id, include_candidates=True)
            correct = bool(match and match[0] == episode.workflow_key)
            rows.append({"episode_id": episode.episode_id, "expected": episode.workflow_key,
                         "selected": match[0] if match else None, "confidence": match[1] if match else 0.0,
                         "correct": correct})
            if correct:
                data = self.procedure_node_data[match[2]]
                if len(set(data["triggers"])) >= 2:
                    data["status"] = "validated"
                    data["validation_episode_ids"].append(episode.episode_id)
                    node = self.agent.forest.get(match[2])
                    if node is not None:
                        self._update_node_metadata(match[2], {
                            "status": "validated",
                            "validation_episode_ids": list(data["validation_episode_ids"]),
                        }, defaults={"type": "study_procedure", "user_id": data.get("user_id"),
                                     "workflow_key": data["workflow_key"]})
        correct = sum(row["correct"] for row in rows)
        return {"episode_count": len(rows), "accuracy": correct / len(rows) if rows else None, "episodes": rows}

    def _evaluate_preference_fit(self, episodes: list[StudyEpisode]) -> dict[str, Any]:
        """Measure explicit preference recall without using validation to update state."""
        if not episodes:
            return {"episode_count": 0, "accuracy": None}
        predictions = [self.predict(episode) for episode in episodes]
        correct = sum(
            prediction.answer == episode.expected_answer
            for episode, prediction in zip(episodes, predictions)
        )
        return {
            "episode_count": len(episodes),
            "accuracy": correct / len(episodes),
            "accuracy_ci95": _wilson_interval(correct, len(episodes)),
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "preference_key": episode.preference_key,
                    "expected_value": episode.expected_answer,
                    "selected_value": prediction.answer,
                }
                for episode, prediction in zip(episodes, predictions)
            ],
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
        calibration_episodes: Iterable[StudyEpisode] = (),
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
        fit_result = self.fit(
            [], train_episodes, validation_episodes,
            calibration_episodes=calibration_episodes,
        )
        after = self.evaluate(retention)
        accuracy_delta = after["accuracy"] - before["accuracy"]
        before_by_task = before.get("per_task_type", {})
        after_by_task = after.get("per_task_type", {})
        task_deltas = {
            task: after_by_task[task]["accuracy"] - before_by_task[task]["accuracy"]
            for task in before_by_task.keys() & after_by_task.keys()
        }
        task_gates = {
            task: delta >= -max_accuracy_drop
            for task, delta in task_deltas.items()
        }
        overall_gate = accuracy_delta >= -max_accuracy_drop
        return {
            "fit": fit_result,
            "retention_before": before,
            "retention_after": after,
            "retention_accuracy_delta": accuracy_delta,
            "retention_max_accuracy_drop": max_accuracy_drop,
            "retention_task_accuracy_deltas": task_deltas,
            "retention_task_gates_passed": task_gates,
            "retention_overall_gate_passed": overall_gate,
            "retention_gate_passed": overall_gate and all(task_gates.values()),
        }

    def fit_curriculum(
        self,
        phases: Iterable[dict[str, Any]],
        *,
        retention_episodes: Iterable[StudyEpisode] = (),
        initial_validation_episodes: Iterable[StudyEpisode] = (),
        max_accuracy_drop: float = 0.05,
    ) -> dict[str, Any]:
        """Train an ordered, cumulative curriculum and audit retention per phase.

        Each phase is a mapping with ``name``, ``sources``, ``train_episodes``,
        optional ``validation_episodes``, and optional ``test_episodes``. Each
        phase test is measured read-only after its source is ingested but
        before supervision, then again after fitting, to expose actual
        acquisition benefit. Tests for a completed phase become retention
        checks in later phases. An optional fixed ``retention_episodes`` suite
        is also checked before and after every phase. Test labels are never
        used for fitting or calibration. ``initial_validation_episodes`` keeps
        the fitted checkpoint's earlier validation set in the calibration pool
        when continuing a saved training run.
        """
        phase_items = list(phases)
        if not phase_items:
            raise ValueError("curriculum requires at least one training phase")
        retention = list(retention_episodes)
        initial_validation = list(initial_validation_episodes)
        self._validate_split(retention, "test")
        self._validate_split(initial_validation, "validation")
        if not 0.0 <= max_accuracy_drop <= 1.0:
            raise ValueError("max_accuracy_drop must be between 0 and 1")

        train_val_ids = {
            episode.episode_id
            for phase in phase_items
            for episode in [
                *phase.get("train_episodes", ()), *phase.get("validation_episodes", ())
            ]
        }
        phase_test_ids = [
            episode.episode_id
            for phase in phase_items
            for episode in phase.get("test_episodes", ())
        ]
        if len(phase_test_ids) != len(set(phase_test_ids)):
            raise ValueError("test episode IDs must be unique across curriculum phases")
        if train_val_ids & set(phase_test_ids):
            raise ValueError("phase test episode IDs overlap training or validation episodes")
        if train_val_ids & {episode.episode_id for episode in retention}:
            raise ValueError("retention episode IDs overlap training or validation episodes")
        initial_validation_ids = {episode.episode_id for episode in initial_validation}
        if initial_validation_ids & train_val_ids:
            raise ValueError("initial validation IDs overlap the current curriculum")
        if initial_validation_ids & {episode.episode_id for episode in retention}:
            raise ValueError("initial validation IDs overlap retention tests")
        if self._fit_episode_ids and not initial_validation_ids <= self._fit_episode_ids:
            raise ValueError("initial calibration episodes must belong to the loaded checkpoint")
        if set(phase_test_ids) & {episode.episode_id for episode in retention}:
            raise ValueError("phase test IDs overlap fixed retention episode IDs")
        for phase in phase_items:
            self._validate_split(list(phase.get("test_episodes", ())), "test")

        reports = []
        names: set[str] = set()
        completed_phase_tests: list[StudyEpisode] = []
        # Continue calibrating from the original validation history after a
        # checkpoint restart so local examples cannot replace prior evidence.
        completed_validation: list[StudyEpisode] = list(initial_validation)
        for index, phase in enumerate(phase_items, start=1):
            name = str(phase.get("name") or f"phase-{index}")
            if name in names:
                raise ValueError(f"duplicate curriculum phase name: {name}")
            names.add(name)
            if "train_episodes" not in phase:
                raise ValueError(f"curriculum phase {name!r} is missing train_episodes")
            current_retention = list(retention)
            retained_ids = {episode.episode_id for episode in current_retention}
            current_retention.extend(
                episode for episode in completed_phase_tests
                if episode.episode_id not in retained_ids
            )
            phase_tests = list(phase.get("test_episodes", ()))
            # This is a diagnostic baseline: sources are available to HPM, but
            # this phase's labels and supervised patterns are not yet used.
            self.ingest(phase.get("sources", ()))
            phase_test_before = self.evaluate(phase_tests) if phase_tests else None
            if current_retention:
                result = self.fit_phase(
                    (),
                    phase["train_episodes"],
                    phase.get("validation_episodes", ()),
                    retention_episodes=current_retention,
                    max_accuracy_drop=max_accuracy_drop,
                    calibration_episodes=completed_validation,
                )
            else:
                fit_result = self.fit(
                    [], phase["train_episodes"], phase.get("validation_episodes", ()),
                    calibration_episodes=completed_validation,
                )
                result = {
                    "fit": fit_result,
                    "retention_before": None,
                    "retention_after": None,
                    "retention_accuracy_delta": None,
                    "retention_max_accuracy_drop": max_accuracy_drop,
                    "retention_task_accuracy_deltas": {},
                    "retention_task_gates_passed": {},
                    "retention_overall_gate_passed": None,
                    "retention_gate_passed": None,
                }
            phase_test_metrics = self.evaluate(phase_tests) if phase_tests else None
            phase_test_task_deltas = {}
            if phase_test_before is not None and phase_test_metrics is not None:
                before_tasks = phase_test_before["per_task_type"]
                after_tasks = phase_test_metrics["per_task_type"]
                phase_test_task_deltas = {
                    task: after_tasks[task]["accuracy"] - before_tasks[task]["accuracy"]
                    for task in before_tasks.keys() & after_tasks.keys()
                }
            result["phase_test_metrics"] = phase_test_metrics
            result["phase_test_before_metrics"] = phase_test_before
            result["phase_test_accuracy_delta"] = (
                phase_test_metrics["accuracy"] - phase_test_before["accuracy"]
                if phase_test_before is not None and phase_test_metrics is not None else None
            )
            result["phase_test_task_accuracy_deltas"] = phase_test_task_deltas
            result["phase_test_episode_count"] = len(phase_tests)
            reports.append({"name": name, **result})
            completed_validation.extend(phase.get("validation_episodes", ()))
            completed_phase_tests.extend(phase_tests)
            self._log({
                "event": "curriculum_phase_complete",
                "phase": name,
                "phase_index": index,
                "retention_gate_passed": result["retention_gate_passed"],
                "retention_accuracy_delta": result["retention_accuracy_delta"],
                "phase_test_metrics": phase_test_metrics,
                "phase_test_before_metrics": phase_test_before,
                "phase_test_accuracy_delta": result["phase_test_accuracy_delta"],
                "knowledge_growth": result["fit"]["knowledge_growth"],
            })

        return {
            "phase_count": len(reports),
            "phases": reports,
            "all_retention_gates_passed": (
                all(report["retention_gate_passed"] for report in reports
                    if report["retention_gate_passed"] is not None)
                if any(report["retention_gate_passed"] is not None for report in reports)
                else None
            ),
            "retention_episode_count": len(retention),
            "initial_validation_episode_count": len(initial_validation),
            "phase_test_episode_count": len(completed_phase_tests),
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
                "correction_query_vectors": [],
                "correction_query_texts": [],
                "correction_query_episode_ids": [],
            }
        data = self.fact_node_data[fact_id]
        data.setdefault("corrects_node_ids", [])
        data.setdefault("correction_episode_ids", [])
        data.setdefault("correction_query_vectors", [])
        data.setdefault("correction_query_texts", [])
        data.setdefault("correction_query_episode_ids", [])
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
        if episode.task_type == "correction" and episode.episode_id not in data["correction_query_episode_ids"]:
            encoded_question = self.agent.config.encode_passage(episode.question)
            s_dim, dim = self.agent.config.S_DIM, self.agent.config.DIM
            data["correction_query_vectors"].append(
                encoded_question[s_dim : s_dim + dim].astype(float).tolist()
            )
            data["correction_query_texts"].append(episode.question)
            data["correction_query_episode_ids"].append(episode.episode_id)
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
            self._update_node_metadata(fact_id, {
                "correction_of": episode.correction_of,
                "corrects_node_ids": list(data["corrects_node_ids"]),
            }, defaults={"type": "study_fact", "text": data.get("text"),
                         "user_id": data.get("user_id"), "status": data.get("status")})
        self.source_to_fact_nodes.setdefault(episode.evidence_id, [])
        if fact_id not in self.source_to_fact_nodes[episode.evidence_id]:
            self.source_to_fact_nodes[episode.evidence_id].append(fact_id)
        return fact_id, created, support_added, correction_added

    def _learn_concept(self, episode: StudyEpisode, fact_id: str) -> tuple[str, bool, bool]:
        """Group explicitly labeled source facts under a scoped reusable concept."""
        concept_name = " ".join((episode.concept_key or "").casefold().split())
        key = f"{episode.user_id}\0{concept_name}"
        node_id = self.concept_key_to_node.get(key)
        created = node_id is None
        if created:
            node_id = f"study_concept_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            fact_node = self.agent.forest.get(fact_id)
            node = HFN(
                mu=fact_node.mu.copy(),
                sigma=np.ones(self.agent.m_dim) * 0.2,
                id=node_id,
                use_diag=True,
            )
            node.relation_type = "study_concept"
            node.metadata = {
                "type": "study_concept",
                "concept_key": concept_name,
                "user_id": episode.user_id,
                "status": "candidate",
                "fact_node_ids": [],
                "source_ids": [],
                "validation_episode_ids": [],
            }
            self.agent.observer.register(node, protected=False, initial_weight=0.5)
            self.agent.patterns[node_id] = node
            self.concept_key_to_node[key] = node_id
            self.concept_node_data[node_id] = {
                "concept_key": concept_name,
                "user_id": episode.user_id,
                "status": "candidate",
                "fact_node_ids": [],
                "source_ids": [],
                "validation_episode_ids": [],
            }
        data = self.concept_node_data[node_id]
        support_added = fact_id not in data["fact_node_ids"]
        if support_added:
            data["fact_node_ids"].append(fact_id)
            fact_node = self.agent.forest.get(fact_id)
            concept_node = self.agent.forest.get(node_id)
            concept_node.add_child(fact_node)
            fact_node.add_edge(fact_node, concept_node, "instance_of")
        if episode.evidence_id not in data["source_ids"]:
            data["source_ids"].append(episode.evidence_id)
        fact_nodes = [
            self.agent.forest.get(candidate_id)
            for candidate_id in data["fact_node_ids"]
        ]
        fact_nodes = [node for node in fact_nodes if node is not None]
        if fact_nodes:
            concept_node = self.agent.forest.get(node_id)
            concept_node.mu = np.mean([node.mu for node in fact_nodes], axis=0)
            self._update_node_metadata(node_id, {
                "fact_node_ids": list(data["fact_node_ids"]),
                "source_ids": list(data["source_ids"]),
            }, defaults={"type": "study_concept", "user_id": episode.user_id,
                         "concept_key": episode.concept_key})
        return node_id, created, support_added

    def _validate_concept_patterns(self, episodes: list[StudyEpisode]) -> dict[str, int]:
        """Promote concepts only after multi-source support and held-out utility."""
        checked = passed = promoted = 0
        for episode in episodes:
            if not episode.concept_key or not episode.evidence_id or not episode.expected_answer:
                continue
            concept_name = " ".join(episode.concept_key.casefold().split())
            node_id = self.concept_key_to_node.get(f"{episode.user_id}\0{concept_name}")
            if node_id is None:
                continue
            checked += 1
            prediction = self.predict_question(
                episode.question,
                episode.episode_id,
                episode.task_type,
                user_id=episode.user_id,
            )
            answer_f1 = _token_f1(episode.expected_answer, prediction.answer or "")
            valid = (
                prediction.selected_evidence_id == episode.evidence_id
                and answer_f1 >= 0.8
            )
            data = self.concept_node_data[node_id]
            if valid:
                passed += 1
                if episode.episode_id not in data["validation_episode_ids"]:
                    data["validation_episode_ids"].append(episode.episode_id)
            source_count = len(set(data["source_ids"]))
            new_status = (
                "promoted" if source_count >= 3 and valid
                else "validated" if source_count >= 2 and valid
                else "candidate"
            )
            old_status = data["status"]
            if old_status != "promoted" and new_status != "candidate":
                data["status"] = new_status
                node = self.agent.forest.get(node_id)
                self._update_node_metadata(node_id, {
                    "status": new_status,
                    "validation_episode_ids": list(data["validation_episode_ids"]),
                }, defaults={"type": "study_concept", "user_id": data.get("user_id"),
                             "concept_key": data.get("concept_key")})
                if new_status == "promoted":
                    promoted += 1
            self._log({
                "event": "concept_validation",
                "episode_id": episode.episode_id,
                "concept_node_id": node_id,
                "selected_evidence_id": prediction.selected_evidence_id,
                "expected_evidence_id": episode.evidence_id,
                "answer_token_f1": answer_f1,
                "valid": valid,
                "support_source_count": source_count,
                "status": data["status"],
            })
        return {"checked": checked, "passed": passed, "promoted": promoted}

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
                "anchor_tokens": tokenise(episode.question),
                "episode_ids": [],
            }
        data = self.abstention_node_data[node_id]
        if episode.episode_id not in data["episode_ids"]:
            data["episode_ids"].append(episode.episode_id)
        node = self.agent.forest.get(node_id)
        self._update_node_metadata(node_id, {
            "episode_ids": list(data["episode_ids"]),
            "status": "supervised_negative",
        }, defaults={"type": "study_abstention", "user_id": episode.user_id})
        return node_id, created

    def _learn_preference(self, episode: StudyEpisode) -> tuple[str, bool, bool, Optional[str]]:
        """Persist an explicit user-scoped preference, replacing its prior value."""
        key = f"{episode.user_id}\0{episode.preference_key}"
        node_id = self.preference_key_to_node.get(key)
        created = node_id is None
        if created:
            node_id = f"study_preference_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            node = HFN(
                mu=self.agent.config.encode_passage(episode.question),
                sigma=np.ones(self.agent.m_dim) * 0.2,
                id=node_id,
                use_diag=True,
            )
            node.relation_type = "study_preference"
            node.metadata = {
                "type": "study_preference",
                "user_id": episode.user_id,
                "preference_key": episode.preference_key,
                "value": episode.expected_answer,
                "episode_ids": [episode.episode_id],
            }
            self.agent.observer.register(node, protected=False, initial_weight=0.5)
            self.agent.patterns[node_id] = node
            self.preference_key_to_node[key] = node_id
            self.preference_node_data[node_id] = {
                "user_id": episode.user_id,
                "preference_key": episode.preference_key,
                "value": episode.expected_answer,
                "episode_ids": [episode.episode_id],
                "history": [{
                    "value": episode.expected_answer,
                    "episode_id": episode.episode_id,
                    "question": episode.question,
                    "superseded_by": None,
                }],
            }
            return node_id, True, False, None

        data = self.preference_node_data[node_id]
        previous = data.get("value")
        updated = previous != episode.expected_answer
        if updated:
            data["value"] = episode.expected_answer
        history = data.setdefault("history", [])
        if updated and history:
            history[-1]["superseded_by"] = episode.episode_id
        if not history or updated:
            history.append({
                "value": episode.expected_answer,
                "episode_id": episode.episode_id,
                "question": episode.question,
                "superseded_by": None,
            })
        if episode.episode_id not in data.setdefault("episode_ids", []):
            data["episode_ids"].append(episode.episode_id)
        node = self.agent.forest.get(node_id)
        if node is not None:
            self._update_node_metadata(node_id, {
                "value": data["value"],
                "episode_ids": list(data["episode_ids"]),
            }, defaults={"type": "study_preference", "user_id": episode.user_id,
                         "preference_key": episode.preference_key})
        return node_id, False, updated, previous

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
                self._update_node_metadata(fact_id, {"status": status}, defaults={
                    "type": "study_fact", "text": data.get("text"),
                    "user_id": data.get("user_id"),
                })
                promotions += 1
            self._log({
                "event": "fact_validation",
                "episode_id": episode.episode_id,
                "fact_node_id": fact_id,
                "status": data["status"],
                "answer_token_f1": _token_f1(episode.expected_answer, predicted.answer),
            })
        return promotions

    def _fact_utility(
        self, source_id: str, answer_node_id: str, *, question: str, user_id: str
    ) -> float:
        for fact_id in self.source_to_fact_nodes.get(source_id, ()):
            data = self.fact_node_data[fact_id]
            if data.get("user_id", "default") != user_id:
                continue
            if answer_node_id not in data["answer_sentence_ids"]:
                continue
            if data.get("correction_episode_ids"):
                query_similarity = self._correction_query_similarity(question, data)
                answer_similarity = self._correction_answer_similarity(
                    question, data, answer_node_ids=(answer_node_id,)
                )
                if not self._correction_transfer_matches(query_similarity, answer_similarity):
                    continue
            weight = float(self.agent.observer.get_weight(fact_id))
            level = 0.20 if data["status"] == "promoted" else 0.10 if data["status"] == "validated" else 0.05
            return 1.0 + level * weight
        return 1.0

    def _calibrate_threshold(self, episodes: list[StudyEpisode]) -> float:
        scored = [(self._rank(ep.question, user_id=ep.user_id), ep) for ep in episodes]
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
            if episode.task_type == "preference":
                task_correct = prediction.answer == episode.expected_answer
            elif episode.task_type == "procedure":
                task_correct = prediction.answer == episode.workflow_key
            elif episode.task_type == "compare":
                task_correct = set(prediction.selected_evidence_ids) == set(episode.evidence_ids)
            else:
                task_correct = prediction.selected_evidence_id == episode.evidence_id
            bucket["correct"] += int(task_correct)
        correct = sum(v["correct"] for v in metrics.values())
        evidence_answerable = [ep for ep in items if ep.evidence_id is not None]
        comparison_episodes = [ep for ep in items if ep.task_type == "compare"]
        answerable = [
            ep for ep in items
            if ep.evidence_id is not None or ep.task_type == "preference"
        ]
        unsupported = [
            ep for ep in items
            if ep.evidence_id is None and ep.task_type not in {"preference", "procedure", "compare"}
        ]
        evidence_correct = sum(
            p.selected_evidence_id == ep.evidence_id
            for ep, p in zip(items, predictions)
            if ep.evidence_id is not None
        )
        comparison_expected = sum(len(ep.evidence_ids) for ep in comparison_episodes)
        comparison_retrieved = sum(
            len(set(prediction.selected_evidence_ids) & set(episode.evidence_ids))
            for episode, prediction in zip(items, predictions)
            if episode.task_type == "compare"
        )
        comparison_precision = sum(
            len(set(prediction.selected_evidence_ids) & set(episode.evidence_ids))
            for episode, prediction in zip(items, predictions)
            if episode.task_type == "compare"
        )
        comparison_selected = sum(
            len(prediction.selected_evidence_ids)
            for episode, prediction in zip(items, predictions)
            if episode.task_type == "compare"
        )
        evidence_total = len(evidence_answerable) + comparison_expected
        evidence_correct_total = evidence_correct + comparison_retrieved
        grounded_answer_episodes = [
            (episode, prediction) for episode, prediction in zip(items, predictions)
            if episode.evidence_id is not None and episode.expected_answer is not None
        ]
        grounded_answer_correct = sum(
            not prediction.abstained
            and prediction.selected_evidence_id == episode.evidence_id
            and _token_f1(episode.expected_answer or "", prediction.answer or "") >= 0.8
            for episode, prediction in grounded_answer_episodes
        )
        return {
            "accuracy": correct / len(items),
            "accuracy_ci95": _wilson_interval(correct, len(items)),
            "correct": correct,
            "total": len(items),
            "evidence_accuracy": (
                evidence_correct_total / evidence_total if evidence_total else None
            ),
            "evidence_accuracy_ci95": (
                _wilson_interval(evidence_correct_total, evidence_total) if evidence_total else None
            ),
            "answerable_total": len(evidence_answerable) + len(comparison_episodes),
            "abstention_rate": sum(p.abstained for p in predictions) / len(items),
            "unsupported_rejection_rate": (
                sum(
                    p.abstained for ep, p in zip(items, predictions)
                    if ep.evidence_id is None and ep.task_type not in {"preference", "procedure", "compare"}
                )
                / len(unsupported) if unsupported else None
            ),
            "false_abstention_rate": (
                sum(
                    p.abstained for ep, p in zip(items, predictions)
                    if ep.evidence_id is not None or ep.task_type == "compare"
                )
                / (len(evidence_answerable) + len(comparison_episodes))
                if evidence_answerable or comparison_episodes else None
            ),
            "median_latency_ms": float(np.median(latencies_ms)),
            "answer_reference_count": sum(
                ep.expected_answer is not None and (
                    ep.evidence_id is not None or ep.task_type == "preference"
                ) for ep in items
            ),
            "answer_token_f1": _mean_answer_f1(items, predictions),
            "answer_exact_match": _answer_exact_match(items, predictions),
            "grounded_answer_accuracy": (
                grounded_answer_correct / len(grounded_answer_episodes)
                if grounded_answer_episodes else None
            ),
            "grounded_answer_correct": grounded_answer_correct,
            "grounded_answer_total": len(grounded_answer_episodes),
            "grounded_answer_f1_threshold": 0.8,
            "comparison_evidence_recall": (
                comparison_retrieved / comparison_expected if comparison_expected else None
            ),
            "comparison_evidence_precision": (
                comparison_precision / comparison_selected if comparison_selected else None
            ),
            "episode_results": [
                {
                    "episode_id": episode.episode_id,
                    "task_type": episode.task_type,
                    "expected_evidence_id": episode.evidence_id,
                    "expected_evidence_ids": list(episode.evidence_ids),
                    "expected_workflow_key": episode.workflow_key,
                    "selected_evidence_id": prediction.selected_evidence_id,
                    "selected_evidence_ids": list(prediction.selected_evidence_ids),
                    "selected_workflow_key": prediction.answer if episode.task_type == "procedure" else None,
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
            "concept_node_data": self.concept_node_data,
            "concept_key_to_node": self.concept_key_to_node,
            "abstention_node_data": self.abstention_node_data,
            "preference_node_data": self.preference_node_data,
            "preference_key_to_node": self.preference_key_to_node,
            "procedure_node_data": self.procedure_node_data,
            "procedure_key_to_node": self.procedure_key_to_node,
            "procedure_similarity_threshold": self.procedure_similarity_threshold,
            "correction_query_similarity_threshold": self.correction_query_similarity_threshold,
            "correction_answer_similarity_threshold": self.correction_answer_similarity_threshold,
            "correction_transfer_score_threshold": self.correction_transfer_score_threshold,
            "fact_retrieval_similarity_threshold": self.fact_retrieval_similarity_threshold,
            "concept_retrieval_similarity_threshold": self.concept_retrieval_similarity_threshold,
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
                    *self.abstention_node_data, *self.preference_node_data,
                    *self.concept_node_data,
                    *self.procedure_node_data,
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
        self.concept_node_data = dict(state.get("concept_node_data", {}))
        self.concept_key_to_node = dict(state.get("concept_key_to_node", {}))
        self.abstention_node_data = dict(state.get("abstention_node_data", {}))
        self.preference_node_data = dict(state.get("preference_node_data", {}))
        self.preference_key_to_node = dict(state.get("preference_key_to_node", {}))
        self.procedure_node_data = dict(state.get("procedure_node_data", {}))
        self.procedure_key_to_node = dict(state.get("procedure_key_to_node", {}))
        self.procedure_similarity_threshold = float(state.get("procedure_similarity_threshold", self.procedure_similarity_threshold))
        self.correction_query_similarity_threshold = float(
            state.get("correction_query_similarity_threshold", CORRECTION_QUERY_SIMILARITY_THRESHOLD)
        )
        self.correction_answer_similarity_threshold = float(
            state.get("correction_answer_similarity_threshold", CORRECTION_ANSWER_SIMILARITY_THRESHOLD)
        )
        self.correction_transfer_score_threshold = float(
            state.get("correction_transfer_score_threshold", CORRECTION_TRANSFER_SCORE_THRESHOLD)
        )
        self.fact_retrieval_similarity_threshold = float(
            state.get("fact_retrieval_similarity_threshold", FACT_RETRIEVAL_SIMILARITY_THRESHOLD)
        )
        self.concept_retrieval_similarity_threshold = float(
            state.get("concept_retrieval_similarity_threshold", CONCEPT_RETRIEVAL_SIMILARITY_THRESHOLD)
        )
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
        missing = set(episode.evidence_ids) - self.source_to_node.keys()
        if missing:
            raise ValueError(
                f"episode {episode.episode_id} references unknown evidence IDs {sorted(missing)}"
            )

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


_WORKFLOW_STOP_WORDS = {"a", "an", "the", "of", "for", "from", "to", "me", "my", "these", "this", "that", "in", "on", "about", "can", "could", "would", "you", "please", "give", "make", "create", "write", "provide", "turn", "into", "material", "notes", "what", "is", "using"}
_WORKFLOW_ALIASES = {
    "recap": "summarize", "summary": "summarize", "summarise": "summarize",
    "summarize": "summarize", "overview": "summarize", "definition": "define",
    "defined": "define", "flashcard": "flashcards", "cards": "flashcards",
    "card": "flashcards", "structure": "outline", "structured": "outline",
    "meaning": "define", "means": "define",
}


def _workflow_tokens(text: str) -> set[str]:
    return {_WORKFLOW_ALIASES.get(token, token) for token in tokenise(text)} - _WORKFLOW_STOP_WORDS


def _workflow_vector(text: str, dimensions: int = 256) -> np.ndarray:
    """Hash salient trigger words independently of the source-document vocabulary."""
    vector = np.zeros(dimensions, dtype=float)
    for token in _workflow_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


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
        or episode.task_type == "preference" and episode.expected_answer is not None
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
        and (episode.evidence_id is not None or episode.task_type == "preference")
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
        or episode.task_type == "preference" and episode.expected_answer is not None
    ]
    if not pairs:
        return None
    normalize = lambda text: " ".join(tokenise(text))
    return sum(normalize(expected) == normalize(answer or "") for expected, answer in pairs) / len(pairs)
