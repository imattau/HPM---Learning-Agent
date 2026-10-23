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
    ) -> None:
        if not hasattr(agent, "observe_passage") or not hasattr(agent, "query_scored_nodes"):
            raise TypeError("agent must provide ReaderAgent passage and retrieval methods")
        self.agent = agent
        self.event_log = Path(event_log) if event_log else None
        self.feedback_gain = feedback_gain
        self.feedback_penalty = feedback_penalty
        self.abstain_threshold = 0.0
        self.sources: dict[str, SourceDocument] = {}
        self.source_to_node: dict[str, str] = {}
        self.node_to_source: dict[str, str] = {}
        self.source_to_answer_nodes: dict[str, list[str]] = {}
        self.answer_node_data: dict[str, dict[str, Any]] = {}
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
            index = self.agent.observe_passage(source.text, initial_weight=0.5)
            node_id = f"passage_{index}"
            self.sources[source.source_id] = source
            self.source_to_node[source.source_id] = node_id
            self.node_to_source[node_id] = source.source_id
            self._index_answer_sentences(source)
            self._log({"event": "source_ingested", "source_id": source.source_id, "node_id": node_id})

    def _index_answer_sentences(self, source: SourceDocument) -> None:
        """Create persistent sentence patterns with offsets for grounded answers."""
        self.source_to_answer_nodes[source.source_id] = []
        for index, (sentence, start, end) in enumerate(_sentences_with_offsets(source.text)):
            digest = hashlib.sha1(f"{source.source_id}:{start}:{sentence}".encode()).hexdigest()[:12]
            node_id = f"answer_sentence_{digest}"
            if node_id not in self.answer_node_data:
                mu = self.agent.config.encode_passage(sentence)
                node = HFN(
                    mu=mu,
                    sigma=np.ones(self.agent.m_dim) * 0.2,
                    id=node_id,
                    use_diag=True,
                )
                node.relation_type = "answer_sentence"
                node.metadata = {"type": "answer_sentence", "source_id": source.source_id, "start": start, "end": end}
                self.agent.observer.register(node, protected=False, initial_weight=0.5)
                self.agent.patterns[node_id] = node
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
            ranked.append((node_id, similarity * max(0.0, min(1.0, weight))))
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
        """Rank evidence by text match modulated by Observer utility weights."""
        ranked: list[tuple[str, float]] = []
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
            ranked.append((source_id, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def predict(self, episode: StudyEpisode) -> StudyPrediction:
        return self.predict_question(episode.question, episode.episode_id, episode.task_type)

    def predict_question(
        self,
        question: str,
        episode_id: str = "ad-hoc",
        task_type: str = "grounded_qa",
    ) -> StudyPrediction:
        """Read-only evidence selection for a free-form question (no labels needed)."""
        if task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task_type: {task_type}")
        ranked = self._rank(question)
        confidence = ranked[0][1] if ranked else 0.0
        selected = ranked[0][0] if ranked and confidence >= self.abstain_threshold else None
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
        )

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
        for episode in train:
            self._validate_evidence(episode)
            ranked = self._rank(episode.question)
            predicted = ranked[0][0] if ranked else None
            correct = predicted == episode.evidence_id
            if episode.evidence_id is not None:
                gold_node = self.source_to_node[episode.evidence_id]
                self.agent.observer.boost_id(gold_node, gain=self.feedback_gain)
                if episode.expected_answer:
                    target_answer_node = self._gold_answer_node(
                        episode.evidence_id, episode.expected_answer
                    )
                    if target_answer_node is not None:
                        self.agent.observer.boost_id(target_answer_node, gain=self.feedback_gain)
                        if predicted is not None:
                            predicted_answers = self._rank_answers(episode.question, predicted)
                            if predicted_answers and predicted_answers[0][0] != target_answer_node:
                                self.agent.observer.penalize_id(
                                    predicted_answers[0][0], penalty=self.feedback_penalty
                                )
            if predicted is not None and predicted != episode.evidence_id:
                self.agent.observer.penalize_id(
                    self.source_to_node[predicted], penalty=self.feedback_penalty
                )
            self._log({
                "event": "supervision",
                "episode_id": episode.episode_id,
                "split": episode.split,
                "predicted_evidence_id": predicted,
                "gold_evidence_id": episode.evidence_id,
                "correct": correct,
                "correction_of": episode.correction_of,
                "user_id": episode.user_id,
            })

        if validation:
            self.abstain_threshold = self._calibrate_threshold(validation)
        self._fit_episode_ids.update(ep.episode_id for ep in train + validation)
        return {
            "train_episodes": len(train),
            "validation_episodes": len(validation),
            "source_count": len(self.sources),
            "abstain_threshold": self.abstain_threshold,
        }

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
                    "citation": prediction.citation,
                    "latency_ms": latency_ms,
                }
                for episode, prediction, latency_ms in zip(items, predictions, latencies_ms)
            ],
            "per_task_type": {
                task: {
                    **counts,
                    "accuracy": counts["correct"] / counts["total"],
                    "accuracy_ci95": _wilson_interval(counts["correct"], counts["total"]),
                }
                for task, counts in sorted(metrics.items())
            },
        }

    def save_checkpoint(self, directory: str | Path) -> Path:
        """Persist ReaderAgent state and the trainer's source/evidence mapping."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.agent.save_agent(str(path))
        state = {
            "abstain_threshold": self.abstain_threshold,
            "sources": [asdict(source) for source in self.sources.values()],
            "source_to_node": self.source_to_node,
            "source_to_answer_nodes": self.source_to_answer_nodes,
            "answer_node_data": self.answer_node_data,
            "feedback_gain": self.feedback_gain,
            "feedback_penalty": self.feedback_penalty,
            "observer_weights": {
                node_id: self.agent.observer.get_weight(node_id)
                for node_id in (*self.node_to_source, *self.answer_node_data)
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
        if set(self.source_to_node) != set(self.sources):
            raise ValueError("checkpoint source map does not match stored sources")
        for node_id, weight in state.get("observer_weights", {}).items():
            self.agent.observer.set_weight(node_id, float(weight))
        self.feedback_gain = float(state.get("feedback_gain", self.feedback_gain))
        self.feedback_penalty = float(state.get("feedback_penalty", self.feedback_penalty))
        self.abstain_threshold = float(state.get("abstain_threshold", 0.0))
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
