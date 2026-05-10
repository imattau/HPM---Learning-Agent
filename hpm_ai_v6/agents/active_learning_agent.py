from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import heapq
import random
import string

from hpm_ai_v6.hpm_model.core.cell import Cell


@dataclass
class DocumentCursor:
    doc_id: int
    sentences: List[str]
    next_idx: int = 0
    exhausted: bool = False


@dataclass
class ActiveCorpus:
    documents: List[DocumentCursor] = field(default_factory=list)

    @classmethod
    def from_documents(cls, documents: Sequence[Sequence[str]]) -> "ActiveCorpus":
        return cls(
            documents=[
                DocumentCursor(doc_id=doc_id, sentences=list(sentences))
                for doc_id, sentences in enumerate(documents)
                if sentences
            ]
        )

    def frontier_entries(self, frontier_width: int = 1) -> List[Tuple[int, int, str]]:
        entries: List[Tuple[int, int, str]] = []
        for doc in self.documents:
            if doc.exhausted:
                continue
            upper = min(len(doc.sentences), doc.next_idx + frontier_width)
            for sent_idx in range(doc.next_idx, upper):
                entries.append((doc.doc_id, sent_idx, doc.sentences[sent_idx]))
        return entries

    def get_sentence(self, doc_id: int, sent_idx: int) -> str:
        return self.documents[doc_id].sentences[sent_idx]

    def mark_consumed(self, doc_id: int, sent_idx: int):
        doc = self.documents[doc_id]
        if sent_idx == doc.next_idx:
            doc.next_idx += 1
        else:
            doc.next_idx = max(doc.next_idx, sent_idx + 1)
        if doc.next_idx >= len(doc.sentences):
            doc.exhausted = True

    def unread_doc_ids(self) -> List[int]:
        return [doc.doc_id for doc in self.documents if not doc.exhausted]


class ActiveLearningAgent:
    """
    Scores frontier sentences for informativeness and selects the next one to read.
    The agent is read-only: it observes other agents' current state but never trains them.
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        tag_fn: Callable[[List[str]], List[str]],
        refresh_interval: int = 10,
        exploration_rate: float = 0.1,
        frontier_width: int = 1,
        agent_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
    ):
        self.agents = agents
        self.tag_fn = tag_fn
        self.refresh_interval = refresh_interval
        self.exploration_rate = exploration_rate
        self.frontier_width = frontier_width
        self.agent_weights = agent_weights or {name: 1.0 for name in agents}
        self.rng = random.Random(seed)

        self.corpus = ActiveCorpus()
        self.heap: List[Tuple[float, int, int, int]] = []
        self.step = 0
        self.version = 0
        self.enqueued: set[Tuple[int, int]] = set()
        self.consumed: set[Tuple[int, int]] = set()

    @staticmethod
    def _clean_words(text: str) -> List[str]:
        return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]

    def _sequence_for_agent(self, agent_name: str, sentence: str) -> Tuple[List[Cell], List[Cell]]:
        agent = self.agents.get(agent_name)
        if agent is None:
            return [], []

        if agent_name == "char":
            chars = [c for c in sentence.lower() if c in agent.char_cells]
            return [agent.char_cells[c] for c in chars], list(agent.char_cells.values())

        if agent_name == "word":
            words = self._clean_words(sentence)
            population = list(agent.word_cells.values())
            seq: List[Cell] = []
            extra: List[Cell] = []
            for word in words:
                if word in agent.word_cells:
                    seq.append(agent.word_cells[word])
                else:
                    if population:
                        proto = agent.word_cells.get("alice") or population[0]
                        emb = proto.as_tensor() * 0.0
                    else:
                        emb = [0.0] * 16
                    cell = Cell(name=f"word_cf_{word}", dim=0, embedding=emb)
                    seq.append(cell)
                    extra.append(cell)
            return seq, population + extra

        if agent_name == "phrase":
            tags = self.tag_fn(self._clean_words(sentence))
            return [agent.pos_cells[t] for t in tags if t in agent.pos_cells], list(agent.pos_cells.values())

        if agent_name == "semantic":
            cell = agent._get_or_create_sent_cell(sentence)
            seq = [cell] if cell is not None else []
            return seq, list(agent.sent_cells.values())

        return [], []

    def _agent_score(self, agent_name: str, sentence: str) -> float:
        agent = self.agents.get(agent_name)
        if agent is None or not agent.patterns:
            return 0.0

        observation_seq, population = self._sequence_for_agent(agent_name, sentence)
        if len(observation_seq) < 2 or not population:
            return 0.0

        field_amplifications = {}
        if getattr(agent, "shared_field", None) is not None:
            field_amplifications = agent.shared_field.get_amplifications(agent.patterns)

        evaluator = agent.learner.epistemic
        weights = agent.get_weights()
        total_score = 0.0
        total_weight = 0.0
        for idx, pattern in enumerate(agent.patterns):
            result = evaluator.evaluate(
                pattern,
                {
                    "observation_seq": observation_seq,
                    "population": population,
                    "field_amplification": field_amplifications.get(pattern.name, 0.0),
                },
            )
            match_count = int(result.metadata.get("count", 0))
            if match_count <= 0:
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            weight = max(weight, 1e-6)
            nll = float(result.metadata.get("nll", 0.0))
            total_score += nll * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return total_score / total_weight

    def compute_score(self, sentence: str) -> float:
        total = 0.0
        for name in self.agents:
            total += self.agent_weights.get(name, 1.0) * self._agent_score(name, sentence)
        return total

    def _enqueue_frontier(self, doc_id: int):
        doc = self.corpus.documents[doc_id]
        if doc.exhausted:
            return
        upper = min(len(doc.sentences), doc.next_idx + self.frontier_width)
        for sent_idx in range(doc.next_idx, upper):
            key = (doc_id, sent_idx)
            if key in self.consumed or key in self.enqueued:
                continue
            sentence = doc.sentences[sent_idx]
            score = self.compute_score(sentence)
            heapq.heappush(self.heap, (-score, self.version, doc_id, sent_idx))
            self.enqueued.add(key)

    def update_pool(self, corpus: ActiveCorpus):
        self.corpus = corpus
        self.heap = []
        self.step = 0
        self.version = 0
        self.enqueued = set()
        self.consumed = set()
        for doc in self.corpus.documents:
            self._enqueue_frontier(doc.doc_id)

    def _sample_exploration_candidate(self) -> Optional[Tuple[int, int]]:
        unread_docs = self.corpus.unread_doc_ids()
        if not unread_docs:
            return None
        doc_id = self.rng.choice(unread_docs)
        doc = self.corpus.documents[doc_id]
        return doc_id, doc.next_idx

    def select_next_sentence(self) -> Optional[str]:
        if not self.heap:
            return None

        self.step += 1
        self.version += 1

        if self.exploration_rate > 0.0 and self.rng.random() < self.exploration_rate:
            candidate = self._sample_exploration_candidate()
            if candidate is not None:
                doc_id, sent_idx = candidate
                key = (doc_id, sent_idx)
                if key not in self.consumed:
                    self.consumed.add(key)
                    self.enqueued.discard(key)
                    self.corpus.mark_consumed(doc_id, sent_idx)
                    self._enqueue_frontier(doc_id)
                    return self.corpus.get_sentence(doc_id, sent_idx)

        while self.heap:
            neg_score, version, doc_id, sent_idx = heapq.heappop(self.heap)
            key = (doc_id, sent_idx)
            if key in self.consumed:
                continue

            doc = self.corpus.documents[doc_id]
            if sent_idx < doc.next_idx or doc.exhausted and sent_idx >= len(doc.sentences):
                self.consumed.add(key)
                continue

            if version < self.version and self.refresh_interval > 0 and (self.step % self.refresh_interval == 0):
                score = self.compute_score(self.corpus.get_sentence(doc_id, sent_idx))
                heapq.heappush(self.heap, (-score, self.version, doc_id, sent_idx))
                continue

            self.consumed.add(key)
            self.enqueued.discard(key)
            sentence = self.corpus.get_sentence(doc_id, sent_idx)
            self.corpus.mark_consumed(doc_id, sent_idx)
            self._enqueue_frontier(doc_id)
            return sentence
        return None
