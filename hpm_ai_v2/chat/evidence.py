"""Choose the set of source sentences that answers a question.

The trainer picks one best sentence. A complete answer often needs more: the
mechanism that follows a definition, the second half of a compound question, a
sentence from another source. This module grows the evidence set from that
seed by greedy marginal gain over the structured store:

* coverage: query content stems the set does not mention yet, found in any
  source through the stem map (so ``release`` finds ``released``);
* structure (process questions only): sentences next to the set in reading
  order, or sharing concepts with it, with gain falling at each addition.

Width is decided by the gain, not fixed: growth stops when the best remaining
candidate adds less than a threshold. Summaries pick sentences by weighted
coverage of the source's own stems. The rules are fixed and hand-set, not
learned, and every parameter is named in ``AccessParams``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Optional

from hpm_ai_v2.agents.text_structure import StructureIndex, content_stems, stem
from hpm_ai_v2.chat.feedback import find_answer_node
from hpm_ai_v2.chat.language import QUESTION_WORDS
from hpm_ai_v2.chat.planner import GroundedPlanner
from hpm_ai_v2.chat.schema import Citation, Claim, ResponsePlan

_QUESTION_WORDS = QUESTION_WORDS
_PROCESS = re.compile(
    r"\b(?:how (?:does|do|did|is|are|can|might)|why|explain|describe|what happens|stages?|steps?)\b", re.I
)
_COMPOUND = re.compile(r"\b(?:and|or)\b[^?.]*\b(?:what|how|why|where|which|who|when)\b", re.I)
_DEFINITIONAL = re.compile(r"^\s*(?:what|who)\s+(?:is|are|was|were)\b", re.I)
_SUMMARY_FORMATS = {"summarize", "outline", "flashcards"}


@dataclass(frozen=True)
class AccessParams:
    cap: int = 6                    # most sentences in one answer
    theta_cov: float = 0.25         # least share of query weight a coverage addition must newly cover
    theta_struct: float = 0.10      # least gain for a structural (neighbour or concept) addition
    decay: float = 0.6              # structural gain is multiplied by decay**k after k structural additions
    adjacency_weight: float = 0.5   # weight of reading-order adjacency in structural score
    link_weight: float = 0.5        # weight of shared-concept overlap in structural score
    source_ratio: float = 0.5       # other sources must score at least this fraction of the best source
    forward_weight: float = 1.0     # adjacency to a following sentence (elaboration usually follows)
    backward_weight: float = 0.5    # adjacency to a preceding sentence
    structure_after_coverage: bool = False   # also grow structurally when coverage already added sentences
    max_structural: int = 3         # most sentences structural growth may add (its own budget, apart from cap)
    require_link: bool = True       # a structural addition must share a concept with the evidence so far
    seed_min_share: float = 0.34    # no structural growth unless the seed covers this share of the question's weight
    unseen_ratio: float = 0.5       # if this share of the question's content words is in no source, return the seed alone
    compound_blocks_process: bool = True     # a question with a second clause gets coverage growth only
    swap_seed_share_max: float = 0.0         # the seed gives way only if it covers at most this share of the question
    definitional_first_mention: bool = True  # "what is X" and "how does X work" start where X is introduced
    summary_theta: float = 0.06     # least share of source stem weight a summary addition must cover
    summary_min: int = 3            # summaries are at least this wide when the source allows
    summary_cap: int = 6
    expand_structure: bool = True   # ablation switches
    expand_coverage: bool = True


@dataclass(frozen=True)
class Selection:
    sentence_ids: tuple[str, ...]           # reading order
    roles: dict                              # sentence id -> "seed" | "coverage" | "structure" | "summary"


class EvidenceSelector:
    def __init__(self, trainer, params: AccessParams | None = None):
        self.trainer = trainer
        self.params = params or AccessParams()
        self._index: Optional[StructureIndex] = None
        self._index_key: tuple[int, int] = (-1, -1)

    # -- store views ---------------------------------------------------------------
    def index(self) -> StructureIndex:
        key = (len(self.trainer.sources), len(self.trainer.agent.patterns))
        if self._index is None or key != self._index_key:
            self._index, self._index_key = StructureIndex(self.trainer.agent), key
        return self._index

    def _stems(self, sentence_id: str) -> set[str]:
        return set(self.trainer.agent.patterns[sentence_id].metadata.get("stems", ()))

    def _weight(self, term: str) -> float:
        df = len(self.trainer.agent.term_sentences.get(term, ()))
        return math.log(1 + len(self.trainer.answer_node_data) / df) if df else 0.0

    def _is_correction_seed(self, sentence_id: str, user_id: str) -> bool:
        return any(
            data.get("correction_episode_ids") and sentence_id in data.get("answer_sentence_ids", ())
            and data.get("user_id", "default") == user_id
            for data in self.trainer.fact_node_data.values()
        )

    # -- selection -----------------------------------------------------------------
    def select(self, question: str, plan: ResponsePlan, *, user_id: str = "default",
               scope: Optional[str] = None) -> Optional[Selection]:
        if self.getattr_storage() < 2:
            return None
        seeds = []
        for claim in plan.claims:
            for c in claim.citations:
                node = find_answer_node(self.trainer, c.source_id, c.start, c.end)
                if node is not None and node not in seeds:
                    seeds.append(node)
        if not seeds:
            return None
        if plan.fmt in _SUMMARY_FORMATS:
            return self._summary(seeds[0])
        return self._answer(question.split(" Context:")[0], seeds[0], user_id, scope)

    def _allowed_sources(self, question: str, seed_source: str, user_id: str, scope: Optional[str]) -> set[str]:
        if scope is not None:
            return {scope}
        ranked = self.trainer.predict_question(question, episode_id="evidence", user_id=user_id).ranked_evidence
        best = max((score for _sid, score in ranked), default=0.0)
        allowed = {sid for sid, score in ranked if best > 0 and score >= self.params.source_ratio * best}
        return allowed | {seed_source}

    def getattr_storage(self) -> int:
        return getattr(self.trainer, "storage_version", 1)

    def _answer(self, question: str, seed: str, user_id: str, scope: Optional[str]) -> Selection:
        p, index = self.params, self.index()
        if self._is_correction_seed(seed, user_id):
            # the user chose this span explicitly; widening it could re-add the sentence they corrected
            return Selection((seed,), {seed: "seed"})
        process = bool(_PROCESS.search(question)) and not (p.compound_blocks_process and _COMPOUND.search(question))
        content = [t for t in content_stems(question) if t not in _QUESTION_WORDS]
        unseen = [t for t in content if not self.trainer.agent.term_sentences.get(t)]
        if content and len(unseen) / len(content) >= p.unseen_ratio:
            # most of the question's own words are in no source: it is probably out of scope, so do not widen the seed
            return Selection((seed,), {seed: "seed"})
        allowed = self._allowed_sources(question, index.source_of(seed), user_id, scope)
        # a query word counts only if some relevant source says it (a stray "work" in an unrelated source does not)
        query = [
            t for t in content_stems(question)
            if t not in _QUESTION_WORDS
            and any(index.source_of(sid) in allowed for sid in index.sentences_with_stem(t))
        ]
        total = sum(self._weight(t) for t in query)
        if (p.definitional_first_mention and (_DEFINITIONAL.match(question) or process) and 0 < len(query) <= 2
                and not self._is_correction_seed(seed, user_id)):
            mentions = [sid for sid in index.sentence_ids(index.source_of(seed)) if set(query) <= self._stems(sid)]
            if mentions:
                seed = mentions[0]
        chosen: dict[str, str] = {seed: "seed"}

        def covered() -> set[str]:
            return {t for sid in chosen for t in self._stems(sid) if t in query}

        def pool_ok(sid: str) -> bool:
            return index.source_of(sid) in allowed

        if p.expand_coverage and total > 0:
            # a seed that mentions little of the question, when one sentence mentions most of it, gives way
            best = max(
                (sid for t in query for sid in index.sentences_with_stem(t) if pool_ok(sid)),
                key=lambda sid: (sum(self._weight(t) for t in self._stems(sid) if t in query), sid),
                default=None,
            )
            if best is not None and best != seed and not self._is_correction_seed(seed, user_id):
                share = lambda sid: sum(self._weight(t) for t in self._stems(sid) if t in query) / total
                if share(seed) <= p.swap_seed_share_max and share(best) >= 0.5:
                    chosen = {best: "coverage"}
            while len(chosen) < p.cap:
                missing = [t for t in query if t not in covered()]
                if not missing:
                    break
                options = {sid for t in missing for sid in index.sentences_with_stem(t) if pool_ok(sid)} - set(chosen)
                if not options:
                    break
                def gain(sid):
                    return sum(self._weight(t) for t in self._stems(sid) if t in missing) / total
                top = max(options, key=lambda sid: (gain(sid), -self._distance(index, sid, chosen), sid))
                if gain(top) < p.theta_cov:
                    break
                chosen[top] = "coverage"
        grew_by_coverage = len(chosen) > 1
        seed_share = (
            sum(self._weight(t) for t in self._stems(seed) if t in query) / total if total > 0 else 1.0
        )
        if (p.expand_structure and process and (p.structure_after_coverage or not grew_by_coverage)
                and seed_share >= p.seed_min_share):
            additions = 0
            while len(chosen) < p.cap and additions < p.max_structural:
                options = {n for sid in chosen for n in index.neighbours(sid, 1, 1)} - set(chosen)
                options |= {o for sid in chosen for o in index.related(sid)} - set(chosen)
                options = {o for o in options if pool_ok(o)}
                scored = {o: self._structure_score(index, o, chosen) for o in options}
                scored = {o: s * (p.decay ** (additions + 1)) for o, s in scored.items() if s > 0}
                if not scored:
                    break
                top = max(scored, key=lambda o: (scored[o], o))
                if scored[top] < p.theta_struct:
                    break
                chosen[top] = "structure"
                additions += 1
        order = sorted(chosen, key=lambda sid: index.position(sid) or ("", 0))
        return Selection(tuple(order), chosen)

    def _structure_score(self, index: StructureIndex, sid: str, chosen: dict) -> float:
        p = self.params
        pos = index.position(sid)
        adjacent = 0.0
        for c in chosen:
            there = index.position(c)
            if pos and there and there[0] == pos[0] and abs(there[1] - pos[1]) == 1:
                adjacent = max(adjacent, p.forward_weight if pos[1] > there[1] else p.backward_weight)
        mine = {t for t in self._stems(sid) if len(self.trainer.agent.term_sentences.get(t, ())) >= 2}
        theirs = {t for c in chosen for t in self._stems(c)}
        den = sum(self._weight(t) for t in mine)
        link = sum(self._weight(t) for t in mine & theirs) / den if den else 0.0
        if p.require_link and link <= 0:
            return 0.0
        return p.adjacency_weight * adjacent + p.link_weight * link

    def _distance(self, index: StructureIndex, sid: str, chosen: dict) -> int:
        pos = index.position(sid)
        near = [abs(index.position(c)[1] - pos[1]) for c in chosen
                if pos and index.position(c) and index.position(c)[0] == pos[0]]
        return min(near) if near else 99

    def _summary(self, seed: str) -> Selection:
        p, index = self.params, self.index()
        source_id = index.source_of(seed)
        pool = index.sentence_ids(source_id)
        weight = {t: self._weight(t) for sid in pool for t in self._stems(sid)}
        # weights within the source: rare-in-source stems say more about it
        df: dict[str, int] = {}
        for sid in pool:
            for t in self._stems(sid):
                df[t] = df.get(t, 0) + 1
        weight = {t: math.log(1 + len(pool) / n) for t, n in df.items()}
        total = sum(weight.values())
        chosen: dict[str, str] = {}
        covered: set[str] = set()
        while len(chosen) < p.summary_cap and len(chosen) < len(pool):
            options = [sid for sid in pool if sid not in chosen]
            top = max(options, key=lambda sid: (sum(weight[t] for t in self._stems(sid) - covered), sid))
            gain = sum(weight[t] for t in self._stems(top) - covered) / total if total else 0.0
            if gain < p.summary_theta and len(chosen) >= p.summary_min:
                break
            chosen[top] = "summary"
            covered |= self._stems(top)
        order = sorted(chosen, key=lambda sid: index.position(sid)[1])
        return Selection(tuple(order), chosen)


class ExpandingPlanner(GroundedPlanner):
    """GroundedPlanner whose evidence set is grown by ``EvidenceSelector``."""

    def __init__(self, trainer, *, params: AccessParams | None = None, **kwargs):
        super().__init__(trainer, **kwargs)
        self.selector = EvidenceSelector(trainer, params)

    def plan(self, turn_id: str, question: str, *, user_id: str = "default",
             scope_source_id: str | None = None) -> ResponsePlan:
        # The support check runs on the final evidence, not the seed: the selector may replace a seed that
        # mentions none of the question with a sentence that does.
        plan = self._plan(turn_id, question, user_id=user_id, scope_source_id=scope_source_id)
        if plan.act != "answer" or not plan.claims or plan.fmt == "compare":
            return self._abstain_if_unsupported(plan)
        selection = self.selector.select(question, plan, user_id=user_id, scope=scope_source_id)
        if selection is None:
            return self._abstain_if_unsupported(plan)
        claims = []
        for n, sid in enumerate(selection.sentence_ids, start=1):
            data = self.trainer.answer_node_data[sid]
            source = self.trainer.sources[data["source_id"]]
            citation = Citation(data["source_id"], data["start"], data["end"], data["text"], source.title)
            claims.append(Claim(f"claim-{turn_id}-{n}", data["text"], (citation,)))
        roles = ",".join(f"{selection.roles[s]}" for s in selection.sentence_ids)
        return self._abstain_if_unsupported(
            replace(plan, claims=tuple(claims), reason=f"{plan.reason};evidence={roles}")
        )
