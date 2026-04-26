from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator


@dataclass
class EpisodeRecord:
    """Compact episodic trace represented as a node in a multi-polygraph."""

    context: List[int]
    action: int
    reward: float
    tag: str = "observe"
    metadata: Optional[Dict[str, Any]] = None
    stage: str = "unknown"
    policy: str = "unknown"
    community: str = "unknown"
    is_summary: bool = False
    timestamp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["metadata"] is None:
            d["metadata"] = {}
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EpisodeRecord":
        metadata = dict(data.get("metadata", {}) or {})
        return EpisodeRecord(
            context=list(data.get("context", [])),
            action=int(data.get("action", 0)),
            reward=float(data.get("reward", 0.0)),
            tag=str(data.get("tag", "observe")),
            metadata=metadata,
            stage=str(data.get("stage", metadata.get("stage", "unknown"))),
            policy=str(data.get("policy", metadata.get("policy", "unknown"))),
            community=str(data.get("community", metadata.get("community", "unknown"))),
            is_summary=bool(data.get("is_summary", metadata.get("is_summary", False))),
            timestamp=int(data.get("timestamp", metadata.get("timestamp", 0))),
        )


MemoryEpisode = EpisodeRecord


@dataclass
class CommunityState:
    """Consolidated episode family tracked by the polygraph."""

    key: str
    count: int = 0
    total_reward: float = 0.0
    last_seen: int = 0
    exemplar_index: int = -1
    summary_record: Optional[EpisodeRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "count": self.count,
            "total_reward": self.total_reward,
            "last_seen": self.last_seen,
            "exemplar_index": self.exemplar_index,
            "summary_record": self.summary_record.to_dict() if self.summary_record else None,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CommunityState":
        summary = data.get("summary_record")
        return CommunityState(
            key=str(data.get("key", "unknown")),
            count=int(data.get("count", 0)),
            total_reward=float(data.get("total_reward", 0.0)),
            last_seen=int(data.get("last_seen", 0)),
            exemplar_index=int(data.get("exemplar_index", -1)),
            summary_record=EpisodeRecord.from_dict(summary) if summary else None,
        )


class EpisodicPolygraph:
    """Multi-projection episodic store for HPM-style retrieval."""

    def __init__(self, capacity: int = 256, window: int = 24):
        self.capacity = capacity
        self.window = window
        self.records: List[EpisodeRecord] = []
        self.communities: Dict[str, CommunityState] = {}
        self.context_index: Dict[Tuple[int, ...], List[int]] = {}
        self.stage_index: Dict[str, List[int]] = {}
        self.policy_index: Dict[str, List[int]] = {}
        self.action_index: Dict[int, List[int]] = {}
        self.outcome_index: Dict[int, List[int]] = {}
        self._step = 0
        self.summary_threshold = 3
        self.summary_interval = 2
        self.graph_weights = {
            "context": 0.40,
            "action": 0.15,
            "stage": 0.15,
            "policy": 0.10,
            "outcome": 0.20,
            "summary": 0.10,
            "recency": 0.05,
            "support": 0.03,
        }

    def __len__(self) -> int:
        return len(self.records)

    def set_records(self, records: Sequence[EpisodeRecord | Dict[str, Any]]) -> None:
        self.records = [
            rec if isinstance(rec, EpisodeRecord) else EpisodeRecord.from_dict(rec)
            for rec in records
        ]
        self._step = len(self.records)
        self._trim()
        self._rebuild_indices()
        self._rebuild_communities()

    def add(
        self,
        record: EpisodeRecord,
    ) -> None:
        if record.timestamp <= 0:
            record.timestamp = self._step
        record.community = self._community_key(record)
        self._step += 1
        self.records.append(record)
        self._trim()
        self._index_record(len(self.records) - 1, record)
        self._update_community(len(self.records) - 1, record)

    def query(
        self,
        context_obs: Sequence[int],
        top_k: int = 6,
        min_reward: Optional[float] = None,
        query_action: Optional[int] = None,
        query_stage: Optional[str] = None,
        query_policy: Optional[str] = None,
    ) -> List[EpisodeRecord]:
        if not self.records:
            return []

        context = list(context_obs[-self.window:]) if context_obs else []
        candidate_map = self._candidate_map(context, query_action, query_stage, query_policy)
        if not candidate_map:
            candidate_map = {idx: {"fallback"} for idx in range(len(self.records))}

        scored: List[Tuple[float, EpisodeRecord]] = []
        for idx, sources in candidate_map.items():
            rec = self.records[idx]
            if min_reward is not None and rec.reward < min_reward:
                continue
            score = self._resonance_score(context, rec, query_action, query_stage, query_policy)
            score += self.graph_weights["support"] * max(0, len(sources) - 1)
            scored.append((score, rec))

        for rec in self._summary_candidates(context, query_action, query_stage, query_policy):
            if min_reward is not None and rec.reward < min_reward:
                continue
            score = self._resonance_score(context, rec, query_action, query_stage, query_policy)
            score += 0.10 * min(1.0, rec.reward)
            if rec.is_summary:
                score += 0.15
                if rec.metadata:
                    score += 0.02 * float(rec.metadata.get("count", 0.0))
            scored.append((score, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def projection_summary(
        self,
        context_obs: Sequence[int],
        query_action: Optional[int] = None,
        query_stage: Optional[str] = None,
        query_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return how strongly each projection graph supports the current query."""
        context = list(context_obs[-self.window:]) if context_obs else []
        candidate_map = self._candidate_map(context, query_action, query_stage, query_policy)
        counts = {
            "context": 0,
            "action": 0,
            "stage": 0,
            "policy": 0,
            "outcome": 0,
            "summary": 0,
        }
        for sources in candidate_map.values():
            for source in sources:
                if source in counts:
                    counts[source] += 1
        multi_supported = sum(1 for sources in candidate_map.values() if len(sources) > 1)
        return {
            "graph_counts": counts,
            "candidate_count": len(candidate_map),
            "multi_supported_count": multi_supported,
            "summary_count": sum(1 for rec in self.records if rec.is_summary),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "window": self.window,
            "step": self._step,
            "records": [rec.to_dict() for rec in self.records],
            "communities": {key: community.to_dict() for key, community in self.communities.items()},
            "summary_threshold": self.summary_threshold,
            "summary_interval": self.summary_interval,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.window = int(state.get("window", self.window))
        self._step = int(state.get("step", 0))
        records = state.get("records")
        if records is None:
            records = state.get("memory", [])
        self.set_records(records)
        self.summary_threshold = int(state.get("summary_threshold", self.summary_threshold))
        self.summary_interval = int(state.get("summary_interval", self.summary_interval))
        communities = state.get("communities", {})
        if communities:
            self.communities = {
                str(key): CommunityState.from_dict(value)
                for key, value in communities.items()
            }
        else:
            self._rebuild_communities()

    def _trim(self) -> None:
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity :]

    def _rebuild_indices(self) -> None:
        self.context_index = {}
        self.stage_index = {}
        self.policy_index = {}
        self.action_index = {}
        self.outcome_index = {}
        for idx, rec in enumerate(self.records):
            self._index_record(idx, rec)

    def _rebuild_communities(self) -> None:
        self.communities = {}
        for idx, rec in enumerate(self.records):
            self._update_community(idx, rec, rebuild=True)

    def _index_record(self, idx: int, rec: EpisodeRecord) -> None:
        for key in self._context_keys(rec.context):
            self.context_index.setdefault(key, []).append(idx)
        self.stage_index.setdefault(rec.stage, []).append(idx)
        self.policy_index.setdefault(rec.policy, []).append(idx)
        self.action_index.setdefault(int(rec.action), []).append(idx)
        self.outcome_index.setdefault(self._outcome_bucket(rec), []).append(idx)

    def _update_community(self, idx: int, rec: EpisodeRecord, rebuild: bool = False) -> None:
        key = self._community_key(rec)
        community = self.communities.get(key)
        if community is None:
            community = CommunityState(key=key)
            self.communities[key] = community
        community.count += 1
        community.total_reward += float(rec.reward)
        community.last_seen = max(community.last_seen, int(rec.timestamp))
        if community.exemplar_index < 0 or rec.reward >= self.records[community.exemplar_index].reward:
            community.exemplar_index = idx
        if community.count >= self.summary_threshold and (
            community.summary_record is None
            or community.count % self.summary_interval == 0
            or rebuild
        ):
            community.summary_record = self._build_summary_record(key, community)

    def _summary_candidates(
        self,
        context: Sequence[int],
        query_action: Optional[int],
        query_stage: Optional[str],
        query_policy: Optional[str],
    ) -> List[EpisodeRecord]:
        summaries: List[EpisodeRecord] = []
        for key, community in self.communities.items():
            summary = community.summary_record
            if summary is None:
                continue
            if query_stage and summary.stage != query_stage:
                continue
            if query_policy and summary.policy != query_policy:
                continue
            if query_action is not None and summary.action != int(query_action):
                continue
            if self._community_resonance(context, community) <= 0.15:
                continue
            summaries.append(summary)
        return summaries

    def _candidate_map(
        self,
        context: Sequence[int],
        query_action: Optional[int],
        query_stage: Optional[str],
        query_policy: Optional[str],
    ) -> Dict[int, set[str]]:
        candidate_map: Dict[int, set[str]] = {}

        def add(idx: int, source: str) -> None:
            candidate_map.setdefault(int(idx), set()).add(source)

        for key in self._context_keys(context):
            for idx in self.context_index.get(key, []):
                add(idx, "context")
        if query_action is not None:
            for idx in self.action_index.get(int(query_action), []):
                add(idx, "action")
        if query_stage:
            for idx in self.stage_index.get(query_stage, []):
                add(idx, "stage")
        if query_policy:
            for idx in self.policy_index.get(query_policy, []):
                add(idx, "policy")

        # Outcome graph: bias toward high-value episodes and summary records.
        for bucket in (3, 2):
            for idx in self.outcome_index.get(bucket, []):
                add(idx, "outcome")
        for idx, rec in enumerate(self.records):
            if rec.is_summary:
                add(idx, "summary")

        return candidate_map

    def _community_key(self, rec: EpisodeRecord) -> str:
        context_tail = tuple(rec.context[-4:]) if rec.context else tuple()
        outcome_bucket = self._outcome_bucket(rec)
        return f"{rec.stage}|{rec.policy}|{int(rec.action)}|{outcome_bucket}|{context_tail}"

    def _community_resonance(self, context: Sequence[int], community: CommunityState) -> float:
        if not community.summary_record:
            return 0.0
        ctx_sim = self._context_similarity(context, community.summary_record.context)
        avg_reward = community.total_reward / max(1, community.count)
        recency = float(np.exp(-(max(0, self._step - community.last_seen)) / max(1.0, float(self.window))))
        return 0.55 * ctx_sim + 0.25 * min(1.0, max(0.0, avg_reward)) + 0.20 * recency

    def _build_summary_record(self, key: str, community: CommunityState) -> EpisodeRecord:
        member_records = [rec for rec in self.records if rec.community == key]
        if not member_records:
            member_records = [self.records[community.exemplar_index]] if 0 <= community.exemplar_index < len(self.records) else []
        if not member_records:
            return EpisodeRecord(context=[], action=0, reward=0.0, tag="summary", community=key, is_summary=True)

        avg_len = max(1, int(round(sum(len(rec.context) for rec in member_records) / len(member_records))))
        context = self._merge_contexts([rec.context for rec in member_records], avg_len)
        action_counts: Dict[int, int] = {}
        stage_counts: Dict[str, int] = {}
        policy_counts: Dict[str, int] = {}
        for rec in member_records:
            action_counts[rec.action] = action_counts.get(rec.action, 0) + 1
            stage_counts[rec.stage] = stage_counts.get(rec.stage, 0) + 1
            policy_counts[rec.policy] = policy_counts.get(rec.policy, 0) + 1
        action = max(action_counts.items(), key=lambda item: item[1])[0]
        stage = max(stage_counts.items(), key=lambda item: item[1])[0]
        policy = max(policy_counts.items(), key=lambda item: item[1])[0]
        reward = float(sum(rec.reward for rec in member_records) / len(member_records))
        metadata = {
            "community": key,
            "count": len(member_records),
            "summary": True,
            "avg_reward": reward,
        }
        return EpisodeRecord(
            context=context,
            action=int(action),
            reward=reward,
            tag="summary",
            metadata=metadata,
            stage=stage,
            policy=policy,
            community=key,
            is_summary=True,
            timestamp=community.last_seen,
        )

    def _merge_contexts(self, contexts: List[List[int]], target_len: int) -> List[int]:
        if not contexts:
            return []
        merged: List[int] = []
        for ctx in contexts:
            merged.extend(ctx[-target_len:])
        if not merged:
            return []
        tail = merged[-target_len:]
        if len(tail) < target_len:
            tail = [0] * (target_len - len(tail)) + tail
        return tail

    def _candidate_ids(
        self,
        context: Sequence[int],
        query_action: Optional[int],
        query_stage: Optional[str],
        query_policy: Optional[str],
    ) -> List[int]:
        candidate_map = self._candidate_map(context, query_action, query_stage, query_policy)
        if not candidate_map:
            return []
        return sorted(candidate_map.keys())

    def _context_keys(self, context: Sequence[int]) -> List[Tuple[int, ...]]:
        if not context:
            return []
        keys: List[Tuple[int, ...]] = []
        for width in (2, 4, 8, self.window):
            if len(context) >= width:
                keys.append(tuple(context[-width:]))
        if not keys:
            keys.append(tuple(context[-min(len(context), self.window):]))
        return keys

    def _outcome_bucket(self, rec: EpisodeRecord) -> int:
        reward = float(rec.reward)
        if rec.metadata:
            reward += 0.2 * float(rec.metadata.get("plausibility", 0.0))
            reward += 0.2 * float(rec.metadata.get("token_agreement", 0.0))
            reward += 0.1 * float(rec.metadata.get("structural_score", 0.0))
        if reward < 0.25:
            return 0
        if reward < 0.50:
            return 1
        if reward < 0.75:
            return 2
        return 3

    def _resonance_score(
        self,
        context: Sequence[int],
        rec: EpisodeRecord,
        query_action: Optional[int],
        query_stage: Optional[str],
        query_policy: Optional[str],
    ) -> float:
        ctx_sim = self._context_similarity(context, rec.context)
        reward = float(rec.reward)
        outcome = reward
        if rec.metadata:
            outcome += 0.2 * float(rec.metadata.get("plausibility", 0.0))
            outcome += 0.2 * float(rec.metadata.get("token_agreement", 0.0))
            outcome += 0.15 * float(rec.metadata.get("structural_score", 0.0))
        outcome = max(0.0, outcome)
        stage_sim = 1.0 if query_stage and rec.stage == query_stage else 0.0
        policy_sim = 1.0 if query_policy and rec.policy == query_policy else 0.0
        action_sim = 1.0 if query_action is not None and int(rec.action) == int(query_action) else 0.0
        age = max(0, self._step - int(rec.timestamp))
        recency = float(np.exp(-age / max(1.0, float(self.window))))
        bucket_bonus = 0.1 * float(self._outcome_bucket(rec))
        return (
            0.40 * ctx_sim
            + 0.15 * action_sim
            + 0.15 * stage_sim
            + 0.10 * policy_sim
            + 0.15 * min(1.0, outcome)
            + 0.05 * recency
            + bucket_bonus
        )

    def _context_similarity(self, left: Sequence[int], right: Sequence[int]) -> float:
        if not left or not right:
            return 0.0
        widths = (2, 4, 8, self.window)
        total = 0.0
        weight_sum = 0.0
        for width in widths:
            if len(left) < width or len(right) < width:
                continue
            l = list(left[-width:])
            r = list(right[-width:])
            matches = sum(1 for a, b in zip(l, r) if a == b)
            total += (matches / max(1, width)) * width
            weight_sum += width
        if weight_sum <= 0.0:
            n = min(len(left), len(right), self.window)
            matches = sum(1 for a, b in zip(left[-n:], right[-n:]) if a == b)
            return matches / max(1, n)
        return total / weight_sum


class Reasoner:
    """Deliberative reasoning layer sitting above the HPM core."""

    def __init__(
        self,
        agent,
        dictionary: Optional[DictionaryValidator] = None,
        grammar: Optional[GrammarValidator] = None,
    ):
        self.agent = agent
        self.dictionary = dictionary
        self.grammar = grammar
        self.context_window = 40
        self.beam_width = 3
        self.candidate_top_k = 3
        self.memory_window = 24
        self.memory_capacity = 256
        self.memory_top_k = 6
        self.memory_weight = 0.25
        self.polygraph = EpisodicPolygraph(capacity=self.memory_capacity, window=self.memory_window)

    @property
    def memory_size(self) -> int:
        return len(self.polygraph)

    @property
    def memory(self) -> List[MemoryEpisode]:
        return self.polygraph.records

    @memory.setter
    def memory(self, value: Sequence[MemoryEpisode | Dict[str, Any]]) -> None:
        self.polygraph.set_records(value)

    def get_relevant_patterns(
        self,
        context_obs: List[int],
        top_k: int = 5,
        lookback: Optional[int] = None,
    ) -> List[HierarchicalPattern]:
        """Return patterns with highest predictive likelihood for given context."""
        if not context_obs:
            return sorted(self.agent.patterns, key=lambda p: p.weight, reverse=True)[:top_k]
        window = context_obs[-(lookback or self.context_window):]
        scores = []
        for p in self.agent.patterns:
            ll = p.log_likelihood(window)
            score = p.weight * np.exp(ll / max(1, len(window)))
            scores.append((p, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scores[:top_k]]

    def record_episode(
        self,
        context_obs: Sequence[int],
        action: int,
        reward: float,
        tag: str = "observe",
        metadata: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> None:
        """Store a compact trace for later retrieval."""
        context = list(context_obs[-self.memory_window:]) if context_obs else []
        meta = dict(metadata or {})
        stage_name = stage or str(meta.get("stage", getattr(getattr(self.agent, "development", None), "level", "unknown")))
        policy_name = policy or str(meta.get("policy", getattr(self.agent, "_last_decoder_choice", "unknown")))
        episode = EpisodeRecord(
            context=context,
            action=int(action),
            reward=float(reward),
            tag=tag,
            metadata=meta,
            stage=stage_name,
            policy=policy_name,
            timestamp=int(meta.get("timestamp", 0)),
        )
        self.polygraph.add(episode)

    def retrieve_memory(
        self,
        context_obs: Sequence[int],
        top_k: Optional[int] = None,
        min_reward: Optional[float] = None,
        query_action: Optional[int] = None,
    ) -> List[MemoryEpisode]:
        """Return best matching memory traces for the current context."""
        if not self.memory:
            return []

        top_k = top_k or self.memory_top_k
        query_stage, query_policy = self._current_control_signature()
        return self.polygraph.query(
            context_obs=context_obs,
            top_k=top_k,
            min_reward=min_reward,
            query_action=query_action,
            query_stage=query_stage,
            query_policy=query_policy,
        )

    def _memory_similarity(self, context_obs: Sequence[int], memory_ctx: Sequence[int]) -> float:
        return self.polygraph._context_similarity(context_obs, memory_ctx)

    def _memory_distribution(
        self,
        context_obs: Sequence[int],
        memory_hits: Optional[List[MemoryEpisode]] = None,
    ) -> np.ndarray:
        obs_dim = max(2, int(getattr(self.agent, "obs_dim", 2)))
        if memory_hits is None:
            memory_hits = self.retrieve_memory(context_obs, top_k=self.memory_top_k)
        if not memory_hits:
            return np.ones(obs_dim, dtype=np.float32) / float(obs_dim)

        dist = np.zeros(obs_dim, dtype=np.float32)
        total = 0.0
        ctx = list(context_obs[-self.memory_window:]) if context_obs else []
        for ep in memory_hits:
            sim = self._memory_similarity(ctx, ep.context)
            reward = float(ep.reward)
            if ep.metadata:
                reward += 0.2 * float(ep.metadata.get("plausibility", 0.0))
                reward += 0.2 * float(ep.metadata.get("token_agreement", 0.0))
                reward += 0.1 * float(ep.metadata.get("structural_score", 0.0))
            weight = max(0.05, sim) * max(0.1, reward + 1.0)
            dist[ep.action % obs_dim] += weight
            total += weight
        if total <= 0.0:
            return np.ones(obs_dim, dtype=np.float32) / float(obs_dim)
        dist /= total
        dist /= dist.sum() + 1e-12
        return dist.astype(np.float32)

    def control_context(
        self,
        context_obs: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarise episodic resonance for decoder/control selection."""
        if context_obs is None:
            context_obs = list(self.agent.obs_buffer[-self.context_window:]) if self.agent.obs_buffer else []
        graph_summary = self.polygraph.projection_summary(context_obs)
        hits = self.retrieve_memory(context_obs, top_k=top_k or self.memory_top_k)
        family_scores: Dict[str, float] = {}
        mode_scores: Dict[str, float] = {}
        stage_scores: Dict[str, float] = {}
        community_scores: Dict[str, float] = {}
        summary_count = 0

        for rec in hits:
            base = max(0.05, float(rec.reward))
            if rec.metadata:
                base += 0.2 * float(rec.metadata.get("plausibility", 0.0))
                base += 0.2 * float(rec.metadata.get("token_agreement", 0.0))
                base += 0.1 * float(rec.metadata.get("structural_score", 0.0))
                if rec.metadata.get("summary", False):
                    base += 0.15
            if rec.is_summary:
                summary_count += 1
            if rec.policy != "unknown":
                family_scores[rec.policy] = family_scores.get(rec.policy, 0.0) + base
            mode = str(rec.metadata.get("mode", "unknown")) if rec.metadata else "unknown"
            if mode != "unknown":
                mode_scores[mode] = mode_scores.get(mode, 0.0) + base
            if rec.stage != "unknown":
                stage_scores[rec.stage] = stage_scores.get(rec.stage, 0.0) + base
            if rec.community != "unknown":
                community_scores[rec.community] = community_scores.get(rec.community, 0.0) + base

        family_prior = self._normalize_scores(family_scores)
        mode_prior = self._normalize_scores(mode_scores)
        stage_prior = self._normalize_scores(stage_scores)
        dominant_family = max(family_prior, key=family_prior.get) if family_prior else None
        dominant_mode = max(mode_prior, key=mode_prior.get) if mode_prior else None
        dominant_stage = max(stage_prior, key=stage_prior.get) if stage_prior else None
        top_community = max(community_scores, key=community_scores.get) if community_scores else None
        community_strength = float(sum(community_scores.values()) / max(1, len(community_scores))) if community_scores else 0.0

        return {
            "family_prior": family_prior,
            "mode_prior": mode_prior,
            "stage_prior": stage_prior,
            "community_strength": community_strength,
            "dominant_family": dominant_family,
            "dominant_mode": dominant_mode,
            "dominant_stage": dominant_stage,
            "top_community": top_community,
            "summary_count": summary_count,
            "retrieved_count": len(hits),
            "graph_summary": graph_summary,
        }

    def _memory_bonus(
        self,
        context_obs: Sequence[int],
        action: int,
        memory_hits: Optional[List[MemoryEpisode]] = None,
    ) -> float:
        if memory_hits is None:
            memory_hits = self.retrieve_memory(context_obs, top_k=self.memory_top_k, query_action=action)
        if not memory_hits:
            return 0.0
        ctx = list(context_obs[-self.memory_window:]) if context_obs else []
        bonus = 0.0
        for ep in memory_hits:
            sim = self._memory_similarity(ctx, ep.context)
            if sim <= 0.0:
                continue
            if action == ep.action:
                bonus += self.memory_weight * sim * ep.reward
            else:
                bonus -= 0.05 * self.memory_weight * sim * max(0.0, ep.reward)
        return bonus

    def compose_predictions(self, patterns: List[HierarchicalPattern], obs_seq: List[int]) -> np.ndarray:
        """Combine multiple patterns by weighted averaging of their predictive distributions."""
        if not patterns:
            return np.array([0.5, 0.5])

        preds = []
        for p in patterns:
            dist = p.predict_next_distribution(obs_seq)
            preds.append((p.weight, dist))

        total_weight = sum(w for w, _ in preds) + 1e-12
        blended = np.zeros_like(preds[0][1])
        for w, d in preds:
            blended += (w / total_weight) * d
        return blended

    def counterfactual(self, pattern: HierarchicalPattern, obs_seq: List[int], intervention_idx: int):
        """
        Force a latent state or emission and observe the change in prediction.
        Returns (original_dist, intervened_dist).
        """
        orig_dist = pattern.predict_next_distribution(obs_seq)

        if not obs_seq:
            next_state = pattern.pi @ pattern.A
        else:
            alpha, _ = pattern._forward(obs_seq[-20:])
            next_state = alpha[-1] @ pattern.A

        intervened_dist = np.zeros(pattern.obs_dim, dtype=np.float32)
        intervened_dist[intervention_idx % pattern.obs_dim] = 1.0

        return orig_dist, intervened_dist

    def simulate(self, pattern: HierarchicalPattern, initial_obs_seq: List[int], steps: int = 10) -> List[int]:
        """Generate a possible future sequence using the pattern as a generative model."""
        if initial_obs_seq:
            alpha, _ = pattern._forward(initial_obs_seq[-20:])
            state_dist = alpha[-1]
        else:
            state_dist = pattern.pi.copy()

        simulated = []
        K = pattern.latent_dim
        for _ in range(steps):
            z = np.random.choice(K, p=state_dist / (state_dist.sum() + 1e-12))
            obs_probs = pattern.B[z]
            next_obs = np.random.choice(pattern.obs_dim, p=obs_probs / (obs_probs.sum() + 1e-12))
            simulated.append(int(next_obs))
            state_dist = pattern.A[z]
        return simulated

    def simulate_future(self, steps: int = 10, top_k: int = 3, lookback: Optional[int] = None) -> List[int]:
        """Generate imagined future sequence by sampling from blended population prediction."""
        context = list(self.agent.obs_buffer[-(lookback or self.context_window):]) if self.agent.obs_buffer else []
        simulated: List[int] = []

        for _ in range(steps):
            relevant = self.get_relevant_patterns(context, top_k=top_k, lookback=lookback)
            if not relevant:
                simulated.append(0)
                continue
            dist = self.compose_predictions(relevant, context)
            memory_hits = self.retrieve_memory(context, top_k=self.memory_top_k)
            if memory_hits:
                mem_dist = self._memory_distribution(context, memory_hits=memory_hits)
                alpha = min(0.35, 0.05 * len(memory_hits))
                dist = (1.0 - alpha) * dist + alpha * mem_dist
            dist = dist / (dist.sum() + 1e-12)
            obs = int(np.random.choice(len(dist), p=dist))
            simulated.append(obs)
            context.append(obs)
            if len(context) > 40:
                context = context[-40:]

        return simulated

    def _candidate_actions(self, dist: np.ndarray, top_k: int) -> List[int]:
        if dist.size == 0:
            return []
        top = np.argsort(dist)[::-1][:max(1, top_k)]
        return [int(i) for i in top]

    def _action_bonus(
        self,
        context_obs: List[int],
        action: int,
        goal_state: Optional[int],
        require_valid_words: bool,
        require_grammatical: bool,
    ) -> float:
        bonus = 0.0
        if goal_state is not None:
            span = max(1, self.agent.obs_dim - 1)
            bonus += 0.35 * (1.0 - abs(action - goal_state) / span)
        if self._is_word_boundary(context_obs, action):
            if self.grammar and require_grammatical:
                prev_word = self._last_word(context_obs)
                curr_word = self._partial_word(context_obs, action)
                if prev_word and curr_word:
                    bonus += 0.25 if self.grammar.is_valid_transition(prev_word, curr_word) else -0.25
        else:
            if self.dictionary and require_valid_words:
                word_so_far = self._partial_word(context_obs, action)
                if word_so_far:
                    bonus += 0.15 if self.dictionary.is_prefix(word_so_far) else -0.25
        return bonus

    def _target_sequence_bonus(
        self,
        action: int,
        step_idx: int,
        target_sequence: Optional[List[int]],
        target_weight: float,
    ) -> float:
        """Reward actions that match an explicit target continuation."""
        if not target_sequence:
            return 0.0
        if step_idx >= len(target_sequence):
            expected = target_sequence[-1]
            tail_penalty = 0.05 * (step_idx - len(target_sequence) + 1)
        else:
            expected = target_sequence[step_idx]
            tail_penalty = 0.0

        if expected is None:
            return -tail_penalty

        if action == expected:
            return target_weight - tail_penalty
        if self.agent.obs_dim > 1 and abs(action - expected) == 1:
            return target_weight * 0.35 - tail_penalty
        return -target_weight * 0.25 - tail_penalty

    def plan_sequence(
        self,
        target_sequence: List[int],
        horizon: Optional[int] = None,
        require_valid_words: bool = True,
        require_grammatical: bool = True,
        strategy: str = "beam",
        beam_width: Optional[int] = None,
        candidate_top_k: Optional[int] = None,
        lookback: Optional[int] = None,
        target_weight: float = 2.5,
    ) -> List[int]:
        """Plan against an explicit target continuation."""
        horizon = horizon if horizon is not None else len(target_sequence)
        return self.plan(
            goal_state=target_sequence[-1] if target_sequence else None,
            horizon=horizon,
            num_rollouts=max(10, horizon),
            require_valid_words=require_valid_words,
            require_grammatical=require_grammatical,
            strategy=strategy,
            beam_width=beam_width,
            candidate_top_k=candidate_top_k,
            lookback=lookback,
            target_sequence=target_sequence,
            target_weight=target_weight,
        )

    def plan(
        self,
        goal_state: Optional[int],
        horizon: int = 5,
        num_rollouts: int = 10,
        require_valid_words: bool = True,
        require_grammatical: bool = True,
        strategy: str = "beam",
        beam_width: Optional[int] = None,
        candidate_top_k: Optional[int] = None,
        lookback: Optional[int] = None,
        target_sequence: Optional[List[int]] = None,
        target_weight: float = 2.5,
    ) -> List[int]:
        """
        Search for a sequence of observations that reaches the goal state or matches a target continuation.
        Beam search is the default; stochastic rollouts are available as a fallback.
        """
        if strategy not in {"beam", "rollout"}:
            raise ValueError(f"Unsupported planning strategy: {strategy!r}")

        best_seq = []
        best_score = -np.inf

        curr_obs_base = list(self.agent.obs_buffer[-(lookback or self.context_window):]) if self.agent.obs_buffer else []
        beam_width = beam_width or self.beam_width
        candidate_top_k = candidate_top_k or self.candidate_top_k

        if strategy == "rollout":
            for _ in range(num_rollouts):
                seq = []
                curr_obs = list(curr_obs_base)
                seq_score = 0.0
                for _ in range(horizon):
                    relevant = self.get_relevant_patterns(curr_obs, top_k=3, lookback=lookback)
                    if not relevant:
                        break

                    dist = self.compose_predictions(relevant, curr_obs)
                    memory_hits = self.retrieve_memory(curr_obs, top_k=self.memory_top_k)
                    if memory_hits:
                        mem_dist = self._memory_distribution(curr_obs, memory_hits=memory_hits)
                        alpha = min(0.35, 0.05 * len(memory_hits))
                        dist = (1.0 - alpha) * dist + alpha * mem_dist
                    dist = dist / (dist.sum() + 1e-12)
                    action = int(np.random.choice(len(dist), p=dist))
                    seq_score += float(np.log(dist[action] + 1e-12))
                    seq_score += self._action_bonus(curr_obs, action, goal_state, require_valid_words, require_grammatical)
                    seq_score += self._target_sequence_bonus(action, len(seq), target_sequence, target_weight)
                    seq_score += self._memory_bonus(curr_obs, action, memory_hits=memory_hits)
                    seq.append(action)
                    curr_obs.append(action)

                if seq:
                    if goal_state is not None:
                        seq_score += -abs(seq[-1] - goal_state)
                    if target_sequence:
                        overlap = sum(1 for a, b in zip(seq, target_sequence) if a == b)
                        seq_score += target_weight * (overlap / max(1, len(target_sequence)))
                    if self._word_completed(curr_obs):
                        last_word = self._last_word(curr_obs)
                        if self.dictionary:
                            seq_score += 0.5 * self.dictionary.score_word(last_word)
                        if self.grammar:
                            prev_word = self._word_before_last(curr_obs)
                            if prev_word and last_word and self.grammar.is_valid_transition(prev_word, last_word):
                                seq_score += 0.3
                    if seq_score > best_score:
                        best_score = seq_score
                        best_seq = seq
            return best_seq

        beams = [([], list(curr_obs_base), 0.0)]
        for _ in range(horizon):
            expanded = []
            for seq, curr_obs, seq_score in beams:
                relevant = self.get_relevant_patterns(curr_obs, top_k=3, lookback=lookback)
                if not relevant:
                    continue
                dist = self.compose_predictions(relevant, curr_obs)
                memory_hits = self.retrieve_memory(curr_obs, top_k=self.memory_top_k)
                if memory_hits:
                    mem_dist = self._memory_distribution(curr_obs, memory_hits=memory_hits)
                    alpha = min(0.35, 0.05 * len(memory_hits))
                    dist = (1.0 - alpha) * dist + alpha * mem_dist
                dist = dist / (dist.sum() + 1e-12)
                for action in self._candidate_actions(dist, candidate_top_k):
                    next_seq = seq + [action]
                    next_obs = curr_obs + [action]
                    next_score = seq_score + float(np.log(dist[action] + 1e-12))
                    next_score += self._action_bonus(curr_obs, action, goal_state, require_valid_words, require_grammatical)
                    next_score += self._target_sequence_bonus(action, len(seq), target_sequence, target_weight)
                    next_score += self._memory_bonus(curr_obs, action, memory_hits=memory_hits)
                    next_score += -0.05 * len(next_seq)
                    expanded.append((next_seq, next_obs, next_score))

            if not expanded:
                break
            expanded.sort(key=lambda item: item[2], reverse=True)
            beams = expanded[:beam_width]

        for seq, curr_obs, seq_score in beams:
            if not seq:
                continue
            final_score = seq_score
            if goal_state is not None:
                final_score += -abs(seq[-1] - goal_state)
            if target_sequence:
                overlap = sum(1 for a, b in zip(seq, target_sequence) if a == b)
                final_score += target_weight * (overlap / max(1, len(target_sequence)))
            if self._word_completed(curr_obs):
                last_word = self._last_word(curr_obs)
                if self.dictionary:
                    final_score += 0.5 * self.dictionary.score_word(last_word)
                if self.grammar:
                    prev_word = self._word_before_last(curr_obs)
                    if prev_word and last_word and self.grammar.is_valid_transition(prev_word, last_word):
                        final_score += 0.3
            if final_score > best_score:
                best_score = final_score
                best_seq = seq

        return best_seq

    def _is_word_boundary(self, obs_seq, next_obs) -> bool:
        """Space (approx token 0 in TextAdapter or 2 in CharClass) indicates boundary."""
        if not obs_seq:
            return True
        return obs_seq[-1] in (0, 2) and next_obs not in (0, 2)

    def _partial_word(self, obs_seq, next_obs) -> str:
        """Reconstruct partial word from recent tokens."""
        tokens = obs_seq + [next_obs]
        last_space = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in (0, 2):
                last_space = i
                break

        word_tokens = tokens[last_space + 1 :]
        if not word_tokens:
            return ""

        chars = []
        for t in word_tokens:
            if t < 256:
                chars.append(chr(t + 32))
            else:
                chars.append("?")
        return "".join(chars)

    def _word_completed(self, obs_seq) -> bool:
        """True if last token was a space or punctuation."""
        if not obs_seq:
            return False
        return obs_seq[-1] in (0, 2)

    def _last_word(self, obs_seq) -> str:
        if not obs_seq:
            return ""
        idx = -1
        for i in range(len(obs_seq) - 2, -1, -1):
            if obs_seq[i] in (0, 2):
                idx = i
                break
        return self._partial_word(obs_seq[:idx + 1], obs_seq[idx + 1] if idx + 1 < len(obs_seq) else 0)

    def _word_before_last(self, obs_seq) -> str:
        """Extract the word before the last completed word."""
        if not obs_seq:
            return ""
        spaces = [i for i, t in enumerate(obs_seq) if t in (0, 2)]
        if len(spaces) < 2:
            return ""

        last_space = spaces[-1]
        second_last_space = spaces[-2]

        word_tokens = obs_seq[second_last_space + 1 : last_space]
        if not word_tokens:
            return ""

        chars = [chr(t + 32) if t < 256 else "?" for t in word_tokens]
        return "".join(chars).lower()

    def observe_outcome(
        self,
        actual_obs: int,
        context_obs: List[int],
        lambda_l: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Feed actual observation back as per-pattern prediction error."""
        errors: List[float] = []
        for p in self.agent.patterns:
            dist = p.predict_next_distribution(context_obs)
            error = float(-np.log(dist[actual_obs % p.obs_dim] + 1e-12))
            errors.append(error)
            p.running_loss = (1 - lambda_l) * p.running_loss + lambda_l * error

        avg_error = float(np.mean(errors)) if errors else 0.0
        reward = 1.0 / (1.0 + avg_error)

        meta = dict(metadata or {})
        if meta.get("parseable", False):
            reward = min(1.0, reward + 0.03)
        if meta.get("execution_match", False):
            reward = min(1.0, reward + 0.12)
        if meta.get("canonical_match", False):
            reward = min(1.0, reward + 0.05)
        if meta.get("syntax_error", False):
            reward = max(0.0, reward - 0.08)
        if meta.get("semantic_mismatch", False):
            reward = max(0.0, reward - 0.06)

        if self.dictionary and self._word_completed(context_obs + [actual_obs]):
            word = self._last_word(context_obs + [actual_obs])
            if word:
                lex_score = self.dictionary.score_word(word)
                bonus = (lex_score - 0.5) * 0.1
                reward = max(0.0, reward + bonus)
                for p in self.agent.patterns:
                    p.running_loss = max(0.0, p.running_loss - bonus)

        if self.grammar and self._word_completed(context_obs + [actual_obs]):
            prev_word = self._word_before_last(context_obs + [actual_obs])
            curr_word = self._last_word(context_obs + [actual_obs])
            if prev_word and curr_word:
                gram_score = self.grammar.score_sequence([prev_word, curr_word])
                bonus = (gram_score - 0.5) * 0.08
                reward = max(0.0, reward + bonus)
                for p in self.agent.patterns:
                    p.running_loss = max(0.0, p.running_loss - bonus)

        self.record_episode(
            context_obs,
            actual_obs,
            reward,
            tag="observe",
            metadata={
                "stage": getattr(getattr(self.agent, "development", None), "level", "unknown"),
                "policy": getattr(self.agent, "_last_decoder_choice", "unknown"),
                "token_agreement": 0.0,
                "plausibility": self._lexical_plausibility(context_obs + [actual_obs]),
                **meta,
            },
        )

    def explain(self, pattern: HierarchicalPattern) -> str:
        """Translate pattern structure into human-readable description."""
        if pattern.complexity >= 1:
            likely_obs = np.argmax(pattern.B[np.argmax(pattern.pi)])
            return (
                f"Hierarchical pattern (ID:{pattern.id}) predicts observation {likely_obs} "
                f"via {pattern.latent_dim} latent states."
            )
        else:
            return f"Pattern (ID:{pattern.id}) is unknown."

    def state_dict(self) -> Dict[str, Any]:
        return {
            "context_window": self.context_window,
            "beam_width": self.beam_width,
            "candidate_top_k": self.candidate_top_k,
            "memory_window": self.memory_window,
            "memory_capacity": self.memory_capacity,
            "memory_top_k": self.memory_top_k,
            "memory_weight": self.memory_weight,
            "memory": [ep.to_dict() for ep in self.memory],
            "polygraph": self.polygraph.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.context_window = int(state.get("context_window", self.context_window))
        self.beam_width = int(state.get("beam_width", self.beam_width))
        self.candidate_top_k = int(state.get("candidate_top_k", self.candidate_top_k))
        self.memory_window = int(state.get("memory_window", self.memory_window))
        self.memory_capacity = int(state.get("memory_capacity", self.memory_capacity))
        self.memory_top_k = int(state.get("memory_top_k", self.memory_top_k))
        self.memory_weight = float(state.get("memory_weight", self.memory_weight))
        if "polygraph" in state:
            self.polygraph.load_state_dict(state.get("polygraph", {}))
        else:
            self.memory = [MemoryEpisode.from_dict(item) for item in state.get("memory", [])]
        self.polygraph.capacity = self.memory_capacity
        self.polygraph.window = self.memory_window
        self.polygraph._trim()
        self.polygraph._rebuild_indices()

    def _current_control_signature(self) -> Tuple[Optional[str], Optional[str]]:
        stage = getattr(getattr(self.agent, "development", None), "level", None)
        policy = getattr(self.agent, "_last_decoder_choice", None)
        if stage is not None:
            stage = str(stage)
            if stage == "unknown":
                stage = None
        if policy is not None:
            policy = str(policy)
            if policy == "unknown":
                policy = None
        return stage, policy

    def _lexical_plausibility(self, obs_seq: Sequence[int]) -> float:
        tokens = [str(v) for v in obs_seq]
        scores: List[float] = []
        if self.dictionary:
            for tok in tokens:
                scores.append(self.dictionary.score_word(tok))
        if self.grammar and len(tokens) > 1:
            prev = None
            for tok in tokens:
                if prev is not None:
                    scores.append(1.0 if self.grammar.is_valid_transition(prev, tok) else 0.0)
                prev = tok
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(max(0.0, v) for v in scores.values()))
        if total <= 0.0:
            return {}
        return {key: max(0.0, value) / total for key, value in scores.items() if value > 0.0}
