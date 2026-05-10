"""ATIS-specific agents for v5 multi-agent pipelines."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable
from collections import defaultdict

from .base import BaseAgent, AgentInput, AgentOutput
from .packet import AgentPacket
from ..pipeline import HPMPipeline
from ..adapter import AdapterPacket
from ..core import PatternEngine, PatternManager, PolygraphPatternRetriever
from ..core.store import MatchResult
from ..polygraphs.base import PolygraphView
from ..adapter.comprehension_feedback import ComprehensionFeedbackPostprocessor


@dataclass
class ATISRouterAgent:
    """Routes ATIS queries by identifying task domain (e.g., flight, fare, city)."""
    name: str = "atis_router"

    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        raw = packet.raw_input.lower()
        
        # Heuristic routing
        if any(w in raw for w in ["flight", "fly", "plane", "airline"]):
            route = "flight_info"
        elif any(w in raw for w in ["fare", "cost", "price", "expensive", "cheap"]):
            route = "financial_info"
        elif any(w in raw for w in ["airport", "ground", "transport", "limo"]):
            route = "ground_info"
        else:
            route = "general_query"
            
        packet.context["atis_route"] = route
        packet.agent_trace.append(self.name)
        packet.log(self.name, {"route": route}, role="agent")
        return packet

    def step(self, packet: AgentPacket) -> AgentPacket:
        return self.step_packet(packet)


@dataclass
class ATISIntentAgent(BaseAgent):
    """Specialist agent for intent recognition."""
    
    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        # Carry over the route as a goal or context bias
        route = packet.context.get("atis_route", "general")
        packet.context["intent_specialist_mode"] = route
        
        # Execute the standard HPM pipeline step via BaseAgent
        return super().step_packet(packet)


@dataclass
class PatternOverseerAgent:
    """Agent that oversees pattern health, triggering consolidation and pruning."""
    manager: PatternManager
    name: str = "pattern_overseer"

    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        # The overseer acts on the engine/store globally, typically at episode boundaries.
        if packet.context.get("end_episode", False):
            # Find an engine to operate on. Usually passed in packet context or retrieved from previous steps.
            # For ATIS, we look at the last core action's engine if available.
            engine = None
            if packet.core_actions:
                # Assuming the last core action is a PipelineResult which has an engine reference
                # or we can pass it explicitly. For now, we try to find it.
                res = packet.core_actions[-1]
                if hasattr(res, 'engine'):
                    engine = res.engine

            if engine:
                packet.log(self.name, {"action": "consolidating_patterns"}, role="agent")
                self.manager.end_episode(engine, context=packet.context)
        
        packet.agent_trace.append(self.name)
        return packet

    def step(self, packet: AgentPacket) -> AgentPacket:
        return self.step_packet(packet)


@dataclass
class ATISInferenceAgent:
    """Pure-inference agent: runs preprocessing + store.match() only — no learning.

    Designed for the '1 training - 1 testing' split where this agent handles
    the testing phase using patterns discovered by a training specialist.
    """
    view_engines: dict[str, PatternEngine]
    pattern_intent: dict[str, str]
    pipeline: HPMPipeline
    feedback_postprocessor: ComprehensionFeedbackPostprocessor = field(default_factory=ComprehensionFeedbackPostprocessor)
    carry_context: dict[str, Any] = field(default_factory=dict)
    name: str = "atis_intent_inference"

    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        packet.context.update(self.carry_context)
        adapter_packet, views = _preprocess_atis_packet(self.pipeline, packet)
        packet.context.update(adapter_packet.context)
        matcher = ViewStoreMatcher(self.view_engines)
        primary_match = self.pipeline.engine.store.match(adapter_packet.states[-1].value)
        matches = matcher.collect(self.pipeline, views, primary_match)
        prediction, intent_votes, evidence = FlatIntentStrategy(
            self.pattern_intent,
            feedback_context=self.carry_context,
            use_feedback_reweighting=False,
            use_kb_bias=False,
        ).predict(matches)
        
        packet.context["predicted_intent"] = prediction
        packet.context["intent_votes"] = dict(intent_votes)
        packet.context["intent_vote_trace"] = evidence
        packet.final_output = prediction
        packet.agent_trace.append(self.name)
        
        # Track if any hit was a variant for B4 metrics
        has_variant = any(match.status == "variant" for _, match in matches if match)
        
        packet.log(self.name, {
            "intent": prediction, 
            "votes": dict(intent_votes),
            "vote_trace": evidence,
            "match_status": primary_match.status if primary_match else "none",
            "has_variant_hit": has_variant
        }, role="agent")
        feedback_packet = AdapterPacket(raw=packet.raw_input, context=dict(packet.context))
        feedback_packet = self.feedback_postprocessor.run(feedback_packet)
        self.carry_context = {k[6:]: v for k, v in feedback_packet.context.items() if k.startswith("carry_")}
        packet.context.update(self.carry_context)
        return packet

    def step(self, packet: AgentPacket) -> AgentPacket:
        return self.step_packet(packet)


def _match_target_name(match: MatchResult | None) -> tuple[str | None, float]:
    if not match:
        return None, 0.0
    if match.status == "variant" and match.variant:
        return match.variant.name, match.variant.score
    if match.pattern:
        return match.pattern.name, match.pattern.utility
    return None, 0.0


def _match_weight(status: str) -> float:
    if status == "exact":
        return 1.0
    if status == "near":
        return 0.55
    if status == "variant":
        return 0.35
    return 0.0


def _effective_source_weights(
    base_weights: dict[str, float],
    carry_context: dict[str, Any],
) -> dict[str, float]:
    weights = dict(base_weights)
    top_source = carry_context.get("intent_feedback_top_source")
    if not top_source:
        return weights
    penalty = float(carry_context.get("intent_feedback_top_source_penalty", 1.0))
    if penalty <= 0.0 or penalty >= 1.0:
        return weights
    prior = weights.get(str(top_source), 1.0)
    weights[str(top_source)] = max(0.1, prior * penalty)
    return weights


def _kb_intent_factor(intent: str, carry_context: dict[str, Any]) -> tuple[float, bool]:
    if not bool(carry_context.get("kb_error_supported", False)):
        return 1.0, False
    score = float(carry_context.get("kb_error_score", 0.0))
    expected = {
        str(item)
        for item in carry_context.get("kb_expected_intents", ())
        if item is not None
    }
    if not expected:
        return 1.0, False
    if intent in expected:
        return 1.0 + (0.35 * score), True
    return max(0.5, 1.0 - (0.2 * score)), False


def _preprocess_atis_packet(pipeline: HPMPipeline, packet: AgentPacket) -> tuple[AdapterPacket, list[PolygraphView]]:
    route = packet.context.get("atis_route", "general")
    packet.context["intent_specialist_mode"] = route

    adapters = pipeline.preprocessing_pipeline.adapters
    for adapter in adapters.values():
        if hasattr(adapter, "reset"):
            adapter.reset()

    adapter_packet = AdapterPacket(raw=packet.raw_input, context=packet.context)
    adapter_packet = pipeline.preprocessing_pipeline.run(
        adapter_packet,
        target_outputs=list(adapters.keys()),
    )
    views: list[PolygraphView] = []
    if pipeline.polygraph_generator:
        views = pipeline.polygraph_generator.generate(
            packet.raw_input, context=adapter_packet.context
        )
    return adapter_packet, views


MatchList = list[tuple[str, MatchResult | None]]


@dataclass(slots=True)
class ViewStoreMatcher:
    """Collect primary and view-level store matches for ATIS inference."""

    view_engines: dict[str, PatternEngine]

    def collect(
        self,
        pipeline: HPMPipeline,
        views: Iterable[PolygraphView],
        primary_match: MatchResult | None,
    ) -> MatchList:
        view_stores = {name: eng.store for name, eng in self.view_engines.items()}
        matches: MatchList = [("primary", primary_match)]
        for view in views:
            store = view_stores.get(view.name, pipeline.engine.store)
            matches.append((view.name, store.match(view.state.value)))
        return matches


@dataclass(slots=True)
class FlatIntentStrategy:
    """Baseline ATIS voting across primary + view matches."""

    pattern_intent: dict[str, str]
    near_weight: float = 0.5
    variant_weight: float = 0.5
    source_weights: dict[str, float] = field(default_factory=dict)
    feedback_context: dict[str, Any] = field(default_factory=dict)
    use_feedback_reweighting: bool = False
    use_kb_bias: bool = False

    def predict(self, matches: MatchList) -> tuple[str | None, dict[str, float], list[dict[str, object]]]:
        intent_votes: dict[str, float] = defaultdict(float)
        evidence: list[dict[str, object]] = []
        effective_source_weights = (
            _effective_source_weights(self.source_weights, self.feedback_context)
            if self.use_feedback_reweighting else dict(self.source_weights)
        )
        route_hint = self.feedback_context.get("intent_feedback_route_hint") if self.use_feedback_reweighting else None
        route_boost = float(self.feedback_context.get("intent_feedback_route_boost", 1.0)) if self.use_feedback_reweighting else 1.0
        for source, match in matches:
            target_name, utility = _match_target_name(match)
            if not target_name:
                continue
            weight = (
                1.0 if match.status == "exact"
                else self.near_weight if match.status == "near"
                else self.variant_weight if match.status == "variant"
                else 0.0
            )
            if weight <= 0.0:
                continue
            intent = self.pattern_intent.get(target_name)
            if intent:
                source_weight = effective_source_weights.get(source, 1.0)
                contribution = source_weight * weight * max(utility, 0.1)
                route_aligned = bool(route_hint and intent == route_hint)
                if route_aligned and route_boost > 1.0:
                    contribution *= route_boost
                kb_factor, kb_expected = _kb_intent_factor(intent, self.feedback_context) if self.use_kb_bias else (1.0, False)
                contribution *= kb_factor
                intent_votes[intent] += contribution
                evidence.append({
                    "source": source,
                    "status": match.status,
                    "target_name": target_name,
                    "intent": intent,
                    "utility": utility,
                    "weight": weight,
                    "source_weight": source_weight,
                    "route_aligned": route_aligned,
                    "route_boost": route_boost if route_aligned else 1.0,
                    "kb_expected": kb_expected,
                    "kb_factor": kb_factor,
                    "contribution": contribution,
                })
        prediction = max(intent_votes, key=intent_votes.__getitem__) if intent_votes else None
        return prediction, dict(intent_votes), evidence


@dataclass(slots=True)
class AnchorIntentStrategy:
    """Bridge-aware ATIS scoring through shared anchors across views."""

    pattern_intent: dict[str, str]
    feedback_context: dict[str, Any] = field(default_factory=dict)
    use_dialogue_anchor_scoring: bool = False

    def _anchor_factor(self, anchor_id: str, intent: str) -> float:
        if not self.use_dialogue_anchor_scoring:
            return 1.0
        factor = 1.0
        if anchor_id.startswith("dialogue::route::"):
            hinted_intent = anchor_id.rsplit("::", 1)[-1]
            factor *= 1.35 if intent == hinted_intent else 0.85
        elif anchor_id.startswith("dialogue::focus::"):
            focused_intent = anchor_id.rsplit("::", 1)[-1]
            factor *= 1.2 if intent == focused_intent else 0.9
        elif anchor_id.startswith("dialogue::kb_expected::"):
            expected_intent = anchor_id.rsplit("::", 1)[-1]
            factor *= 1.5 if intent == expected_intent else 0.75
        elif anchor_id == "dialogue::kb_contradiction":
            kb_factor, _ = _kb_intent_factor(intent, self.feedback_context)
            factor *= kb_factor
        elif anchor_id == "dialogue::low_margin":
            if intent == self.feedback_context.get("intent_feedback_focus_intent"):
                factor *= 1.1
        return factor

    def predict(
        self,
        primary_match: MatchResult | None,
        views: list[PolygraphView],
        store: PatternEngine,
    ) -> tuple[str | None, dict[str, float], dict[str, dict[str, float]], bool]:
        intent_votes: dict[str, float] = defaultdict(float)
        anchor_support: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        multi_view_hits = False

        primary_name, primary_utility = _match_target_name(primary_match)
        primary_intent = self.pattern_intent.get(primary_name) if primary_name else None
        if primary_intent:
            primary_weight = _match_weight(primary_match.status) * max(primary_utility, 0.1)
            intent_votes[primary_intent] += primary_weight
            anchor_support["intent::primary"][primary_intent] += primary_weight

        for view in views:
            match = store.store.match(view.state.value)
            target_name, utility = _match_target_name(match)
            if not target_name:
                continue
            intent = self.pattern_intent.get(target_name)
            if not intent:
                continue
            base_weight = _match_weight(match.status)
            if base_weight <= 0.0:
                continue
            utility = max(utility, 0.1)
            source_weight = 1.0
            if view.name == "content_view":
                source_weight = 1.25
            elif view.name == "skeleton_bigram_view":
                source_weight = 1.1
            anchor_ids = view.anchor_ids or ("intent::utterance",)
            for anchor_id in anchor_ids:
                anchor_factor = self._anchor_factor(anchor_id, intent)
                anchor_support[anchor_id][intent] += base_weight * source_weight * utility * anchor_factor

        for anchor_id, scores in anchor_support.items():
            if not scores:
                continue
            supporting_views = {view.name for view in views if anchor_id in view.anchor_ids}
            boost = 1.2 if len(supporting_views) >= 2 else 1.0
            if boost > 1.0:
                multi_view_hits = True
            for intent, score in scores.items():
                intent_votes[intent] += score * boost

        prediction = max(intent_votes, key=intent_votes.__getitem__) if intent_votes else None
        return prediction, dict(intent_votes), {k: dict(v) for k, v in anchor_support.items()}, multi_view_hits


@dataclass(slots=True)
class InterconnectedATISInferenceAgent:
    """Inference agent that aggregates support through shared bridge anchors."""

    pattern_intent: dict[str, str]
    pipeline: HPMPipeline
    retriever: PolygraphPatternRetriever | None = None
    allow_retriever_override: bool = False
    feedback_postprocessor: ComprehensionFeedbackPostprocessor = field(default_factory=ComprehensionFeedbackPostprocessor)
    carry_context: dict[str, Any] = field(default_factory=dict)
    name: str = "atis_interconnected_inference"

    def step_packet(self, packet: AgentPacket, *, use_retriever: bool = True, use_concept_scoring: bool = False) -> AgentPacket:
        packet.context.update(self.carry_context)
        adapter_packet, views = _preprocess_atis_packet(self.pipeline, packet)
        packet.context.update(adapter_packet.context)
        primary_state = adapter_packet.states[-1].value if adapter_packet.states else ()
        primary_match = self.pipeline.engine.store.match(primary_state)
        if use_retriever and self.retriever is not None:
            self.retriever.pattern_intent = dict(self.pattern_intent)
            self.retriever.use_concept_scoring = use_concept_scoring
            retriever_candidates = self.retriever.retrieve(views, top_k=3)
        else:
            retriever_candidates = []
        prediction, votes, anchor_support, multi_view_hits = AnchorIntentStrategy(
            self.pattern_intent,
            feedback_context=self.carry_context,
            use_dialogue_anchor_scoring=False,
        ).predict(primary_match, views, self.pipeline.engine)
        anchor_only_prediction = prediction
        if retriever_candidates and self.allow_retriever_override:
            top_candidate = retriever_candidates[0]
            retriever_intent = self.pattern_intent.get(top_candidate.pattern_name)
            if retriever_intent:
                votes[retriever_intent] = votes.get(retriever_intent, 0.0) + top_candidate.total_score
                prediction = max(votes, key=votes.__getitem__) if votes else prediction

        packet.context["predicted_intent"] = prediction
        packet.context["anchor_only_predicted_intent"] = anchor_only_prediction
        packet.context["anchor_support"] = anchor_support
        packet.context["multi_view_anchor_hit"] = multi_view_hits
        packet.context["retriever_enabled"] = use_retriever
        packet.context["retriever_override_enabled"] = self.allow_retriever_override
        packet.context["concept_scoring_enabled"] = use_concept_scoring
        packet.context["retriever_changed_prediction"] = prediction != anchor_only_prediction
        packet.context["polygraph_candidates"] = [
            {
                "pattern_name": candidate.pattern_name,
                "total_score": candidate.total_score,
                "agreement_count": candidate.agreement_count,
                "view_scores": candidate.view_scores,
                "anchor_support": candidate.anchor_support,
                "concept_support": candidate.concept_support,
                "concept_overlap_count": candidate.concept_overlap_count,
                "specific_concept_count": candidate.specific_concept_count,
            }
            for candidate in retriever_candidates
        ]
        packet.final_output = prediction
        packet.agent_trace.append(self.name)
        packet.log(self.name, {
            "intent": prediction,
            "anchor_only_intent": anchor_only_prediction,
            "votes": votes,
            "anchors": anchor_support,
            "polygraph_candidates": packet.context["polygraph_candidates"],
            "retriever_enabled": use_retriever,
            "retriever_override_enabled": self.allow_retriever_override,
            "concept_scoring_enabled": use_concept_scoring,
            "retriever_changed_prediction": packet.context["retriever_changed_prediction"],
            "multi_view_anchor_hit": multi_view_hits,
        }, role="agent")
        feedback_packet = AdapterPacket(raw=packet.raw_input, context=dict(packet.context))
        feedback_packet = self.feedback_postprocessor.run(feedback_packet)
        self.carry_context = {k[6:]: v for k, v in feedback_packet.context.items() if k.startswith("carry_")}
        packet.context.update(self.carry_context)
        return packet


@dataclass(slots=True)
class AuditableIntentInferenceAgent:
    """Shared baseline intent inference with auditable vote traces."""

    view_engines: dict[str, PatternEngine]
    pattern_intent: dict[str, str]
    pipeline: HPMPipeline
    near_weight: float = 0.5
    variant_weight: float = 0.5
    source_weights: dict[str, float] = field(default_factory=dict)
    feedback_postprocessor: ComprehensionFeedbackPostprocessor = field(default_factory=ComprehensionFeedbackPostprocessor)
    carry_context: dict[str, Any] = field(default_factory=dict)
    name: str = "auditable_intent_inference"

    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        packet.context.update(self.carry_context)
        adapter_packet, views = _preprocess_atis_packet(self.pipeline, packet)
        packet.context.update(adapter_packet.context)
        matcher = ViewStoreMatcher(self.view_engines)
        primary_match = self.pipeline.engine.store.match(adapter_packet.states[-1].value)
        matches = matcher.collect(self.pipeline, views, primary_match)
        prediction, intent_votes, evidence = FlatIntentStrategy(
            self.pattern_intent,
            near_weight=self.near_weight,
            variant_weight=self.variant_weight,
            source_weights=self.source_weights,
            feedback_context=self.carry_context,
            use_feedback_reweighting=False,
            use_kb_bias=False,
        ).predict(matches)

        packet.context["predicted_intent"] = prediction
        packet.context["intent_votes"] = dict(intent_votes)
        packet.context["intent_vote_trace"] = evidence
        packet.final_output = prediction
        packet.agent_trace.append(self.name)

        has_variant = any(match.status == "variant" for _, match in matches if match)
        packet.log(self.name, {
            "intent": prediction,
            "votes": dict(intent_votes),
            "vote_trace": evidence,
            "match_status": primary_match.status if primary_match else "none",
            "has_variant_hit": has_variant,
        }, role="agent")
        feedback_packet = AdapterPacket(raw=packet.raw_input, context=dict(packet.context))
        feedback_packet = self.feedback_postprocessor.run(feedback_packet)
        self.carry_context = {k[6:]: v for k, v in feedback_packet.context.items() if k.startswith("carry_")}
        packet.context.update(self.carry_context)
        return packet

    def step(self, packet: AgentPacket) -> AgentPacket:
        return self.step_packet(packet)

    def step(self, packet: AgentPacket) -> AgentPacket:
        return self.step_packet(packet)
