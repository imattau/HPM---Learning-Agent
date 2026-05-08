"""Minimal end-to-end pipeline for v5."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from .adapter import AdapterPacket, AdapterRegistry
from .core import Action, PatternEngine
from .core.store import MatchResult
from .core.evaluator import PolygraphAgreement, PolygraphEvaluator, PolygraphScore
from .preprocessors.base import PreprocessedInput, Preprocessor
from .polygraphs.base import PolygraphGenerator
from .postprocessors.base import Postprocessor
from .core.state import State


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Structured result from preprocessing through output."""

    input: PreprocessedInput
    action: Action
    output: Any | None
    polygraph_scores: dict[str, PolygraphScore] | None = None
    polygraph_agreement: PolygraphAgreement | None = None
    carry_context: dict = field(default_factory=dict)


class HPMPipeline:
    """Preprocess → core → postprocess."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        engine: PatternEngine,
        postprocessor: Postprocessor,
        *,
        polygraph_generator: PolygraphGenerator | None = None,
        polygraph_evaluator: PolygraphEvaluator | None = None,
        polygraph_every_n_steps: int = 1,
        polygraph_min_patterns: int = 0,
        polygraph_confidence_skip: float = 0.85,
    ) -> None:
        self.preprocessor = preprocessor
        self.engine = engine
        self.postprocessor = postprocessor
        self.polygraph_generator = polygraph_generator
        self.polygraph_evaluator = polygraph_evaluator or PolygraphEvaluator()
        self.polygraph_every_n_steps = polygraph_every_n_steps
        self.polygraph_min_patterns = polygraph_min_patterns
        self.polygraph_confidence_skip = polygraph_confidence_skip
        self._step_count: int = 0
        self._cached_polygraph_scores = None
        self._cached_selected_view = None
        self.view_engines: dict[str, PatternEngine] = {}
        self.view_matches: dict[str, MatchResult] = {}
        self._view_pattern_names: dict[str, set[str]] = defaultdict(set)
        self.preprocessing_pipeline = AdapterRegistry()
        self.postprocessing_pipeline = AdapterRegistry()
        self.preprocessing_pipeline.register(preprocessor)
        self.postprocessing_pipeline.register(postprocessor)

    def register_preprocessor(self, adapter) -> None:
        self.preprocessing_pipeline.register(adapter)

    def register_postprocessor(self, adapter) -> None:
        self.postprocessing_pipeline.register(adapter)

    def step(self, raw: Any, *, goal: dict[str, float] | None = None, context: dict[str, Any] | None = None) -> PipelineResult:
        packet = AdapterPacket(raw=raw, goal=goal, context=dict(context or {}))
        packet = self.preprocessing_pipeline.run(
            packet,
            target_outputs=list(self.preprocessing_pipeline.adapters.keys())
        )
        if not packet.states:
            raise ValueError("Preprocessing pipeline produced no state")
        preprocessed_state = packet.states[-1]

        # Use goal from packet as it might have been modified by adapters (e.g., RewardToGoalAdapter)
        active_goal = packet.goal or {}

        preprocessed = PreprocessedInput(state=preprocessed_state, context=dict(preprocessed_state.context), raw=raw, packet=packet)
        plan_horizon = int(active_goal.get("plan_horizon", 1))

        # Observe all states in the packet (enables multi-state preprocessors)
        for s in packet.states[:-1]:
            self.engine.observe(s)
            
        self.engine.observe(preprocessed.state)
        action = self.engine.act(goal=active_goal, horizon=plan_horizon)

        if self.polygraph_generator is None:
            polygraph_scores = None
            polygraph_agreement = None
        else:
            # Determine whether to run the polygraph this step
            should_run_polygraph = (
                self._step_count % self.polygraph_every_n_steps == 0
                and len(self.engine.store.patterns) >= self.polygraph_min_patterns
                and action.confidence < self.polygraph_confidence_skip
            )

            if should_run_polygraph:
                views = self.polygraph_generator.generate(raw, context=preprocessed.context)
                polygraph_scores = {}
                self.view_matches = {}

                # Save primary engine state to avoid contamination from views
                primary_state = self.engine.current_state
                primary_history = list(self.engine.history)
                primary_trace = list(self.engine.pattern_trace)
                primary_last_match = self.engine.last_match

                # Observe and score all views cheaply
                for view in views:
                    # Reset engine state for this view
                    # Use a fresh start for each view to ensure it can learn from origin
                    self.engine.current_state = State(value=())
                    self.engine.history = []
                    self.engine.pattern_trace = []

                    # Tier 1: Fast path - observe through primary engine (shared store)
                    if hasattr(view, "states") and view.states:
                        # Establish origin for multi-state views
                        self.engine.current_state = None
                        for i, s in enumerate(view.states):
                            is_last = (i == len(view.states) - 1)
                            match = self.engine.observe(s, update_state=not is_last)
                    else:
                        match = self.engine.observe(view.state, update_state=False)
                    
                    if match is not None:
                        self.view_matches[view.name] = match
                    else:
                        # Fallback for single-state views or first-step observations
                        # where observe() returns None but we still want a match result
                        # against the pattern store.
                        direct_match = self.engine.store.match(view.state.value)
                        if direct_match:
                            self.view_matches[view.name] = direct_match
                            match = direct_match

                    if match and match.pattern:
                        self._view_pattern_names[view.name].add(match.pattern.name)

                    # Tier 2: Use existing engine or heuristic score
                    if view.name in self.view_engines:
                        engine = self.view_engines[view.name]
                        if hasattr(view, "states") and view.states:
                            for s in view.states:
                                engine.observe(s)
                        else:
                            engine.observe(view.state)
                        polygraph_scores[view.name] = self.polygraph_evaluator.score_engine(engine)
                    else:
                        # Heuristic score for selection when no engine exists
                        if match and match.pattern:
                            p = match.pattern
                            # fragmentation proxy: how many patterns learned for this view signature
                            fragmentation = float(len(self._view_pattern_names[view.name]))
                            
                            score_val = (self.polygraph_evaluator.alpha * 1.0 + 
                                         self.polygraph_evaluator.beta * p.density - 
                                         self.polygraph_evaluator.gamma * fragmentation)
                            polygraph_scores[view.name] = PolygraphScore(
                                concentration=1.0, average_density=p.density, fragmentation=fragmentation, score=score_val
                            )
                        else:
                            polygraph_scores[view.name] = PolygraphScore(0.0, 0.0, 0.0, 0.0)

                # Restore primary engine state
                self.engine.current_state = primary_state
                self.engine.history = primary_history
                self.engine.pattern_trace = primary_trace
                self.engine.last_match = primary_last_match

                selected_view = self.polygraph_evaluator.select_view(polygraph_scores)
                if selected_view is None and views:
                    selected_view = views[0].name

                # Tier 2: Create full engine only for selected view if needed for act()
                if selected_view is not None and selected_view not in self.view_engines:
                    selected_view_obj = next((v for v in views if v.name == selected_view), None)
                    if selected_view_obj:
                        engine = PatternEngine(config=self.engine.config)
                        if hasattr(selected_view_obj, "states") and selected_view_obj.states:
                            for s in selected_view_obj.states:
                                engine.observe(s)
                        else:
                            engine.observe(selected_view_obj.state)
                        self.view_engines[selected_view] = engine

                # Only call act() on the selected view engine
                if selected_view is not None and selected_view in self.view_engines:
                    selected_action = self.view_engines[selected_view].act(goal=active_goal, horizon=plan_horizon)
                else:
                    selected_action = action

                # Cache for future skipped steps
                self._cached_polygraph_scores = polygraph_scores
                self._cached_selected_view = selected_view

                long_horizon = plan_horizon > 1 or bool(active_goal.get("planning_mode") == "long")
                polygraph_agreement = None

                if long_horizon:
                    # Build a single-entry actions dict for agreement (only selected view acted)
                    view_actions = {selected_view: selected_action} if selected_view is not None else {}
                    polygraph_agreement = self.polygraph_evaluator.agreement(view_actions, polygraph_scores)
                    if polygraph_agreement.selected_key is not None:
                        action = selected_action
                    else:
                        action = selected_action
                    action = Action(
                        action_type=action.action_type,
                        value=action.value,
                        confidence=action.confidence,
                        selected_pattern=action.selected_pattern,
                        selected_sequence=action.selected_sequence,
                        selected_view=selected_view,
                        trace={
                            **action.trace,
                            "polygraph_scores": {name: score.score for name, score in polygraph_scores.items()},
                            "polygraph_agreement": None
                            if polygraph_agreement is None
                            else {
                                "selected_key": polygraph_agreement.selected_key,
                                "support": polygraph_agreement.support,
                                "dispersion": polygraph_agreement.dispersion,
                                "score": polygraph_agreement.score,
                            },
                            "selection_mode": "agreement",
                        },
                        forecast=action.forecast,
                    )
                else:
                    selected_polygraph_score = polygraph_scores.get(selected_view) if selected_view is not None else None
                    polygraph_bias = 0.05 * selected_polygraph_score.score if selected_polygraph_score is not None else 0.0
                    confidence = max(0.0, min(1.0, selected_action.confidence + polygraph_bias))
                    # Keep the primary engine forecast and matched pattern for domains
                    # like CartPole where the postprocessor expects the full primary
                    # state semantics. Polygraph views still contribute via reliability
                    # scores and confidence shaping, but their compact forecasts are not
                    # assumed to be executable control forecasts.
                    action = Action(
                        action_type=action.action_type,
                        value=action.value,
                        confidence=confidence,
                        selected_pattern=action.selected_pattern,
                        selected_sequence=action.selected_sequence,
                        selected_view=selected_view,
                        trace={
                            **action.trace,
                            "polygraph_scores": {name: score.score for name, score in polygraph_scores.items()},
                            "selected_polygraph_score": None if selected_polygraph_score is None else selected_polygraph_score.score,
                            "selection_mode": "single_view",
                        },
                        forecast=action.forecast,
                    )
            else:
                # Polygraph skipped — use cached scores, no additional act() calls
                polygraph_scores = self._cached_polygraph_scores
                polygraph_agreement = None
                # action already set from primary engine above

        self._step_count += 1

        output = None
        carry_context: dict = {}
        if action.action_type == "apply_delta":
            post_packet = AdapterPacket(raw=raw, goal=packet.goal, context=dict(preprocessed.context), core_action=action)
            post_packet = self.postprocessing_pipeline.run(post_packet, target_outputs=list(self.postprocessing_pipeline.adapters.keys()))
            output = post_packet.validated_output
            carry_context = {k[6:]: v for k, v in post_packet.context.items() if k.startswith("carry_")}

        return PipelineResult(
            input=preprocessed,
            action=action,
            output=output,
            polygraph_scores=polygraph_scores,
            polygraph_agreement=polygraph_agreement,
            carry_context=carry_context,
        )
