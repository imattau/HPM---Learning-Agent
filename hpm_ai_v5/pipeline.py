"""Minimal end-to-end pipeline for v5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import AdapterPacket, AdapterRegistry
from .core import Action, PatternEngine
from .core.evaluator import PolygraphAgreement, PolygraphEvaluator, PolygraphScore
from .preprocessors.base import PreprocessedInput, Preprocessor
from .polygraphs.base import PolygraphGenerator
from .postprocessors.base import Postprocessor


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Structured result from preprocessing through output."""

    input: PreprocessedInput
    action: Action
    output: Any | None
    polygraph_scores: dict[str, PolygraphScore] | None = None
    polygraph_agreement: PolygraphAgreement | None = None


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
    ) -> None:
        self.preprocessor = preprocessor
        self.engine = engine
        self.postprocessor = postprocessor
        self.polygraph_generator = polygraph_generator
        self.polygraph_evaluator = polygraph_evaluator or PolygraphEvaluator()
        self.view_engines: dict[str, PatternEngine] = {}
        self.preprocessing_pipeline = AdapterRegistry()
        self.postprocessing_pipeline = AdapterRegistry()
        self.preprocessing_pipeline.register(preprocessor)
        self.postprocessing_pipeline.register(postprocessor)

    def register_preprocessor(self, adapter) -> None:
        self.preprocessing_pipeline.register(adapter)

    def register_postprocessor(self, adapter) -> None:
        self.postprocessing_pipeline.register(adapter)

    def step(self, raw: Any, *, goal: dict[str, float] | None = None, context: dict[str, Any] | None = None) -> PipelineResult:
        packet = AdapterPacket(raw=raw, draft_output=dict(context or {}))
        packet = self.preprocessing_pipeline.run(packet, target_outputs=[self.preprocessor.name])
        if not packet.states:
            raise ValueError("Preprocessing pipeline produced no state")
        preprocessed_state = packet.states[-1]
        preprocessed = PreprocessedInput(state=preprocessed_state, context=dict(preprocessed_state.context), raw=raw, packet=packet)
        if self.polygraph_generator is None:
            self.engine.observe(preprocessed.state)
            action = self.engine.act(goal=goal or {})
            polygraph_scores = None
        else:
            views = self.polygraph_generator.generate(raw, context=preprocessed.context)
            polygraph_scores = {}
            view_actions: dict[str, Action] = {}
            for view in views:
                engine = self.view_engines.setdefault(view.name, PatternEngine())
                engine.observe(view.state)
                polygraph_scores[view.name] = self.polygraph_evaluator.score_engine(engine)
                view_actions[view.name] = engine.act(goal=goal or {})

            selected_view = self.polygraph_evaluator.select_view(polygraph_scores)
            plan_horizon = float((goal or {}).get("plan_horizon", 1.0))
            long_horizon = plan_horizon > 1.0 or bool((goal or {}).get("planning_mode") == "long")
            polygraph_agreement = None

            if long_horizon:
                polygraph_agreement = self.polygraph_evaluator.agreement(view_actions, polygraph_scores)
                if polygraph_agreement.selected_key is not None:
                    for view_name, candidate_action in view_actions.items():
                        if self.polygraph_evaluator._action_key(candidate_action) == polygraph_agreement.selected_key:
                            selected_view = view_name
                            action = candidate_action
                            break
                    else:
                        selected_view = selected_view or (views[0].name if views else None)
                        action = view_actions[selected_view] if selected_view is not None and selected_view in view_actions else self.engine.act(goal=goal or {})
                else:
                    selected_view = selected_view or (views[0].name if views else None)
                    action = view_actions[selected_view] if selected_view is not None and selected_view in view_actions else self.engine.act(goal=goal or {})
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
                if selected_view is None:
                    selected_view = views[0].name if views else None
                action = view_actions[selected_view] if selected_view is not None and selected_view in view_actions else self.engine.act(goal=goal or {})
                selected_polygraph_score = polygraph_scores.get(selected_view) if selected_view is not None else None
                polygraph_bias = 0.05 * selected_polygraph_score.score if selected_polygraph_score is not None else 0.0
                confidence = max(0.0, min(1.0, action.confidence + polygraph_bias))
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

        output = None
        if action.action_type == "apply_delta":
            post_packet = AdapterPacket(raw=raw, draft_output=dict(preprocessed.context), core_action=action)
            post_packet = self.postprocessing_pipeline.run(post_packet, target_outputs=[self.postprocessor.name])
            output = post_packet.validated_output

        if self.polygraph_generator is None:
            polygraph_agreement = None
        return PipelineResult(
            input=preprocessed,
            action=action,
            output=output,
            polygraph_scores=polygraph_scores,
            polygraph_agreement=polygraph_agreement,
        )
