import pytest
from unittest.mock import MagicMock
from hpm_ai_v3.curriculum import CurriculumManager


def make_curriculum_with_two_phases():
    """Helper: returns a CurriculumManager with at least 2 phases loaded."""
    cm = CurriculumManager()
    if len(cm.patterns) < 2:
        # Inject mock patterns if filesystem has fewer than 2
        from hpm_ai_v3.curriculum import CurriculumPattern
        cm.patterns = [
            CurriculumPattern("phase0", 0, [{"text": "1+1", "answer": 2.0}]),
            CurriculumPattern("phase1", 1, [{"text": "2+2", "answer": 4.0}]),
        ]
        cm.active_pattern_idx = 0
        cm.phase = 0
    return cm


class TestCurriculumManagerAdditions:
    def test_advance_phase_increments_index(self):
        cm = make_curriculum_with_two_phases()
        initial_idx = cm.active_pattern_idx
        cm.advance_phase()
        assert cm.active_pattern_idx == initial_idx + 1

    def test_advance_phase_updates_phase_attribute(self):
        cm = make_curriculum_with_two_phases()
        cm.advance_phase()
        assert cm.phase == cm.patterns[cm.active_pattern_idx].phase

    def test_advance_phase_resets_recent_rewards(self):
        cm = make_curriculum_with_two_phases()
        cm.recent_rewards = [1.0, 1.0, 1.0]
        cm.advance_phase()
        assert cm.recent_rewards == []

    def test_advance_phase_does_not_exceed_last_phase(self):
        cm = make_curriculum_with_two_phases()
        cm.active_pattern_idx = len(cm.patterns) - 1
        cm.advance_phase()
        assert cm.active_pattern_idx == len(cm.patterns) - 1

    def test_set_difficulty_increases(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.5
        cm.set_difficulty(0.1)
        assert abs(cm.difficulty - 0.6) < 1e-6

    def test_set_difficulty_decreases(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.5
        cm.set_difficulty(-0.2)
        assert abs(cm.difficulty - 0.3) < 1e-6

    def test_set_difficulty_clamps_at_zero(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.1
        cm.set_difficulty(-0.5)
        assert cm.difficulty == 0.0

    def test_set_difficulty_clamps_at_one(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.9
        cm.set_difficulty(0.5)
        assert cm.difficulty == 1.0

import numpy as np
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.causal_pattern import CausalPattern


def make_population():
    patterns = [CausalPattern(4, 2, 2) for _ in range(4)]
    weights = [0.4, 0.3, 0.2, 0.1]
    pop = PatternPopulation(patterns)
    for p, w in zip(pop.patterns, weights):
        p.weight = w
    return pop


class TestPatternPopulationAdditions:
    def test_get_population_entropy_is_float(self):
        pop = make_population()
        entropy = pop.get_population_entropy()
        assert isinstance(entropy, float)

    def test_get_population_entropy_uniform_is_max(self):
        pop = make_population()
        for p in pop.patterns:
            p.weight = 0.25
        uniform_entropy = pop.get_population_entropy()
        for p in pop.patterns:
            p.weight = 1.0  # one dominant
        pop.patterns[1].weight = 0.0
        pop.patterns[2].weight = 0.0
        pop.patterns[3].weight = 0.0
        skewed_entropy = pop.get_population_entropy()
        assert uniform_entropy > skewed_entropy

    def test_get_diversity_is_float(self):
        pop = make_population()
        diversity = pop.get_diversity()
        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0

    def test_get_diversity_identical_patterns_is_zero(self):
        p = CausalPattern(4, 2, 2)
        pop = PatternPopulation([p, p])
        assert pop.get_diversity() == 0.0

from hpm_ai_v3.agents.base_discovery import ActionPattern


class TestActionPatternAdditions:
    def test_action_pattern_has_exploration_temperature(self):
        ap = ActionPattern("arithmetic")
        assert hasattr(ap, "exploration_temperature")
        assert ap.exploration_temperature == 1.0

    def test_action_pattern_has_recent_use_count(self):
        ap = ActionPattern("arithmetic")
        assert hasattr(ap, "recent_use_count")
        assert ap.recent_use_count == 0

    def test_mark_used_increments_recent_use_count(self):
        ap = ActionPattern("arithmetic")
        ap.mark_used()
        assert ap.recent_use_count == 1

import torch
from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern, MetaDirective


class TestMetaCognitivePattern:
    def setup_method(self):
        self.mcp = MetaCognitivePattern()

    def test_inherits_hpm_pattern(self):
        from hpm_ai_v3.pattern import HPMPattern
        assert isinstance(self.mcp, HPMPattern)

    def test_policy_network_output_shape(self):
        features = torch.zeros(64)
        probs = self.mcp._policy_forward(features)
        assert probs.shape == (8,)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_act_returns_valid_directive(self):
        features = torch.zeros(64)
        directive = self.mcp.act(features)
        assert directive in list(MetaDirective)

    def test_meta_feature_vector_has_correct_dim(self):
        # Build mock agent with minimal required attributes
        agent = _make_mock_agent()
        curriculum = _make_mock_curriculum()
        features = self.mcp.observe(agent, curriculum)
        assert features.shape == (64,)

    def test_record_transition_stores_entry(self):
        features = torch.zeros(64)
        self.mcp.record_transition(features, MetaDirective.CONTINUE, 1.0)
        assert len(self.mcp._trajectory) == 1

    def test_update_policy_runs_without_error(self):
        features = torch.zeros(64)
        for _ in range(3):
            self.mcp.record_transition(features, MetaDirective.CONTINUE, 0.5)
        self.mcp.update_policy()  # must not raise
        assert len(self.mcp._trajectory) == 0  # cleared after update

    def test_log_prob_returns_scalar_tensor(self):
        obs = {"meta_features": torch.zeros(64)}
        lp = self.mcp.log_prob(obs)
        assert lp.ndim == 0

    def test_structural_distance_same_class(self):
        other = MetaCognitivePattern()
        assert self.mcp.structural_distance(other) == 0.0

    def test_structural_distance_different_class(self):
        from hpm_ai_v3.causal_pattern import CausalPattern
        other = CausalPattern(4, 2, 2)
        assert self.mcp.structural_distance(other) == 1.0

    def test_extract_causal_graph_has_one_node(self):
        g = self.mcp.extract_causal_graph()
        assert len(g.nodes) == 1


def _make_mock_agent():
    agent = MagicMock()
    agent.population.patterns = []
    agent.population.get_population_entropy.return_value = 1.0
    agent.population.get_diversity.return_value = 0.5
    agent.population.get_top_patterns.return_value = []
    agent.run_episode.return_value = (0.5, 1)
    agent.evaluate_solution.return_value = 0.5
    agent._meta_success_history = [0.5] * 10
    agent._steps_since_advance = 5
    return agent


def _make_mock_curriculum():
    cm = MagicMock()
    cm.phase = 0
    cm.difficulty = 0.5
    return cm

from unittest.mock import MagicMock, patch
from hpm_ai_v3.agents.meta_training import MetaTrainingLoop
from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern, MetaDirective


class TestMetaTrainingLoop:
    def test_run_meta_step_calls_observe_and_act(self):
        agent = _make_mock_agent()
        curriculum = _make_mock_curriculum()
        meta = MetaCognitivePattern()
        loop = MetaTrainingLoop(agent, curriculum, meta, N=3)

        with patch.object(meta, "observe_and_act", return_value=MetaDirective.CONTINUE) as mock_act:
            with patch.object(meta, "record_transition") as mock_record:
                with patch.object(meta, "update_policy") as mock_update:
                    loop.run_meta_step()
        mock_act.assert_called_once()

    def test_run_meta_step_calls_update_policy(self):
        agent = _make_mock_agent()
        curriculum = _make_mock_curriculum()
        meta = MetaCognitivePattern()
        loop = MetaTrainingLoop(agent, curriculum, meta, N=3)

        with patch.object(meta, "observe_and_act", return_value=MetaDirective.CONTINUE):
            with patch.object(meta, "record_transition"):
                with patch.object(meta, "update_policy") as mock_update:
                    loop.run_meta_step()
        mock_update.assert_called_once()

    def test_reset_use_counts_resets_recent_use_count(self):
        from hpm_ai_v3.agents.base_discovery import ActionPattern
        agent = _make_mock_agent()
        ap = ActionPattern("arithmetic")
        ap.recent_use_count = 5
        agent.population.patterns = [ap]
        curriculum = _make_mock_curriculum()
        meta = MetaCognitivePattern()
        loop = MetaTrainingLoop(agent, curriculum, meta, N=3)
        loop._reset_use_counts()
        assert ap.recent_use_count == 0
