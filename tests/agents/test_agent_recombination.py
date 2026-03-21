import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.memory import InMemoryStore


RECOMB_KEYS = {
    'total_conflict',
    'recombination_attempted',
    'recombination_accepted',
    'recombination_trigger',
    'insight_score',
    'recomb_parent_ids',
}


def base_cfg(**kwargs):
    cfg = AgentConfig(
        agent_id='t',
        feature_dim=2,
        conflict_threshold=float('inf'),   # suppress conflict trigger by default
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def add_level4_patterns(store, agent_id):
    """Add two Level 4 patterns to the store."""
    p1 = GaussianPattern(mu=np.array([0.0, 0.0]), sigma=np.eye(2))
    p1.level = 4
    p2 = GaussianPattern(mu=np.array([0.5, 0.5]), sigma=np.eye(2))
    p2.level = 4
    store.save(p1, 0.5, agent_id)
    store.save(p2, 0.5, agent_id)
    return p1, p2


# --- Keys present every step ---

def test_return_dict_has_recombination_keys_every_step():
    agent = Agent(base_cfg())
    result = agent.step(np.zeros(2))
    for key in RECOMB_KEYS:
        assert key in result, f"missing key: {key}"


def test_total_conflict_in_return_dict_is_non_negative():
    agent = Agent(base_cfg())
    result = agent.step(np.zeros(2))
    assert result['total_conflict'] >= 0.0


def test_recomb_parent_ids_none_when_not_attempted():
    agent = Agent(base_cfg())
    result = agent.step(np.zeros(2))
    assert result['recomb_parent_ids'] is None


# --- Time trigger ---

def test_recombination_not_attempted_before_T_recomb():
    """conflict_threshold=inf suppresses conflict trigger; step T_recomb-1 times."""
    cfg = base_cfg(T_recomb=5, conflict_threshold=float('inf'))
    agent = Agent(cfg)
    for _ in range(4):
        result = agent.step(np.zeros(2))
        assert not result['recombination_attempted'], \
            f"triggered early at step {result['t']}"


def test_time_trigger_fires_at_T_recomb():
    """recombination_attempted=True at step T_recomb (even if no Level 4 patterns)."""
    cfg = base_cfg(T_recomb=5, conflict_threshold=float('inf'))
    agent = Agent(cfg)
    for i in range(5):
        result = agent.step(np.zeros(2))
    assert result['recombination_attempted'] is True


# --- Conflict trigger ---

def test_conflict_trigger_fires_on_high_tension():
    """conflict_threshold=0.0 fires on any nonzero conflict; T_recomb=1000 prevents time trigger."""
    cfg = base_cfg(T_recomb=1000, conflict_threshold=0.0, recomb_cooldown=0)
    store = InMemoryStore()
    agent = Agent(cfg, store=store)
    add_level4_patterns(store, cfg.agent_id)

    result = agent.step(np.zeros(2))
    assert result['recombination_attempted'] is True
    assert result['recombination_trigger'] == 'conflict'


# --- Cooldown ---

def test_cooldown_blocks_double_trigger():
    """Conflict fires at step 1; step 2 blocked by cooldown=5."""
    cfg = base_cfg(T_recomb=1000, conflict_threshold=0.0, recomb_cooldown=5)
    store = InMemoryStore()
    agent = Agent(cfg, store=store)
    add_level4_patterns(store, cfg.agent_id)

    r1 = agent.step(np.zeros(2))
    r2 = agent.step(np.zeros(2))
    assert r1['recombination_attempted'] is True
    assert r2['recombination_attempted'] is False


# --- Acceptance side effects ---

def test_accepted_pattern_added_to_store():
    store = InMemoryStore()
    cfg = base_cfg(
        T_recomb=1,
        conflict_threshold=float('inf'),
        recomb_cooldown=0,
        kappa_max=1.0,
        alpha_nov=1.0,
        alpha_eff=0.0,
        beta_orig=1.0,
        kappa_0=0.1,
    )
    agent = Agent(cfg, store=store)

    # Replace seeded pattern with two Level 4 patterns
    for p, _ in store.query(cfg.agent_id):
        store.delete(p.id)
    add_level4_patterns(store, cfg.agent_id)

    before = len(store.query(cfg.agent_id))
    result = agent.step(np.zeros(2))

    if result['recombination_accepted']:
        after = len(store.query(cfg.agent_id))
        assert after == before + 1


def test_weights_sum_to_one_after_acceptance():
    store = InMemoryStore()
    cfg = base_cfg(
        T_recomb=1,
        conflict_threshold=float('inf'),
        recomb_cooldown=0,
        kappa_max=1.0,
        alpha_nov=1.0,
        alpha_eff=0.0,
        beta_orig=1.0,
        kappa_0=0.1,
    )
    agent = Agent(cfg, store=store)

    for p, _ in store.query(cfg.agent_id):
        store.delete(p.id)
    add_level4_patterns(store, cfg.agent_id)

    result = agent.step(np.zeros(2))

    if result['recombination_accepted']:
        records = store.query(cfg.agent_id)
        total_w = sum(w for _, w in records)
        assert abs(total_w - 1.0) < 1e-9
