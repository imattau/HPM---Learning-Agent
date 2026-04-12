from hfn.affective import AffectiveEvaluator, AffectiveState


def test_affective_update():
    ae = AffectiveEvaluator()
    # Start neutral
    a, v, s = ae._get_global_state()
    assert s == AffectiveState.NEUTRAL

    # Successful outcome with low surprise
    ae.update_from_outcome("test", success=True, surprise=0.1)
    a, v, s = ae._get_global_state()
    assert v > 0.5  # valence increased
    assert s in (AffectiveState.NEUTRAL, AffectiveState.CURIOUS)

    # Anxious state: high arousal, low valence
    ae._set_global_state(arousal=0.9, valence=0.2, state=AffectiveState.ANXIOUS)
    bonus = ae.get_affective_bonus("test")
    assert bonus > 0.7  # anxiety gives high bonus

    # Persistence under anxiety
    should_keep = ae.should_persist("test", epistemic_loss=0.7)
    # In should_persist: if state=ANXIOUS and arousal>0.7 → threshold=0.8
    # So epistemic_loss=0.7 < 0.8 → True (persist)
    assert should_keep is True

    # Curiosity exploration probability peaks at 0.5 learnability
    p_mid = ae.curiosity_exploration_probability(0.5)
    p_low = ae.curiosity_exploration_probability(0.1)
    assert p_mid > p_low
