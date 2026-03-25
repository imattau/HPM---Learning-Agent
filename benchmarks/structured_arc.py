def _make_structured_orch():
    """Build L1(2-agent) + L2(2-agent) + L3(1-agent) StructuredOrchestrator."""
    (l1_orch, l1_agents, _) = make_orchestrator(n_agents=2, feature_dim=64, agent_ids=['l1_0', 'l1_1'], pattern_types=['gaussian', 'gaussian'], with_monitor=False, gamma_soc=0.5, init_sigma=2.0)
    (l2_orch, l2_agents, _) = make_orchestrator(n_agents=2, feature_dim=9, agent_ids=['l2_0', 'l2_1'], pattern_types=['gaussian', 'gaussian'], with_monitor=False, gamma_soc=0.5, init_sigma=2.0)
    (l3_orch, l3_agents, _) = make_orchestrator(n_agents=1, feature_dim=14, agent_ids=['l3_0'], pattern_types=['gaussian'], with_monitor=False, gamma_soc=0.5, init_sigma=2.0)
    (enc1, enc2, enc3) = (ArcL1Encoder(), ArcL2Encoder(), ArcL3Encoder())
    orch = StructuredOrchestrator(encoders=[enc1, enc2, enc3], orches=[l1_orch, l2_orch, l3_orch], agents=[l1_agents, l2_agents, l3_agents], level_Ks=[1, 1, 3])
    return (orch, l1_agents, l2_agents, l3_agents)