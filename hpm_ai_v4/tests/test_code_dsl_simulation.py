from hpm_ai_v4.io.adapters import CodeDSLAdapter
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.code_dsl_simulation import run_code_dsl_simulation


def test_code_dsl_adapter_round_trip_and_execute():
    adapter = CodeDSLAdapter()
    program = "push 2\npush 3\nadd\npush 4\nmul\nreturn"
    canonical = adapter.to_text(program)
    assert canonical == "PUSH 2\nPUSH 3\nADD\nPUSH 4\nMUL\nRETURN"
    assert adapter.execute(canonical) == 20
    assert adapter.from_text(canonical)[0] == ("PUSH", 2)


def test_observe_code_dsl_feeds_execution_feedback():
    agent = LayeredAgent(num_workers=1)
    program = "push 2\npush 3\nadd\nreturn"
    stats = agent.observe_code_dsl(program, generated_program=program, feedback_mode="hybrid")
    assert stats["execution_match"] is True
    assert stats["parseable"] is True
    assert "kind" in stats
    assert "structure_score" in stats
    assert agent.l1.reasoner.memory_size > 0


def test_run_code_dsl_simulation_short(tmp_path):
    history = run_code_dsl_simulation(
        total_steps=8,
        warmup_programs=2,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) == 8
    final = history[-1]
    assert "repair_agreement" in final
    assert "repair_improvement" in final
    assert "execution_match" in final
    assert "text_signal_score" in final
    assert (tmp_path / "code_dsl_library.l1.pkl").exists()
    assert (tmp_path / "code_dsl_library.l2.pkl").exists()
    assert (tmp_path / "code_dsl_library.l3.pkl").exists()
    assert (tmp_path / "code_dsl_library.l4.pkl").exists()
    assert (tmp_path / "code_dsl_library.l5.pkl").exists()
