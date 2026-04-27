from hpm_ai_v4.simulations.math_text_simulation import run_math_text_simulation


def test_run_math_text_simulation_short(tmp_path):
    history = run_math_text_simulation(
        total_steps=4,
        warmup_cases=2,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) == 4
    final = history[-1]
    assert "symbolic_score" in final
    assert "mixed_domain_score" in final
    assert "math_spans" in final
    assert "repair_agreement" in final
    assert (tmp_path / "math_text_library.l1.pkl").exists()
    assert (tmp_path / "math_text_library.l2.pkl").exists()
    assert (tmp_path / "math_text_library.l3.pkl").exists()
    assert (tmp_path / "math_text_library.l4.pkl").exists()
    assert (tmp_path / "math_text_library.l5.pkl").exists()
