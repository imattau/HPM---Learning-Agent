from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.text_repair_simulation import run_text_repair_simulation


def test_repair_text_returns_readable_text():
    agent = LayeredAgent(num_workers=1)
    for i in range(120):
        agent.perceive(i % 95)

    repaired = agent.repair_text(
        corrupted_text="the qu1ck br0wn f0x",
        target_text="the quick brown fox",
        use_constraints=False,
        update_policy=False,
    )

    assert isinstance(repaired, str)
    assert len(repaired) > 0
    assert 'L3:' not in repaired


def test_run_text_repair_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 30)

    history = run_text_repair_simulation(
        corpus_path=str(corpus),
        total_steps=120,
        log_every=60,
        chunk_size=32,
        warmup_chars=64,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) >= 2
    final = history[-1]
    assert "repair_agreement" in final
    assert "repair_improvement" in final
    assert final["repair_agreement"] >= final["corruption_agreement"]
    assert (tmp_path / "final_repair_library.l1.pkl").exists()
    assert (tmp_path / "final_repair_library.l2.pkl").exists()
    assert (tmp_path / "final_repair_library.l3.pkl").exists()
    assert (tmp_path / "final_repair_library.l4.pkl").exists()
    assert (tmp_path / "final_repair_library.l5.pkl").exists()
