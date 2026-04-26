import os

from hpm_ai_v4.simulations.text_full_simulation import run_text_simulation


def test_run_text_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 30)
    history = run_text_simulation(
        corpus_path=str(corpus),
        total_steps=160,
        log_every=80,
        chunk_size=40,
        warmup_chars=80,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )
    assert len(history) >= 2
    final = history[-1]
    assert "target_agreement" in final
    assert "plausibility" in final
    assert "l2_accuracy" in final
    assert "l3_accuracy" in final
    assert 0.0 <= final["target_agreement"] <= 1.0
    assert (tmp_path / "final_text_library.l1.pkl").exists()
    assert (tmp_path / "final_text_library.l2.pkl").exists()
    assert (tmp_path / "final_text_library.l3.pkl").exists()
    assert (tmp_path / "final_text_library.l4.pkl").exists()
    assert (tmp_path / "final_text_library.l5.pkl").exists()
