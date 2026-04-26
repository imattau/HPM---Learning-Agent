from hpm_ai_v4.simulations.control_transfer_simulation import run_control_transfer_simulation


def test_run_control_transfer_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 60)
    history = run_control_transfer_simulation(
        corpus_path=str(corpus),
        train_steps=120,
        validation_steps=80,
        log_every=80,
        chunk_size=40,
        prompt_size=40,
        warmup_chars=40,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )
    assert len(history) >= 2
    phases = {snap["phase"] for snap in history}
    assert "train" in phases
    assert "validation" in phases
    final_val = [snap for snap in history if snap["phase"] == "validation"][-1]
    assert "transfer_gain" in final_val
    assert "baseline_agreement" in final_val
    assert "decoder_choice" in final_val
    assert (tmp_path / "final_control_transfer_library.l1.pkl").exists()
    assert (tmp_path / "final_control_transfer_library.l2.pkl").exists()
    assert (tmp_path / "final_control_transfer_library.l3.pkl").exists()
    assert (tmp_path / "final_control_transfer_library.l4.pkl").exists()
    assert (tmp_path / "final_control_transfer_library.l5.pkl").exists()
