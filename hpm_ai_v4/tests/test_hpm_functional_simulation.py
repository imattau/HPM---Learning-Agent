from hpm_ai_v4.simulations.hpm_functional_simulation import run_hpm_functional_simulation


def test_run_hpm_functional_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 60)
    history = run_hpm_functional_simulation(
        corpus_path=str(corpus),
        train_steps=120,
        validation_steps=80,
        log_every=80,
        chunk_size=40,
        prompt_size=40,
        warmup_chars=40,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )
    assert len(history) >= 2
    phases = {snap["phase"] for snap in history}
    assert "train" in phases
    assert "validation" in phases
    final_val = [snap for snap in history if snap["phase"] == "validation"][-1]
    assert "transfer_gain" in final_val
    assert "decoder_choice" in final_val or final_val.get("decoder_choice") is not None
    assert (tmp_path / "hpm_functional_library.l1.pkl").exists()
    assert (tmp_path / "hpm_functional_library.l2.pkl").exists()
    assert (tmp_path / "hpm_functional_library.l3.pkl").exists()
    assert (tmp_path / "hpm_functional_library.l4.pkl").exists()
    assert (tmp_path / "hpm_functional_library.l5.pkl").exists()
