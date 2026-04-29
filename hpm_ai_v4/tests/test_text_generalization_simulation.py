from hpm_ai_v4.pattern import FlatPattern
from hpm_ai_v4.simulations.text_generalization_simulation import run_text_generalization_simulation
from hpm_ai_v4.simulations.text_generalization_simulation import _load_library
from hpm_ai_v4.simulations.text_generalization_simulation import run_topdown_suppression_ab_benchmark
from hpm_ai_v4.tools.serializer import PatternSerializer


def test_run_text_generalization_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 40)
    history = run_text_generalization_simulation(
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
    assert "target_agreement" in final_val
    assert "plausibility" in final_val
    assert (tmp_path / "final_generalization_library.l1.pkl").exists()
    assert (tmp_path / "final_generalization_library.l2.pkl").exists()
    assert (tmp_path / "final_generalization_library.l3.pkl").exists()
    assert (tmp_path / "final_generalization_library.l4.pkl").exists()
    assert (tmp_path / "final_generalization_library.l5.pkl").exists()


def test_run_text_generalization_simulation_ascii_surface(tmp_path):
    corpus = tmp_path / "corpus_ascii.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 20)
    history = run_text_generalization_simulation(
        corpus_path=str(corpus),
        train_steps=60,
        validation_steps=40,
        log_every=40,
        chunk_size=20,
        prompt_size=20,
        warmup_chars=20,
        num_workers=1,
        use_dict=False,
        surface_mode="ascii",
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) >= 2
    assert history[-1]["phase"] == "validation"


def test_generalization_simulation_loads_flat_library(tmp_path):
    library = tmp_path / "generalization_seed.pkl"
    PatternSerializer.save([FlatPattern.flat(0, obs_dim=5)], str(library))

    from hpm_ai_v4.simulations.layered_agent import LayeredAgent

    agent = LayeredAgent(num_workers=1)
    loaded = _load_library(agent, str(library))

    assert loaded == 1
    assert len(agent.l1.patterns) == 1


def test_topdown_suppression_ab_benchmark(tmp_path):
    corpus = tmp_path / "corpus_ab.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 60)
    report = run_topdown_suppression_ab_benchmark(
        corpus_path=str(corpus),
        train_steps=80,
        validation_steps=40,
        log_every=40,
        chunk_size=20,
        prompt_size=20,
        warmup_chars=20,
        num_workers=1,
        use_dict=False,
        surface_mode="word",
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )

    assert "suppression_on" in report
    assert "suppression_off" in report
    assert "delta" in report
    assert (tmp_path / "topdown_suppression_benchmark.json").exists()
