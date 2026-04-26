import pytest

from hpm_ai_v4.simulations.build_library import main as build_library_main


def test_build_library_writes_registry_entry(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the cat sat on the mat. " * 20)
    output = tmp_path / "patterns.pkl"
    registry = tmp_path / "registry.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_library.py",
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--steps",
            "20",
            "--min-density",
            "0.0",
            "--registry",
            str(registry),
            "--name",
            "cat_mat_seed",
            "--domain",
            "text",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        build_library_main()

    assert exc.value.code == 0
    assert output.exists()
    assert registry.exists()
