from hpm_ai_v4.simulations import bootstrap_libraries as bootstrap_module
from hpm_ai_v4.pattern import FlatPattern
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


def test_bootstrap_libraries_populates_registry(tmp_path, monkeypatch):
    calls = []

    def fake_runner(*args, **kwargs):
        idx = len(calls)
        calls.append((args, kwargs))
        registry_path = kwargs.get("registry_path")
        if registry_path:
            registry = LibraryRegistry(registry_path)
            names = [
                "text_seed",
                "structured_text_seed",
                "code_dsl_seed",
                "environment_seed",
                "tool_seed",
                "curriculum_seed",
            ]
            domains = [
                "text",
                "structured_text",
                "code",
                "environment",
                "tool",
                "curriculum",
            ]
            registry.upsert(
                name=names[idx],
                path=kwargs.get("output") or str(tmp_path / "bundle.pkl"),
                domain=domains[idx],
                status="seed",
                source="test",
                pattern_count=1,
            )
        return 0

    monkeypatch.setattr(bootstrap_module, "build_library", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_structured_text_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_code_dsl_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_hpm_environment_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_hpm_tool_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_hpm_curriculum_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "_resolve_preferred_text_seed_bundle", lambda: None)

    registry_path = tmp_path / "registry.json"
    root_dir = tmp_path / "bootstrap"
    completed = bootstrap_module.bootstrap_libraries(
        registry_path=str(registry_path),
        root_dir=str(root_dir),
        text_corpus=str(tmp_path / "text.txt"),
    )

    assert completed == [
        "text_seed",
        "structured_text_seed",
        "code_dsl_seed",
        "environment_seed",
        "tool_seed",
        "curriculum_seed",
    ]
    assert len(calls) == 6

    registry = LibraryRegistry(str(registry_path))
    assert registry.require("text_seed").domain == "text"
    assert registry.require("structured_text_seed").domain == "structured_text"
    assert registry.require("code_dsl_seed").domain == "code"
    assert registry.require("environment_seed").domain == "environment"
    assert registry.require("tool_seed").domain == "tool"
    assert registry.require("curriculum_seed").domain == "curriculum"


def test_bootstrap_libraries_clones_preferred_text_bundle(tmp_path, monkeypatch):
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        registry_path = kwargs.get("registry_path")
        if registry_path:
            registry = LibraryRegistry(registry_path)
            names = [
                "structured_text_seed",
                "code_dsl_seed",
                "environment_seed",
                "tool_seed",
                "curriculum_seed",
            ]
            domains = [
                "structured_text",
                "code",
                "environment",
                "tool",
                "curriculum",
            ]
            idx = len(calls) - 1
            registry.upsert(
                name=names[idx],
                path=kwargs.get("output") or str(tmp_path / "bundle.pkl"),
                domain=domains[idx],
                status="seed",
                source="test",
                pattern_count=1,
            )
        return 0

    source_bundle = tmp_path / "nltk_large_nlp_2000.pkl"
    PatternSerializer.save([FlatPattern.flat(0, obs_dim=5), FlatPattern.flat(1, obs_dim=5)], str(source_bundle))

    monkeypatch.setattr(bootstrap_module, "build_library", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_structured_text_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_code_dsl_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_hpm_environment_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_hpm_tool_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "run_hpm_curriculum_simulation", fake_runner)
    monkeypatch.setattr(bootstrap_module, "_resolve_preferred_text_seed_bundle", lambda: str(source_bundle))

    registry_path = tmp_path / "registry.json"
    root_dir = tmp_path / "bootstrap"
    completed = bootstrap_module.bootstrap_libraries(
        registry_path=str(registry_path),
        root_dir=str(root_dir),
        text_corpus=str(tmp_path / "text.txt"),
    )

    assert completed == [
        "text_seed",
        "structured_text_seed",
        "code_dsl_seed",
        "environment_seed",
        "tool_seed",
        "curriculum_seed",
    ]
    assert len(calls) == 5

    text_output = root_dir / "text_seed.pkl"
    assert text_output.exists()
    assert len(PatternSerializer.load(str(text_output))) == 2

    registry = LibraryRegistry(str(registry_path))
    assert registry.require("text_seed").source.startswith("bundle:")
    assert registry.require("text_seed").path == str(text_output)
