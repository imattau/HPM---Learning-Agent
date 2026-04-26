from hpm_ai_v4.simulations import bootstrap_libraries as bootstrap_module
from hpm_ai_v4.tools.library_registry import LibraryRegistry


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
