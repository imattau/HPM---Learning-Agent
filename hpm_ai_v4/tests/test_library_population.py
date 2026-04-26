from hpm_ai_v4.simulations.hpm_environment_simulation import run_hpm_environment_simulation
from hpm_ai_v4.tools.library_registry import LibraryRegistry


def test_environment_simulation_auto_registers_library(tmp_path):
    registry_path = tmp_path / "libraries.json"
    history = run_hpm_environment_simulation(
        train_steps=40,
        validation_steps=10,
        warmup_steps=8,
        log_every=40,
        state_dim=4,
        train_family=0,
        validation_family=1,
        num_workers=1,
        checkpoint_dir=str(tmp_path),
        registry_path=str(registry_path),
    )

    assert history
    assert registry_path.exists()
    registry = LibraryRegistry(str(registry_path))
    entry = registry.require("hpm_environment_library")
    assert entry.domain == "environment"
    assert entry.path.endswith(".pkl")
    assert entry.pattern_count > 0
