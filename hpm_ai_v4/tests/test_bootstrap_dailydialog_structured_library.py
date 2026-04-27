from hpm_ai_v4.simulations import build_dailydialog_corpus as dd
from hpm_ai_v4.simulations.bootstrap_dailydialog_structured_library import (
    bootstrap_dailydialog_structured_library,
)
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.library_registry import LibraryRegistry


def test_bootstrap_dailydialog_structured_library_populates_reasoner(tmp_path, monkeypatch):
    fake_dataset = [
        {"dialog": ["Hello there.", "Hi.", "How can I help you?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["What is new?", "Not much.", "Want to talk about code?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
    ]
    monkeypatch.setattr(dd, "_load_dataset", lambda name, split: fake_dataset)

    output_dir = tmp_path / "structured"
    registry_path = tmp_path / "registry.json"
    result = bootstrap_dailydialog_structured_library(
        output_dir=str(output_dir),
        registry_path=str(registry_path),
        limit=10,
        max_turns=4,
    )

    bundle_base = str(output_dir / "daily_dialog_structured_bundle")
    assert result.dialogs_written == 2
    assert result.episodes_written >= 4
    assert result.memory_size == result.episodes_written
    assert (output_dir / "daily_dialog_structured_bundle.l1.pkl").exists()
    assert (output_dir / "daily_dialog_structured_bundle.reasoner.l1.json").exists()

    registry = LibraryRegistry(str(registry_path))
    entry = registry.require("daily_dialog_structured_seed")
    assert entry.path == bundle_base
    assert entry.domain == "dialogue_episodes"
    assert entry.pattern_count == result.episodes_written

    loaded = LayeredAgent(num_workers=1)
    loaded.load_bundle(bundle_base)
    assert loaded.l1.reasoner.memory_size == result.episodes_written
