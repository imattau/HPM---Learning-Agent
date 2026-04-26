from hpm_ai_v4.io.adapters import StructuredTextAdapter
from hpm_ai_v4.simulations.structured_text_simulation import run_structured_text_simulation


def test_structured_text_adapter_canonical_json():
    adapter = StructuredTextAdapter()
    text = adapter.to_text({"phase": "validation", "kind": "bundle", "count": 2})
    assert text == '{"count":2,"kind":"bundle","phase":"validation"}'
    assert isinstance(adapter.from_text(text), dict)


def test_run_structured_text_simulation_short(tmp_path):
    history = run_structured_text_simulation(
        total_steps=8,
        chunk_size=1,
        warmup_records=2,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) == 8
    final = history[-1]
    assert "repair_agreement" in final
    assert "repair_improvement" in final
    assert "parseable" in final
    assert (tmp_path / "structured_text_library.l1.pkl").exists()
    assert (tmp_path / "structured_text_library.l2.pkl").exists()
    assert (tmp_path / "structured_text_library.l3.pkl").exists()
    assert (tmp_path / "structured_text_library.l4.pkl").exists()
    assert (tmp_path / "structured_text_library.l5.pkl").exists()
