from hpm_ai_v4.tools.library_registry import LibraryEntry, LibraryRegistry


def test_library_registry_register_promote_and_list(tmp_path):
    path = tmp_path / "registry.json"
    registry = LibraryRegistry(str(path))

    entry = registry.upsert(
        name="text_seed",
        path="/tmp/text_seed.pkl",
        domain="text",
        status="seed",
        source="wiki:science",
        density_mean=0.42,
        density_min=0.30,
        density_max=0.55,
        pattern_count=12,
    )
    assert entry.name == "text_seed"
    assert path.exists()

    registry.promote("text_seed", notes="stable under held-out validation")
    promoted = registry.require("text_seed")
    assert promoted.status == "promoted"
    assert "held-out" in promoted.notes

    listed = registry.list(domain="text", status="promoted")
    assert len(listed) == 1
    assert listed[0].name == "text_seed"


def test_library_registry_round_trip(tmp_path):
    path = tmp_path / "registry.json"
    registry = LibraryRegistry(str(path))
    registry.register(
        LibraryEntry(
            name="code_dsl_seed",
            path="/tmp/code_dsl.pkl",
            domain="code",
            status="validated",
            source="synthetic",
            density_mean=0.6,
            pattern_count=8,
        )
    )

    reloaded = LibraryRegistry(str(path))
    entry = reloaded.require("code_dsl_seed")
    assert entry.domain == "code"
    assert entry.status == "validated"
    assert entry.pattern_count == 8
