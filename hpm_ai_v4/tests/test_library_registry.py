from hpm_ai_v4.tools.library_registry import BundleResolver, LibraryEntry, LibraryRegistry


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


def test_bundle_resolver_prefers_stacked_chat_bundle(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = LibraryRegistry(str(registry_path))

    stacked_base = tmp_path / "chat_ultra_bundle"
    (tmp_path / "chat_ultra_bundle.l1.pkl").write_text("stub")
    flat_path = tmp_path / "chat_super_library.pkl"
    flat_path.write_text("stub")
    text_path = tmp_path / "text_seed.pkl"
    text_path.write_text("stub")

    registry.upsert(
        name="chat_super_seed",
        path=str(flat_path),
        domain="chat",
        status="validated",
        bundle_kind="flat",
        level_contract="l1",
        obs_dims=[5],
        pattern_count=8,
    )
    registry.upsert(
        name="chat_ultra_bundle_seed",
        path=str(stacked_base),
        domain="chat",
        status="promoted",
        bundle_kind="stacked",
        level_contract="l1-l5",
        obs_dims=[5, 10, 10, 32, 64],
        pattern_count=64,
    )
    registry.upsert(
        name="text_seed",
        path=str(text_path),
        domain="text",
        status="promoted",
        bundle_kind="flat",
        level_contract="l1",
        obs_dims=[5],
        pattern_count=12,
    )

    resolution = registry.resolve_bundle(view="chat")
    assert resolution is not None
    assert resolution.entry.name == "chat_ultra_bundle_seed"
    assert resolution.path == str(stacked_base)
    assert "bundle_kind=stacked" in resolution.reasons


def test_bundle_resolver_falls_back_to_text_bundle_when_no_chat_stack(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = LibraryRegistry(str(registry_path))
    text_path = tmp_path / "text_seed.pkl"
    text_path.write_text("stub")

    registry.upsert(
        name="text_seed",
        path=str(text_path),
        domain="text",
        status="promoted",
        bundle_kind="flat",
        level_contract="l1",
        obs_dims=[5],
        pattern_count=12,
    )

    resolver = BundleResolver(registry)
    resolution = resolver.resolve(view="chat")
    assert resolution is not None
    assert resolution.entry.name == "text_seed"
    assert resolution.path == str(text_path)
