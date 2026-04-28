"""Bootstrap the first curated HPM library seeds across domains."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from hpm_ai_v4.simulations.build_library import build_library
from hpm_ai_v4.simulations.code_dsl_simulation import run_code_dsl_simulation
from hpm_ai_v4.simulations.hpm_curriculum_simulation import run_hpm_curriculum_simulation
from hpm_ai_v4.simulations.hpm_environment_simulation import run_hpm_environment_simulation
from hpm_ai_v4.simulations.hpm_tool_simulation import run_hpm_tool_simulation
from hpm_ai_v4.simulations.structured_text_simulation import run_structured_text_simulation
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass(frozen=True)
class SeedSpec:
    name: str
    domain: str
    runner: Callable[..., Any]
    output_name: str
    kwargs: Dict[str, Any]


DEFAULT_TEXT_CORPUS = os.path.join(os.path.dirname(__file__), "data", "wiki_sample.txt")
PREFERRED_TEXT_BUNDLE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "library_bootstrap",
    "nltk_large",
    "nltk_large_nlp_2000.pkl",
)


def _resolve_preferred_text_seed_bundle() -> Optional[str]:
    if os.path.exists(PREFERRED_TEXT_BUNDLE):
        return PREFERRED_TEXT_BUNDLE
    return None


def _clone_text_seed_bundle(
    source_bundle: str,
    output_path: str,
    registry_path: Optional[str],
    name: str,
    domain: str,
) -> int:
    patterns = PatternSerializer.load(source_bundle)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PatternSerializer.save(patterns, output_path)
    if registry_path:
        registry = LibraryRegistry(registry_path)
        densities = [float(getattr(p, "density_at_save", 0.0) or 0.0) for p in patterns]
        if not densities:
            densities = [0.0]
        registry.upsert(
            name=name,
            path=output_path,
            domain=domain,
            status="seed",
            bundle_kind="flat",
            level_contract="l1",
            obs_dims=[5],
            source=f"bundle:{os.path.relpath(source_bundle)}",
            density_mean=float(sum(densities) / len(densities)),
            density_min=float(min(densities)),
            density_max=float(max(densities)),
            pattern_count=len(patterns),
        )
    return 0


def default_seed_specs(
    root_dir: str,
    text_corpus: str = DEFAULT_TEXT_CORPUS,
) -> List[SeedSpec]:
    return [
        SeedSpec(
            name="text_seed",
            domain="text",
            runner=build_library,
            output_name="text_seed.pkl",
            kwargs={
                "corpus": text_corpus,
                "steps": 2_000,
                "min_density": 0.2,
            },
        ),
        SeedSpec(
            name="structured_text_seed",
            domain="structured_text",
            runner=run_structured_text_simulation,
            output_name="structured_text_library",
            kwargs={
                "total_steps": 12,
                "warmup_records": 2,
                "checkpoint_dir": root_dir,
            },
        ),
        SeedSpec(
            name="code_dsl_seed",
            domain="code",
            runner=run_code_dsl_simulation,
            output_name="code_dsl_library",
            kwargs={
                "total_steps": 12,
                "warmup_programs": 2,
                "checkpoint_dir": root_dir,
            },
        ),
        SeedSpec(
            name="environment_seed",
            domain="environment",
            runner=run_hpm_environment_simulation,
            output_name="hpm_environment_library",
            kwargs={
                "train_steps": 80,
                "validation_steps": 20,
                "warmup_steps": 10,
                "state_dim": 4,
                "train_family": 0,
                "validation_family": 1,
                "checkpoint_dir": root_dir,
            },
        ),
        SeedSpec(
            name="tool_seed",
            domain="tool",
            runner=run_hpm_tool_simulation,
            output_name="hpm_tool_library",
            kwargs={
                "train_episodes": 60,
                "validation_episodes": 20,
                "warmup_episodes": 8,
                "train_families": (0, 1, 2),
                "validation_families": (3,),
                "checkpoint_dir": root_dir,
            },
        ),
        SeedSpec(
            name="curriculum_seed",
            domain="curriculum",
            runner=run_hpm_curriculum_simulation,
            output_name="hpm_curriculum_library",
            kwargs={
                "train_episodes": 60,
                "validation_episodes": 20,
                "warmup_episodes": 8,
                "episode_length": 3,
                "train_families": (0, 1, 2),
                "validation_families": (3,),
                "checkpoint_dir": root_dir,
            },
        ),
    ]


def bootstrap_libraries(
    registry_path: str,
    root_dir: str,
    text_corpus: str = DEFAULT_TEXT_CORPUS,
    run_text: bool = True,
    run_structured: bool = True,
    run_code: bool = True,
    run_environment: bool = True,
    run_tool: bool = True,
    run_curriculum: bool = True,
) -> List[str]:
    """Populate the first seed libraries and register them."""
    os.makedirs(root_dir, exist_ok=True)
    completed: List[str] = []
    flags = {
        "text": run_text,
        "structured_text": run_structured,
        "code": run_code,
        "environment": run_environment,
        "tool": run_tool,
        "curriculum": run_curriculum,
    }
    preferred_text_bundle = _resolve_preferred_text_seed_bundle()
    for spec in default_seed_specs(root_dir=root_dir, text_corpus=text_corpus):
        if not flags.get(spec.domain, False):
            continue
        if spec.domain == "text":
            output_path = os.path.join(root_dir, spec.output_name)
            if preferred_text_bundle:
                _clone_text_seed_bundle(
                    source_bundle=preferred_text_bundle,
                    output_path=output_path,
                    registry_path=registry_path,
                    name=spec.name,
                    domain=spec.domain,
                )
                completed.append(spec.name)
                continue
            kwargs = dict(spec.kwargs)
            kwargs.update(
                {
                    "output": output_path,
                    "registry_path": registry_path,
                    "name": spec.name,
                    "domain": spec.domain,
                }
            )
        else:
            kwargs = dict(spec.kwargs)
            kwargs["registry_path"] = registry_path
        spec.runner(**kwargs)
        completed.append(spec.name)
    return completed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap first HPM library seeds")
    p.add_argument("--registry", required=True, help="Registry JSON path")
    p.add_argument("--root-dir", default="library_bootstrap", help="Output directory for seed bundles")
    p.add_argument("--text-corpus", default=DEFAULT_TEXT_CORPUS, help="Corpus for the text seed")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-structured", action="store_true")
    p.add_argument("--no-code", action="store_true")
    p.add_argument("--no-environment", action="store_true")
    p.add_argument("--no-tool", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    bootstrap_libraries(
        registry_path=args.registry,
        root_dir=args.root_dir,
        text_corpus=args.text_corpus,
        run_text=not args.no_text,
        run_structured=not args.no_structured,
        run_code=not args.no_code,
        run_environment=not args.no_environment,
        run_tool=not args.no_tool,
        run_curriculum=not args.no_curriculum,
    )
