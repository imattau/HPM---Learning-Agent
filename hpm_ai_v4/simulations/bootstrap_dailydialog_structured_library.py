"""Populate the reasoner polygraph with structured DailyDialog episodes."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

from hpm_ai_v4.simulations import build_dailydialog_corpus as dd
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.episode_mappers import map_dailydialog_dialog
from hpm_ai_v4.tools.library_registry import LibraryRegistry


@dataclass
class DailyDialogStructuredBootstrapResult:
    bundle_base: str
    dialogs_written: int
    episodes_written: int
    memory_size: int


def bootstrap_dailydialog_structured_library(
    *,
    output_dir: str,
    registry_path: Optional[str] = None,
    dataset_name: str = "OpenRL/daily_dialog",
    split: str = "train",
    limit: Optional[int] = 1000,
    max_turns: int = 8,
) -> DailyDialogStructuredBootstrapResult:
    os.makedirs(output_dir, exist_ok=True)
    dataset = dd._load_dataset(dataset_name, split=split)
    agent = LayeredAgent(num_workers=1)

    dialogs_written = 0
    episodes_written = 0
    for row in dataset:
        if limit is not None and dialogs_written >= limit:
            break
        episodes = map_dailydialog_dialog(row, max_turns=max_turns)
        if not episodes:
            continue
        dialogs_written += 1
        for episode in episodes:
            agent.l1.reasoner.record_structured_episode(episode)
            episodes_written += 1

    bundle_base = os.path.join(output_dir, "daily_dialog_structured_bundle")
    agent.save_bundle(bundle_base)

    if registry_path:
        registry = LibraryRegistry(registry_path)
        registry.upsert(
            name="daily_dialog_structured_seed",
            path=bundle_base,
            domain="dialogue_episodes",
            status="seed",
            bundle_kind="stacked",
            level_contract="l1-l5",
            obs_dims=[5, 10, 10, 32, 64],
            decoder_families=["target", "explanation", "dictionary", "grammar"],
            source=f"{dataset_name}:{split}",
            pattern_count=episodes_written,
            notes="DailyDialog structured episodes stored in the reasoner polygraph",
        )

    return DailyDialogStructuredBootstrapResult(
        bundle_base=bundle_base,
        dialogs_written=dialogs_written,
        episodes_written=episodes_written,
        memory_size=agent.l1.reasoner.memory_size,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Populate a structured episode bundle from DailyDialog")
    p.add_argument("--output-dir", default="library_bootstrap/dailydialog_structured", help="Output directory")
    p.add_argument("--registry", default=None, help="Optional registry JSON path")
    p.add_argument("--dataset-name", default="OpenRL/daily_dialog", help="Hugging Face dataset name")
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument("--limit", type=int, default=1000, help="Max number of dialogs to process")
    p.add_argument("--max-turns", type=int, default=8, help="Max turns per dialog")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = bootstrap_dailydialog_structured_library(
        output_dir=args.output_dir,
        registry_path=args.registry,
        dataset_name=args.dataset_name,
        split=args.split,
        limit=args.limit,
        max_turns=args.max_turns,
    )
    print(f"[done] bundle: {result.bundle_base}")
    print(f"[done] dialogs: {result.dialogs_written}")
    print(f"[done] episodes: {result.episodes_written}")
    print(f"[done] memory: {result.memory_size}")
