"""One-time migration: convert per-agent JSONL/JSON archives to SQLite + sqlite-vec."""
from __future__ import annotations

import json
import os
import sys


def migrate_agent_dir(agent_dir: str) -> int:
    """Migrate one agent directory. Returns count of patterns migrated."""
    from hpm_ai_v6.hpm_model.core.cell import Cell
    from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager
    import numpy as np

    agent_name = os.path.basename(agent_dir)
    cache_dir = os.path.dirname(agent_dir)

    pager = PatternPager(cache_dir=cache_dir, agent_name=agent_name)

    payloads: dict[str, dict] = {}

    # Read index.jsonl
    index_path = os.path.join(agent_dir, "index.jsonl")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    payloads[str(p["name"])] = p
                except (json.JSONDecodeError, KeyError):
                    continue

    # Read individual JSON files (may fill gaps not in index)
    for fname in os.listdir(agent_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(agent_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                p = json.load(f)
            name = str(p.get("name", ""))
            if name and name not in payloads:
                payloads[name] = p
        except Exception:
            continue

    count = 0
    for payload in payloads.values():
        try:
            emb = payload.get("embedding")
            if emb is None:
                continue
            cell = Cell(
                name=str(payload["name"]),
                dim=int(payload.get("dim", 1)),
                embedding=np.asarray(emb, dtype=float),
                weight=float(payload.get("weight", 1.0)),
            )
            pager.save(cell)
            count += 1
        except Exception as e:
            print(f"  Skipping {payload.get('name')}: {e}")

    pager.flush()
    pager.close()
    return count


def main(cache_dir: str) -> None:
    if not os.path.isdir(cache_dir):
        print(f"Error: {cache_dir} is not a directory")
        sys.exit(1)

    for agent_name in os.listdir(cache_dir):
        agent_dir = os.path.join(cache_dir, agent_name)
        if not os.path.isdir(agent_dir):
            continue
        if agent_name.startswith("."):
            continue
        # Skip if already migrated (only patterns.db, no JSONL files)
        has_jsonl = os.path.exists(os.path.join(agent_dir, "index.jsonl"))
        has_json = any(f.endswith(".json") for f in os.listdir(agent_dir))
        if not has_jsonl and not has_json:
            print(f"  {agent_name}: already migrated, skipping")
            continue

        print(f"Migrating {agent_name}...")
        count = migrate_agent_dir(agent_dir)
        print(f"  {agent_name}: {count} patterns migrated")

    print("\nDone. Old JSON/JSONL files are preserved — delete them manually when satisfied.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 migrate_pager_to_sqlite.py <cache_dir>")
        sys.exit(1)
    main(sys.argv[1])
