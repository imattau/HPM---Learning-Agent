"""Measure process RSS and retained HFN indexes during source-only pretraining.

Example::

    python -m hpm_ai_v2.experiments.profile_pretrain_memory \
        --corpus data/pretraining/oasst1_train_en_1000.jsonl \
        --max-rows 300 --output-dir /tmp/hpm-oasst-300 \
        --report /tmp/hpm-oasst-300-memory.json

The profiler keeps process memory sampling separate from the training report,
and reports the bytes held by TieredForest's all-node vector index.  It is a
diagnostic tool; it does not impose a memory limit or change training behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import psutil

import hpm_ai_v2.training.pretrain_hfn_corpus as pretraining


def _bounded_input(path: Path, max_rows: int | None) -> tuple[Path, str, int]:
    """Return the source path or a temporary prefix sample and its SHA256."""
    digest = hashlib.sha256()
    if max_rows is None:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with path.open("rb") as stream:
            row_count = sum(1 for line in stream if line.strip())
        return path, digest.hexdigest(), row_count

    temporary = tempfile.NamedTemporaryFile(prefix="hpm-memory-sample-", suffix=".jsonl",
                                             delete=False)
    rows = 0
    try:
        with path.open("rb") as source, temporary:
            for line in source:
                if not line.strip():
                    continue
                temporary.write(line)
                digest.update(line)
                rows += 1
                if rows >= max_rows:
                    break
    except Exception:
        Path(temporary.name).unlink(missing_ok=True)
        raise
    if rows == 0:
        Path(temporary.name).unlink(missing_ok=True)
        raise ValueError(f"no JSONL rows found in {path}")
    return Path(temporary.name), digest.hexdigest(), rows


def profile(corpus: Path, output_dir: Path, max_rows: int | None = None,
            hot_cap: int = 10_000, max_vocab: int = 20_000) -> dict:
    sampled_path, corpus_sha256, row_count = _bounded_input(corpus, max_rows)
    process = psutil.Process()
    state: dict[str, object] = {"peak_rss_bytes": process.memory_info().rss,
                                "stop": False, "agent": None}

    def sample_rss() -> None:
        while not state["stop"]:
            state["peak_rss_bytes"] = max(
                int(state["peak_rss_bytes"]), process.memory_info().rss
            )
            time.sleep(0.1)

    sampler = threading.Thread(target=sample_rss, name="pretrain-rss-sampler", daemon=True)
    sampler.start()
    original_init = pretraining.ReaderAgent.__init__

    def capture_agent(agent, *args, **kwargs):
        original_init(agent, *args, **kwargs)
        # ReaderAgent already creates TieredForest before ingest starts. Apply
        # the requested cache cap there so the profiler also works against
        # checkpoints whose pretrain_corpus API predates a hot_cap argument.
        agent.forest._hot_cap = hot_cap
        state["agent"] = agent

    pretraining.ReaderAgent.__init__ = capture_agent
    started = time.perf_counter()
    try:
        result = pretraining.pretrain_corpus(
            [sampled_path], output_dir, max_vocab=max_vocab,
        )
    finally:
        pretraining.ReaderAgent.__init__ = original_init
        state["stop"] = True
        sampler.join()
        if sampled_path != corpus:
            sampled_path.unlink(missing_ok=True)

    agent = state["agent"]
    if agent is None:
        raise RuntimeError("pretraining completed without constructing a ReaderAgent")
    forest = agent.forest
    loaded_items = getattr(agent.patterns, "loaded_items", None)
    result["memory_profile"] = {
        "measurement": "sampled process RSS every 100 ms; vector-index bytes sum NumPy array sizes",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(state["peak_rss_bytes"]),
        "process_rss_after_pretraining_bytes": process.memory_info().rss,
        "forest_vector_index_nodes": len(forest._mu_index),
        "forest_vector_index_bytes": sum(vector.nbytes for vector in forest._mu_index.values()),
        "forest_hot_node_count": forest.hot_count(),
        "pattern_index_loaded_object_count": (
            len(loaded_items()) if loaded_items is not None else len(agent.patterns)
        ),
        "term_index_terms": len(getattr(agent, "term_sentences", {})),
        "term_index_links": sum(map(len, getattr(agent, "term_sentences", {}).values())),
    }
    result["profile_input"] = {
        "corpus": str(corpus), "row_count": row_count,
        "sample_sha256": corpus_sha256, "max_rows": max_rows,
        "hot_cap": hot_cap, "max_vocab": max_vocab,
    }
    project_root = Path(__file__).resolve().parents[2]
    code_paths = (
        "hpm_ai_v2/experiments/profile_pretrain_memory.py",
        "hpm_ai_v2/training/pretrain_hfn_corpus.py",
        "hpm_ai_v2/agents/base_agent.py",
        "hfn/tiered_forest.py",
    )
    result["code_provenance"] = {
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "files_sha256": {
            name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
            for name in code_paths
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="JSONL source corpus to profile")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for the pretraining checkpoint")
    parser.add_argument("--report", type=Path, required=True,
                        help="JSON path for memory and pretraining metrics")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="profile only the first N non-empty JSONL rows")
    parser.add_argument("--hot-cap", type=int, default=10_000)
    parser.add_argument("--max-vocab", type=int, default=20_000)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    report = profile(args.corpus, args.output_dir, args.max_rows,
                     hot_cap=args.hot_cap, max_vocab=args.max_vocab)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
