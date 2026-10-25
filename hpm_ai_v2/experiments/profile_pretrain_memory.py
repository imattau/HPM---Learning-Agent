"""Measure process RSS and retained HFN indexes during source-only pretraining.

Example::

    python -m hpm_ai_v2.experiments.profile_pretrain_memory \
        --corpus data/pretraining/oasst1_train_en_1000.jsonl \
        --max-rows 300 --output-dir /tmp/hpm-oasst-300 \
        --report /tmp/hpm-oasst-300-memory.json

The profiler keeps process memory sampling separate from the training report,
and reports the bytes held by TieredForest's all-node vector index. An
optional RSS ceiling interrupts the diagnostic run and removes its partial
checkpoint; it does not change the training objective or model updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import _thread
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import psutil

import hpm_ai_v2.training.pretrain_hfn_corpus as pretraining


class RSSLimitExceeded(MemoryError):
    """Carry measurements from a safely aborted profile run."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            "pretraining stopped at the RSS safety limit; partial report is available"
        )


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


def _code_provenance() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    code_paths = (
        "hpm_ai_v2/experiments/profile_pretrain_memory.py",
        "hpm_ai_v2/training/pretrain_hfn_corpus.py",
        "hpm_ai_v2/agents/base_agent.py",
        "hfn/tiered_forest.py",
    )
    return {
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "files_sha256": {
            name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
            for name in code_paths
        },
    }


def _memory_profile(agent, process, peak_rss: int, started: float,
                    rss_safety_limit_gib: float | None) -> dict:
    forest = agent.forest if agent is not None else None
    loaded_items = getattr(agent.patterns, "loaded_items", None) if agent is not None else None
    return {
        "measurement": "sampled process RSS every 100 ms; vector-index bytes sum NumPy array sizes",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(peak_rss),
        "rss_safety_limit_gib": rss_safety_limit_gib,
        "process_rss_at_capture_bytes": process.memory_info().rss,
        "forest_vector_index_nodes": len(forest._mu_index) if forest is not None else None,
        "forest_vector_index_bytes": (
            sum(vector.nbytes for vector in forest._mu_index.values())
            if forest is not None else None
        ),
        "forest_hot_node_count": forest.hot_count() if forest is not None else None,
        "pattern_index_loaded_object_count": (
            len(loaded_items()) if loaded_items is not None
            else len(agent.patterns) if agent is not None else None
        ),
        "term_index_terms": len(getattr(agent, "term_sentences", {})) if agent is not None else None,
        "term_index_links": (
            sum(map(len, getattr(agent, "term_sentences", {}).values()))
            if agent is not None else None
        ),
    }


def profile(corpus: Path, output_dir: Path, max_rows: int | None = None,
            hot_cap: int = 10_000, max_vocab: int = 20_000,
            max_rss_gib: float | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if max_rss_gib is not None and max_rss_gib <= 0:
        raise ValueError("max_rss_gib must be positive")
    sampled_path, corpus_sha256, row_count = _bounded_input(corpus, max_rows)
    process = psutil.Process()
    state: dict[str, object] = {"peak_rss_bytes": process.memory_info().rss,
                                "stop": False, "agent": None,
                                "rss_limit_bytes": int(max_rss_gib * 1024**3)
                                if max_rss_gib is not None else None,
                                "rss_limit_exceeded": False}

    def sample_rss() -> None:
        while not state["stop"]:
            current_rss = process.memory_info().rss
            state["peak_rss_bytes"] = max(int(state["peak_rss_bytes"]), current_rss)
            limit = state["rss_limit_bytes"]
            if limit is not None and current_rss >= int(limit):
                state["rss_limit_exceeded"] = True
                _thread.interrupt_main()
                return
            time.sleep(0.1)

    sampler = threading.Thread(target=sample_rss, name="pretrain-rss-sampler", daemon=True)
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
        sampler.start()
        result = pretraining.pretrain_corpus(
            [sampled_path], output_dir, max_vocab=max_vocab,
        )
    except KeyboardInterrupt as error:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if state["rss_limit_exceeded"]:
            report = {
                "profile_status": "aborted_rss_limit",
                "pretraining_report": None,
                "memory_profile": _memory_profile(
                    state["agent"], process, int(state["peak_rss_bytes"]), started,
                    max_rss_gib,
                ),
                "profile_input": {
                    "corpus": str(corpus), "row_count": row_count,
                    "sample_sha256": corpus_sha256, "max_rows": max_rows,
                    "hot_cap": hot_cap, "max_vocab": max_vocab,
                },
                "code_provenance": _code_provenance(),
            }
            raise RSSLimitExceeded(report) from error
        raise
    finally:
        pretraining.ReaderAgent.__init__ = original_init
        state["stop"] = True
        if sampler.ident is not None:
            sampler.join()
        if sampled_path != corpus:
            sampled_path.unlink(missing_ok=True)

    agent = state["agent"]
    if agent is None:
        raise RuntimeError("pretraining completed without constructing a ReaderAgent")
    if agent is None:
        raise RuntimeError("pretraining completed without constructing a ReaderAgent")
    result["memory_profile"] = {
        **_memory_profile(agent, process, int(state["peak_rss_bytes"]), started,
                          max_rss_gib),
    }
    result["profile_input"] = {
        "corpus": str(corpus), "row_count": row_count,
        "sample_sha256": corpus_sha256, "max_rows": max_rows,
        "hot_cap": hot_cap, "max_vocab": max_vocab,
    }
    result["profile_status"] = "completed"
    result["code_provenance"] = _code_provenance()
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
    parser.add_argument("--max-rss-gib", type=float, default=None,
                        help="interrupt and remove the partial checkpoint at this process RSS")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = profile(args.corpus, args.output_dir, args.max_rows,
                         hot_cap=args.hot_cap, max_vocab=args.max_vocab,
                         max_rss_gib=args.max_rss_gib)
    except RSSLimitExceeded as error:
        args.report.write_text(json.dumps(error.report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(json.dumps(error.report, indent=2, ensure_ascii=False))
        parser.exit(2, f"RSS limit exceeded; partial measurements saved to {args.report}\n")
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
