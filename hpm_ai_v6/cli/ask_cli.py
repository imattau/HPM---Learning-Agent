"""Interactive CLI for asking the HPM reasoning agent questions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Iterable, Optional


def _corpus_path() -> str:
    """Return the bundled local corpus path used by the reader stack."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "corpus", "alice_mini.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at {path}")
    return path


def _pattern_cache_dir(corpus_path: str) -> str:
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
    corpus_dir = os.path.dirname(os.path.abspath(corpus_path))
    return os.path.join(corpus_dir, ".hpm_pattern_cache", corpus_name)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the HPM reasoning agent")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive prompt mode even if a question was provided.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "beam", "backward"],
        default="auto",
        help="Reasoning method to use (default: auto).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full trace payload after the compact summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full trace payload as JSON only.",
    )
    return parser.parse_args(argv)


def build_reader():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader

    corpus = _corpus_path()
    return MultiAgentReader(
        corpus,
        pattern_cache_dir=_pattern_cache_dir(corpus),
        warm_start=True,
    )


def _trace_has_path(trace: dict[str, Any]) -> bool:
    chosen = trace.get("chosen_path")
    return bool(chosen and chosen.get("nodes"))


def _path_labels(path: dict[str, Any]) -> str:
    nodes = path.get("nodes") or []
    labels = [node.get("label", "?") for node in nodes if isinstance(node, dict)]
    return " -> ".join(labels) if labels else "(empty)"


def _summarize_candidates(paths: Iterable[dict[str, Any]], limit: int = 3) -> list[str]:
    summary: list[str] = []
    for idx, path in enumerate(paths):
        if idx >= limit:
            break
        labels = _path_labels(path)
        score = path.get("combined_score")
        if isinstance(score, (int, float)):
            summary.append(f"{idx + 1}. {labels} (score={score:.3f})")
        else:
            summary.append(f"{idx + 1}. {labels}")
    return summary


def format_trace_summary(trace: dict[str, Any]) -> list[str]:
    lines = [
        f"Answer: {trace.get('answer', '')}",
        f"Intent: {trace.get('intent', 'unknown')}",
        f"Mode: {trace.get('mode', 'unknown')}",
        f"Method: {trace.get('method', 'auto')}",
        f"Chosen path: {'yes' if _trace_has_path(trace) else 'no'}",
    ]

    chosen = trace.get("chosen_path")
    if isinstance(chosen, dict) and chosen:
        lines.append(f"Chosen path detail: {_path_labels(chosen)}")

    candidate_paths = trace.get("candidate_paths") or []
    if candidate_paths:
        lines.append("Candidate paths:")
        lines.extend(f"  {line}" for line in _summarize_candidates(candidate_paths))

    evidence = trace.get("evidence") or []
    if evidence:
        lines.append("Evidence:")
        for entry in evidence[:3]:
            if not isinstance(entry, dict):
                continue
            parts = []
            if entry.get("relation"):
                parts.append(str(entry["relation"]))
            if entry.get("agent"):
                parts.append(str(entry["agent"]))
            if entry.get("pattern"):
                parts.append(str(entry["pattern"]))
            if entry.get("score") is not None:
                parts.append(f"score={entry['score']:.3f}" if isinstance(entry["score"], (int, float)) else f"score={entry['score']}")
            if parts:
                lines.append(f"  - {' | '.join(parts)}")

    return lines


def run_question(
    reasoning_agent: Any,
    question: str,
    *,
    method: str = "auto",
    verbose: bool = False,
    json_mode: bool = False,
    stream: Any = None,
) -> dict[str, Any]:
    if stream is None:
        stream = sys.stdout
    trace = reasoning_agent.reason_with_trace(question, method=method)
    if json_mode:
        json.dump(trace, stream, indent=2, sort_keys=True)
        stream.write("\n")
        return trace

    for line in format_trace_summary(trace):
        print(line, file=stream)
    if verbose:
        print("Trace payload:", file=stream)
        print(json.dumps(trace, indent=2, sort_keys=True), file=stream)
    return trace


def run_interactive(
    reasoning_agent: Any,
    *,
    method: str = "auto",
    verbose: bool = False,
    json_mode: bool = False,
    input_fn: Callable[[str], str] = input,
    stream: Any = None,
) -> None:
    if stream is None:
        stream = sys.stdout
    prompt = "Ask a question (type 'quit' to exit): "
    while True:
        try:
            question = input_fn(prompt)
        except EOFError:
            print("", file=stream)
            break
        if question is None:
            break
        stripped = question.strip()
        if not stripped:
            continue
        if stripped.lower() in {"quit", "exit", "q"}:
            break
        run_question(
            reasoning_agent,
            stripped,
            method=method,
            verbose=verbose,
            json_mode=json_mode,
            stream=stream,
        )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    reader = build_reader()
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        raise RuntimeError("MultiAgentReader did not expose reasoning_agent")

    if args.json and args.verbose:
        raise SystemExit("--json and --verbose are mutually exclusive")

    if args.interactive or not args.question:
        run_interactive(
            reasoning_agent,
            method=args.method,
            verbose=args.verbose,
            json_mode=args.json,
        )
    else:
        run_question(
            reasoning_agent,
            args.question,
            method=args.method,
            verbose=args.verbose,
            json_mode=args.json,
        )


if __name__ == "__main__":
    main()
