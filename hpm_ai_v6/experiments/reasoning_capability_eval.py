from __future__ import annotations

import os
import re
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
from hpm_ai_v6.agents.causal_agent import CausalRule
from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.evaluators.reasoning_evaluator import ReasoningBenchmarkCase, ReasoningEvaluator
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None, sent_text_by_name=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}
        self.sent_text_by_name = sent_text_by_name or {}

    def get_weights(self):
        return list(self._weights)

    def _paging_lookup(self):
        return dict(self._lookup)


@dataclass(frozen=True)
class FeedbackABCaseResult:
    query: str
    before_has_path: bool
    after_has_path: bool
    before_steps: int
    after_steps: int
    before_latency_ms: float
    after_latency_ms: float


@dataclass
class FeedbackABReport:
    case_results: List[FeedbackABCaseResult]
    derived_edges_added: int
    focus_words: List[str]

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def before_path_rate(self) -> float:
        return sum(1 for case in self.case_results if case.before_has_path) / max(len(self.case_results), 1)

    @property
    def after_path_rate(self) -> float:
        return sum(1 for case in self.case_results if case.after_has_path) / max(len(self.case_results), 1)

    @property
    def before_avg_steps(self) -> float:
        return sum(case.before_steps for case in self.case_results) / max(len(self.case_results), 1)

    @property
    def after_avg_steps(self) -> float:
        return sum(case.after_steps for case in self.case_results) / max(len(self.case_results), 1)

    @property
    def improved_cases(self) -> int:
        return sum(
            1
            for case in self.case_results
            if case.after_has_path and (
                not case.before_has_path or case.after_steps < case.before_steps
            )
        )

    def render_text(self) -> str:
        lines = [
            f"cases={self.total_cases}",
            f"derived_edges_added={self.derived_edges_added}",
            f"focus_words={','.join(self.focus_words) if self.focus_words else '-'}",
            f"before_path_rate={self.before_path_rate:.3f}",
            f"after_path_rate={self.after_path_rate:.3f}",
            f"before_avg_steps={self.before_avg_steps:.2f}",
            f"after_avg_steps={self.after_avg_steps:.2f}",
            f"improved_cases={self.improved_cases}",
        ]
        for case in self.case_results:
            lines.append(
                f"AB query={case.query!r} before(path={case.before_has_path},steps={case.before_steps},latency_ms={case.before_latency_ms:.2f}) "
                f"after(path={case.after_has_path},steps={case.after_steps},latency_ms={case.after_latency_ms:.2f})"
            )
        return "\n".join(lines)


def make_edge(name, source, target):
    return Cell(
        name=name,
        dim=1,
        embedding=target.as_numpy() - source.as_numpy(),
        source=source,
        target=target,
    )


def make_analogy(name, source_pattern, target_pattern, weight=1.0):
    return Cell(
        name=name,
        dim=2,
        embedding=target_pattern.as_numpy() - source_pattern.as_numpy(),
        source=source_pattern,
        target=target_pattern,
        weight=weight,
    )


def make_rule_with_metadata(name, source_pattern, target_pattern, metadata, weight=1.0):
    return Cell(
        name=name,
        dim=3,
        embedding=target_pattern.as_numpy() - source_pattern.as_numpy(),
        source=source_pattern,
        target=target_pattern,
        weight=weight,
        metadata=metadata,
    )


def build_synthetic_reasoning_agent() -> ReasoningAgent:
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])
    delta = Cell(name="word_delta", dim=0, embedding=[1.0, 1.0, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 1.0, 1.0])
    crown = Cell(name="word_crown", dim=0, embedding=[1.0, 0.0, 1.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.2, 0.8, 0.2])
    hole = Cell(name="word_hole", dim=0, embedding=[0.1, 0.2, 0.9])
    burrow = Cell(name="word_burrow", dim=0, embedding=[0.8, 0.2, 0.9])

    ab = make_edge("w_alpha->beta", alpha, beta)
    bg = make_edge("w_beta->gamma", beta, gamma)
    gd = make_edge("w_gamma->delta", gamma, delta)
    aq = make_edge("w_alpha->queen", alpha, queen)
    qc = make_edge("w_queen->crown", queen, crown)
    rh = make_edge("w_rabbit->hole", rabbit, hole)
    hb = make_edge("w_hole->burrow", hole, burrow)

    analogy = make_analogy("analogy_royal", ab, qc, weight=0.93)
    helper_src = make_analogy("shared_src_helper", ab, aq, weight=0.7)
    helper_tgt = make_analogy("shared_src_helper_tgt", aq, ab, weight=0.68)
    subgraph_rule = make_rule_with_metadata(
        "shared_source_rule",
        helper_src,
        helper_tgt,
        metadata={
            "rule_type": "subgraph_derivation",
            "pair_mode": "any_reachable",
            "antecedent_edges": [
                {"source_var": "x", "target_var": "y"},
                {"source_var": "x", "target_var": "z"},
            ],
            "consequent": {
                "source_var": "y",
                "target_var": "z",
            },
        },
        weight=0.91,
    )
    transitivity_rule = Cell(name="transitivity_rule", dim=3, embedding=[0.1, 0.1, 0.1], weight=0.95)

    cause = Cell(name="cause_rabbit->suddenly@2", dim=0, embedding=[1.0, 2.0, 1.0])
    effect = Cell(name="effect_rabbit->semantic", dim=0, embedding=[1.0, 1.0, 1.0])
    causal_rule = CausalRule(
        name="causal_rabbit_in_semantic",
        intervention="replace 'rabbit' at pos 2",
        effect_magnitude=0.42,
        agent_impacted="semantic",
        source=cause,
        target=effect,
        metadata={"original_word": "rabbit", "agent_impacted": "semantic"},
    )

    word_patterns = [ab, bg, gd, aq, qc, rh, hb, analogy, helper_src, helper_tgt, subgraph_rule, transitivity_rule]
    word_weights = [0.82, 0.79, 0.77, 0.7, 0.74, 0.81, 0.76, 0.93, 0.7, 0.68, 0.91, 0.95]
    lookup = {cell.name: cell for cell in [alpha, beta, gamma, delta, queen, crown, rabbit, hole, burrow]}

    reader = SimpleNamespace(
        agents={
            "word": StubAgent(patterns=word_patterns, weights=word_weights, lookup=lookup),
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": StubAgent(patterns=[causal_rule], weights=[0.9], lookup={cause.name: cause, effect.name: effect}),
        }
    )
    return ReasoningAgent(reader, beam_width=5, max_depth=4)


def build_synthetic_benchmark_cases() -> List[ReasoningBenchmarkCase]:
    return [
        ReasoningBenchmarkCase(
            name="beam_connection",
            question="How does alpha connect to delta?",
            method="beam",
            expected_intent="path",
            expected_mode="connection",
            expected_nodes=["alpha", "beta", "delta"],
            expected_relations=["rule_application", "forward_chain"],
        ),
        ReasoningBenchmarkCase(
            name="backward_connection",
            question="How does alpha connect to delta?",
            method="backward",
            expected_intent="path",
            expected_mode="connection",
            expected_nodes=["alpha", "beta", "gamma", "delta"],
            expected_relations=["rule_application", "lexical_transition", "lexical_transition"],
        ),
        ReasoningBenchmarkCase(
            name="auto_analogy",
            question="What is analogous to alpha beta?",
            method="auto",
            expected_intent="analogy",
            expected_answer_contains=["strongest explicit learned analogies", "w_queen->crown"],
            max_steps=0,
        ),
        ReasoningBenchmarkCase(
            name="backward_subgraph",
            question="How does beta connect to queen?",
            method="backward",
            expected_intent="path",
            expected_mode="connection",
            expected_nodes=["beta", "queen"],
            expected_relations=["subgraph_rule"],
        ),
        ReasoningBenchmarkCase(
            name="causal_explanation",
            question="Why does surprise in semantic after rabbit connect to burrow?",
            method="backward",
            expected_intent="path",
            expected_mode="explanation",
            expected_answer_contains=["strongest causal explanation"],
            expected_nodes=["rabbit to suddenly at 2", "surprise in semantic after rabbit", "burrow"],
            expected_relations=["causal_relation", "subgraph_rule"],
        ),
    ]


def build_feedback_loop_reader():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.2, 0.8, 0.2])
    hole = Cell(name="word_hole", dim=0, embedding=[0.1, 0.2, 0.9])

    tmp_corpus = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus", "alice_mini.txt")
    reader = MultiAgentReader(tmp_corpus, warm_start=False)

    edge_alpha_beta = make_edge("w_alpha->beta", alpha, beta)
    edge_beta_gamma = make_edge("w_beta->gamma", beta, gamma)
    edge_rabbit_hole = make_edge("w_rabbit->hole", rabbit, hole)
    reader.word_agent.patterns = [edge_alpha_beta, edge_beta_gamma, edge_rabbit_hole]
    reader.word_agent.word_cells = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "rabbit": rabbit,
        "hole": hole,
    }
    reader.word_agent._refresh_learner()
    reader.reasoning_agent.invalidate()
    return reader


def run_reasoning_feedback_ab(
    reader: MultiAgentReader,
    queries: Sequence[str],
    *,
    reflect_queries: Sequence[str] | None = None,
) -> FeedbackABReport:
    query_list = list(queries)
    reflection_queries = list(reflect_queries or query_list)
    before_traces = []
    case_results: List[FeedbackABCaseResult] = []

    for query in query_list:
        start = time.perf_counter()
        trace = reader.reasoning_agent.reason_with_trace(query)
        before_latency_ms = (time.perf_counter() - start) * 1000.0
        before_traces.append((query, trace, before_latency_ms))

    signal = reader.reasoning_agent.reflect(reflection_queries)
    added = 0
    seen = {pattern.name for pattern in reader.word_agent.patterns}
    for src, tgt, score in signal.derived_edges:
        name = f"derived_{src.name}_{tgt.name}"
        if name not in seen:
            added += 1
        reader._reinforce_edge(src, tgt, score)
        seen.add(name)

    after_traces = {}
    for query in query_list:
        start = time.perf_counter()
        trace = reader.reasoning_agent.reason_with_trace(query)
        after_latency_ms = (time.perf_counter() - start) * 1000.0
        after_traces[query] = (trace, after_latency_ms)

    for query, before_trace, before_latency_ms in before_traces:
        after_trace, after_latency_ms = after_traces[query]
        before_steps = len((before_trace.get("chosen_path") or {}).get("steps") or [])
        after_steps = len((after_trace.get("chosen_path") or {}).get("steps") or [])
        case_results.append(
            FeedbackABCaseResult(
                query=query,
                before_has_path=bool(before_trace.get("chosen_path")),
                after_has_path=bool(after_trace.get("chosen_path")),
                before_steps=before_steps,
                after_steps=after_steps,
                before_latency_ms=before_latency_ms,
                after_latency_ms=after_latency_ms,
            )
        )

    return FeedbackABReport(
        case_results=case_results,
        derived_edges_added=added,
        focus_words=list(signal.suggested_focus_words),
    )


def _split_story_sentences(text: str) -> List[str]:
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if len(chunk.strip()) > 20]
    return [
        sentence
        for sentence in sentences
        if not any(
            marker in sentence.lower()
            for marker in (
                "project gutenberg",
                "release date",
                "updated editions",
                "author:",
                "contents",
                "*** start of",
                "*** end of",
            )
        )
    ]


def extract_story_sentences(
    corpus_path: str,
    limit: int = 12,
    focus_terms: Sequence[str] = ("alice", "rabbit", "hole"),
    context_window: int = 2,
) -> List[str]:
    text = Path(corpus_path).read_text(encoding="utf-8")
    sentences = _split_story_sentences(text)
    match_indices = [
        idx
        for idx, sentence in enumerate(sentences)
        if any(term in sentence.lower() for term in focus_terms)
    ]
    if not match_indices:
        return sentences[:limit]

    selected: List[str] = []
    for index in match_indices:
        for candidate in range(max(0, index - context_window), min(len(sentences), index + context_window + 1)):
            sentence = sentences[candidate]
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= limit:
                return selected

    if len(selected) < limit:
        for sentence in sentences:
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= limit:
                break
    return selected


def _cell_key(name: str) -> str:
    name = name.strip()
    if name.startswith(("word_", "sent_", "ctx_", "char_", "pos_", "cause_", "effect_")):
        return name.replace("_", ":", 1)
    return f"word:{name}"


def verify_expected_edges(reasoning_agent: ReasoningAgent, expected_edges: Sequence[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    missing: List[Tuple[str, str]] = []
    present: List[Tuple[str, str]] = []
    for source_name, target_name in expected_edges:
        source_cells = reasoning_agent._resolve_cells(source_name)
        target_cells = reasoning_agent._resolve_cells(target_name)
        found = False
        for source_cell in source_cells:
            source_key = reasoning_agent._cell_key(source_cell)
            edges = reasoning_agent._edge_index.get(source_key, [])
            for target_cell in target_cells:
                target_key = reasoning_agent._cell_key(target_cell)
                if any(edge.target_key == target_key for edge in edges):
                    found = True
                    break
            if found:
                break
        if found:
            present.append((source_name, target_name))
        else:
            missing.append((source_name, target_name))
    return {"present": present, "missing": missing}


def corpus_pattern_cache_dir(corpus_path: str) -> str:
    corpus_name = Path(corpus_path).stem
    corpus_dir = Path(corpus_path).resolve().parent
    return str(corpus_dir / ".hpm_pattern_cache" / corpus_name)


def build_corpus_reasoning_agent(corpus_path: str, train_sentence_limit: int = 12, training_epochs: int = 2) -> ReasoningAgent:
    reader = build_corpus_feedback_reader(
        corpus_path,
        train_sentence_limit=train_sentence_limit,
        training_epochs=training_epochs,
        warm_start=True,
    )
    return reader.reasoning_agent


def build_corpus_feedback_reader(
    corpus_path: str,
    train_sentence_limit: int = 12,
    training_epochs: int = 2,
    *,
    warm_start: bool = True,
) -> MultiAgentReader:
    reader = MultiAgentReader(
        corpus_path,
        pattern_cache_dir=corpus_pattern_cache_dir(corpus_path),
        warm_start=warm_start,
    )
    expected_edges = [
        ("alice", "rabbit"),
        ("rabbit", "hole"),
        ("alice", "hole"),
    ]
    passes = [
        {"limit": train_sentence_limit, "context_window": 2, "epochs": training_epochs},
        {"limit": train_sentence_limit * 2, "context_window": 3, "epochs": max(1, training_epochs - 1)},
        {"limit": train_sentence_limit * 3, "context_window": 4, "epochs": 1},
    ]
    trained_count = 0
    for idx, plan in enumerate(passes, start=1):
        sentences = extract_story_sentences(
            corpus_path,
            limit=plan["limit"],
            context_window=plan["context_window"],
        )
        new_sentences = sentences[trained_count:] if trained_count < len(sentences) else []
        if new_sentences:
            for _ in range(max(plan["epochs"], 1)):
                for sentence in new_sentences:
                    reader.train_sequence([sentence], enable_causal=False)
        trained_count = max(trained_count, len(sentences))
        edge_report = verify_expected_edges(reader.reasoning_agent, expected_edges)
        status = "passed" if not edge_report["missing"] else "missing"
        print(
            f"Corpus edge check pass {idx} {status}: "
            + (", ".join(f"{src}->{tgt}" for src, tgt in edge_report["present"]) or "none")
        )
        if not edge_report["missing"]:
            break
    return reader


def build_corpus_benchmark_cases() -> List[ReasoningBenchmarkCase]:
    return [
        ReasoningBenchmarkCase(
            name="corpus_alice_to_rabbit",
            question="How does alice connect to rabbit?",
            method="auto",
            expected_intent="path",
            expected_mode="connection",
            expected_answer_contains=["alice", "rabbit"],
            forbidden_answer_contains=["could not find", "no strong"],
            require_chosen_path=True,
        ),
        ReasoningBenchmarkCase(
            name="corpus_rabbit_to_hole",
            question="How does rabbit connect to hole?",
            method="auto",
            expected_intent="path",
            expected_mode="connection",
            expected_answer_contains=["rabbit", "hole"],
            forbidden_answer_contains=["could not find", "no strong"],
            require_chosen_path=True,
        ),
        ReasoningBenchmarkCase(
            name="corpus_analogy_probe",
            question="What is analogous to alice rabbit?",
            method="auto",
            expected_intent="analogy",
            expected_answer_contains=["analog"],
            forbidden_answer_contains=["no strong analogical matches"],
        ),
    ]


def build_corpus_feedback_queries() -> List[str]:
    return [
        "How does alice connect to hole?",
        "How does rabbit connect to hole?",
    ]


def main() -> None:
    agent = build_synthetic_reasoning_agent()
    evaluator = ReasoningEvaluator(agent)
    synthetic_report = evaluator.evaluate_cases(build_synthetic_benchmark_cases())
    print("== Synthetic Benchmark ==")
    print(synthetic_report.render_text())

    feedback_reader = build_feedback_loop_reader()
    feedback_report = run_reasoning_feedback_ab(
        feedback_reader,
        queries=["How does alpha connect to gamma?"],
    )
    print("\n== Feedback A/B Benchmark ==")
    print(feedback_report.render_text())

    corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus", "alice_mini.txt")
    corpus_feedback_reader = build_corpus_feedback_reader(corpus_path)
    corpus_feedback_report = run_reasoning_feedback_ab(
        corpus_feedback_reader,
        queries=build_corpus_feedback_queries(),
    )
    print("\n== Corpus Feedback A/B Benchmark ==")
    print(corpus_feedback_report.render_text())

    corpus_agent = corpus_feedback_reader.reasoning_agent
    corpus_evaluator = ReasoningEvaluator(corpus_agent)
    corpus_report = corpus_evaluator.evaluate_cases(build_corpus_benchmark_cases())
    print("\n== Corpus Benchmark ==")
    print(corpus_report.render_text())


if __name__ == "__main__":
    main()
