"""Two-turn ambiguity/clarification benchmark for v5 comprehension feedback."""

from __future__ import annotations

from dataclasses import dataclass

from hpm_ai_v5.adapter.atis import load_atis
from hpm_ai_v5.adapter.snips import load_snips
from hpm_ai_v5.experiments.run_atis_benchmark import ATISBenchmark
from hpm_ai_v5.experiments.run_snips_benchmark import SNIPSBenchmark


@dataclass(frozen=True, slots=True)
class ClarificationCase:
    turn1: str
    turn2: str
    gold_intent: str


ATIS_CASES = [
    ClarificationCase(
        turn1="i need information on a ticket from denver to pittsburgh",
        turn2="what is the airfare",
        gold_intent="airfare",
    ),
    ClarificationCase(
        turn1="what day do flights from nashville to tacoma fly",
        turn2="i need the day of the week",
        gold_intent="day_name",
    ),
    ClarificationCase(
        turn1="what kind of transportation is available in denver",
        turn2="i mean ground transportation from the airport",
        gold_intent="ground_service",
    ),
    ClarificationCase(
        turn1="which carrier goes from new york to milwaukee",
        turn2="which airline serves that route",
        gold_intent="airline",
    ),
]


SNIPS_CASES = [
    ClarificationCase(
        turn1="find something playing tonight",
        turn2="i mean a movie screening at the theater",
        gold_intent="search_screening_event",
    ),
    ClarificationCase(
        turn1="book something near downtown",
        turn2="reserve a restaurant table for two",
        gold_intent="book_restaurant",
    ),
    ClarificationCase(
        turn1="play something for me",
        turn2="play songs by coldplay",
        gold_intent="play_music",
    ),
    ClarificationCase(
        turn1="tell me about tomorrow",
        turn2="what will the weather forecast be tomorrow",
        gold_intent="get_weather",
    ),
]


def _score_cases(predictor, inference_agent, cases: list[ClarificationCase], *, clear_carry_each_turn2: bool) -> float:
    correct = 0
    for case in cases:
        predictor(case.turn1)
        if clear_carry_each_turn2 and inference_agent is not None:
            inference_agent.carry_context = {}
        if predictor(case.turn2) == case.gold_intent:
            correct += 1
    return correct / max(len(cases), 1)


def run_snips_clarification_benchmark() -> dict[str, float]:
    train, _ = load_snips()
    bench = SNIPSBenchmark()
    bench.train(train)
    acc_with = _score_cases(bench._predict, bench._inference_agent, SNIPS_CASES, clear_carry_each_turn2=False)

    bench_no = SNIPSBenchmark()
    bench_no.train(train)
    acc_without = _score_cases(bench_no._predict, bench_no._inference_agent, SNIPS_CASES, clear_carry_each_turn2=True)
    return {
        "with_feedback": acc_with,
        "without_feedback": acc_without,
        "delta": acc_with - acc_without,
    }


def run_atis_clarification_benchmark() -> dict[str, float]:
    train, test = load_atis()
    bench = ATISBenchmark()
    bench._CHECKPOINT = "/tmp/atis_multiturn_clarification.pkl"
    bench.run_b1(train[:120], test[:20])
    acc_with = _score_cases(bench._run_and_predict, bench._inference_agent, ATIS_CASES, clear_carry_each_turn2=False)

    bench_no = ATISBenchmark()
    bench_no._CHECKPOINT = "/tmp/atis_multiturn_clarification_no.pkl"
    bench_no.run_b1(train[:120], test[:20])
    acc_without = _score_cases(bench_no._run_and_predict, bench_no._inference_agent, ATIS_CASES, clear_carry_each_turn2=True)
    return {
        "with_feedback": acc_with,
        "without_feedback": acc_without,
        "delta": acc_with - acc_without,
    }


def run_all() -> None:
    print("Running multi-turn clarification benchmark...")
    snips = run_snips_clarification_benchmark()
    atis = run_atis_clarification_benchmark()

    print("\nMULTI-TURN CLARIFICATION RESULTS")
    print(f"SNIPS with feedback:    {snips['with_feedback']:.2%}")
    print(f"SNIPS without feedback: {snips['without_feedback']:.2%}")
    print(f"SNIPS delta:            {snips['delta']:+.2%}")
    print(f"ATIS with feedback:     {atis['with_feedback']:.2%}")
    print(f"ATIS without feedback:  {atis['without_feedback']:.2%}")
    print(f"ATIS delta:             {atis['delta']:+.2%}")


if __name__ == "__main__":
    run_all()
