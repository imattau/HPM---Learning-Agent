"""SP Experiment: Reader Agent Retrieval Accuracy vs Random Baseline."""
from __future__ import annotations
from typing import Dict
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

CORPUS = [
    ("tech", "Machine learning is a subset of artificial intelligence that trains on data."),
    ("tech", "Deep learning uses neural networks with many layers to learn representations."),
    ("tech", "Python is a high-level programming language widely used in data science."),
    ("tech", "Neural networks are inspired by biological neurons in the human brain."),
    ("history", "The Roman Empire was one of the largest empires in ancient history."),
    ("history", "Rome was founded in 753 BC according to ancient Roman tradition."),
    ("history", "Latin was the official language of the Roman Empire for centuries."),
    ("history", "The fall of Rome in 476 AD marked the end of the Western Roman Empire."),
    ("nature", "Photosynthesis is the process by which plants convert sunlight to energy."),
    ("nature", "Rainforests contain over half of the world's plant and animal species."),
    ("nature", "Migration patterns of birds are influenced by seasonal temperature changes."),
    ("nature", "The Amazon River basin supports one of the most diverse ecosystems on Earth."),
]

QUERIES = [
    ("machine learning neural networks deep learning", "tech"),
    ("roman empire ancient history latin", "history"),
    ("photosynthesis plants rainforest nature", "nature"),
    ("python programming data science", "tech"),
    ("amazon river ecosystem species", "nature"),
    ("rome founded BC tradition", "history"),
]


def run_experiment(verbose: bool = True) -> Dict:
    passages = [text for _, text in CORPUS]
    labels = [label for label, _ in CORPUS]
    config = TextDomainConfig.from_passages(passages, max_vocab=100)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)

    correct = 0
    total = len(QUERIES)
    for query_text, expected_label in QUERIES:
        result = agent.query(query_text)
        if result is None:
            continue
        idx = passages.index(result) if result in passages else -1
        returned_label = labels[idx] if idx >= 0 else "unknown"
        if returned_label == expected_label:
            correct += 1
        if verbose:
            print(f"Query: {query_text[:40]}...")
            print(f"  Expected: {expected_label}, Got: {returned_label} ({'✓' if returned_label == expected_label else '✗'})")

    precision_at_1 = correct / total
    n_labels = len(set(labels))
    random_baseline = 1.0 / n_labels

    result_dict = {
        "precision_at_1": precision_at_1,
        "random_baseline": random_baseline,
        "above_baseline": precision_at_1 > random_baseline,
        "correct": correct,
        "total": total,
    }
    if verbose:
        print(f"\nPrecision@1: {precision_at_1:.2f} (random baseline: {random_baseline:.2f})")
        print(f"Above baseline: {result_dict['above_baseline']}")
    return result_dict


if __name__ == "__main__":
    run_experiment(verbose=True)
