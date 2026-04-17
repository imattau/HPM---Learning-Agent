"""SP Experiment: Curiosity-driven vs passive reading — efficiency comparison."""
from __future__ import annotations
from typing import Dict
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

SEED_PASSAGES = [
    "Machine learning trains models on data to make predictions.",
    "Deep learning is a machine learning technique using neural networks.",
]

STREAM = [
    "Machine learning trains models on data.",
    "The Roman Empire controlled most of Europe for centuries.",
    "Neural networks can approximate any continuous function.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Data science combines statistics and machine learning methods.",
    "Ancient Romans built aqueducts to transport water across distances.",
    "Amazon rainforest biodiversity includes millions of species.",
    "Deep learning architectures include CNNs and transformers.",
]

QUERIES = [
    ("roman empire ancient history", "The Roman Empire controlled most of Europe for centuries."),
    ("photosynthesis plants energy", "Photosynthesis converts sunlight into chemical energy in plants."),
    ("amazon rainforest species biodiversity", "Amazon rainforest biodiversity includes millions of species."),
]


def _build_agent(extra_passages):
    all_p = SEED_PASSAGES + extra_passages
    config = TextDomainConfig.from_passages(all_p, max_vocab=80)
    agent = ReaderAgent(config)
    for p in SEED_PASSAGES:
        agent.observe_passage(p)
    return agent


def run_experiment(verbose: bool = True, curiosity_threshold: float = 0.7) -> Dict:
    # Passive: read everything
    passive_agent = _build_agent(STREAM)
    for p in STREAM:
        passive_agent.observe_passage(p)
    passive_read = len(STREAM)

    # Curiosity: only read novel passages
    curiosity_agent = _build_agent(STREAM)
    curiosity_read = 0
    for p in STREAM:
        if curiosity_agent.observe_if_curious(p, threshold=curiosity_threshold):
            curiosity_read += 1

    def precision(agent):
        correct = 0
        for query, expected in QUERIES:
            result = agent.query(query)
            if result == expected:
                correct += 1
        return correct / len(QUERIES)

    passive_p = precision(passive_agent)
    curiosity_p = precision(curiosity_agent)

    result = {
        "passive_precision": passive_p,
        "curiosity_precision": curiosity_p,
        "passages_read_passive": passive_read,
        "passages_read_curiosity": curiosity_read,
        "efficiency_gain": passive_read - curiosity_read,
    }
    if verbose:
        print(f"Passive:   precision={passive_p:.2f}, passages read={passive_read}")
        print(f"Curiosity: precision={curiosity_p:.2f}, passages read={curiosity_read}")
        print(f"Efficiency gain: {result['efficiency_gain']} fewer passages read")
    return result


if __name__ == "__main__":
    run_experiment(verbose=True)
