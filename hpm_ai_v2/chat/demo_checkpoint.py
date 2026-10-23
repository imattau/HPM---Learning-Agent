"""Write a small demo checkpoint (five sources, one tied pair) for the chat UI.

    python -m hpm_ai_v2.chat.demo_checkpoint OUT_DIR
    python -m hpm_ai_v2.chat.server OUT_DIR/checkpoint --data-dir OUT_DIR/data
"""
from __future__ import annotations

import sys
from pathlib import Path

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer

DEMO_SOURCES = (
    SourceDocument("solar", "Solar panels convert sunlight into electricity. Solar energy can reduce household emissions. Photovoltaic cells generate direct current.", "Solar energy"),
    SourceDocument("wind", "Wind turbines convert moving air into electricity. Wind farms are often built offshore. Turbine blades capture kinetic energy.", "Wind power"),
    SourceDocument("brain", "The brain communicates through electrochemical signals across neurons and synapses. Neurons release neurotransmitters at synapses.", "Neuroscience"),
    SourceDocument("notes-a", "Synaptic plasticity strengthens connections between neurons.", "Lecture notes"),
    SourceDocument("notes-b", "Synaptic plasticity strengthens connections between neurons.", "Textbook"),
)


def build(out_dir: Path) -> Path:
    config = TextDomainConfig.from_passages([s.text for s in DEMO_SOURCES], max_vocab=200)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(out_dir / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(DEMO_SOURCES)
    trainer.save_checkpoint(out_dir / "checkpoint")
    return out_dir / "checkpoint"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(build(Path(sys.argv[1])))
