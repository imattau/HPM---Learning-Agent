"""ATIS dataset loader and IntentLabelAdapter for HPM v5."""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, field
from .base import Adapter
from .packet import AdapterPacket


def load_atis() -> tuple[list[dict], list[dict]]:
    """Return (train, test) as lists of {text, intent} dicts from local CSVs."""
    
    def _read_csv(file_path: str) -> list[dict]:
        data = []
        if not os.path.exists(file_path):
            return data
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"text": row["text"], "intent": row["intent"]})
        return data

    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train = _read_csv(os.path.join(_repo, "data", "atis", "atis_train.csv"))
    test = _read_csv(os.path.join(_repo, "data", "atis", "atis_test.csv"))
    
    return train, test


@dataclass(slots=True)
class IntentLabelAdapter(Adapter):
    """Inject gold intent label during training; omit during inference."""

    name: str = "intent_label"
    label: str | None = None
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["intent_label"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if self.label is not None:
            packet.context["intent_label"] = self.label
        return packet
