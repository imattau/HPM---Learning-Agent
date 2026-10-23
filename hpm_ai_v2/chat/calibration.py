"""Abstention calibration for a pretrained checkpoint.

A pretrained checkpoint has no validation episodes, so its abstention threshold is 0.0 and only a question with
no word overlap abstains. The study trainer already knows how to calibrate that threshold from validation
episodes (``SupervisedStudyTrainer.fit``); this module runs that step against a pretrained checkpoint and keeps
the one number it produces in a small file bound to the checkpoint, instead of writing a second copy of a
checkpoint that may be large. Loading the file sets the trainer's own ``abstain_threshold``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


class CalibrationMismatch(RuntimeError):
    """A calibration file was fitted on a different checkpoint."""


@dataclass(frozen=True)
class Calibration:
    abstain_threshold: float
    checkpoint_digest: Optional[str] = None
    fitted_on: dict = field(default_factory=dict)

    def apply(self, trainer) -> None:
        trainer.abstain_threshold = float(self.abstain_threshold)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, *, checkpoint_digest: Optional[str] = None) -> "Calibration":
        loaded = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if checkpoint_digest is not None and loaded.checkpoint_digest not in (None, checkpoint_digest):
            raise CalibrationMismatch(
                "this calibration was fitted on a different checkpoint "
                f"({loaded.checkpoint_digest!r} vs {checkpoint_digest!r}); fit it again"
            )
        return loaded
