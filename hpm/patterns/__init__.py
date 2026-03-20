from .gaussian import GaussianPattern
from .classifier import HPMLevelClassifier

PATTERN_REGISTRY: dict[str, type] = {
    'gaussian': GaussianPattern,
}


def pattern_from_dict(d: dict):
    """Deserialise a pattern dict using the type registry."""
    cls = PATTERN_REGISTRY.get(d['type'])
    if cls is None:
        raise ValueError(f"Unknown pattern type: {d['type']}")
    return cls.from_dict(d)
