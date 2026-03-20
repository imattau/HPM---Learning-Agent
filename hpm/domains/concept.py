from dataclasses import dataclass, field
import numpy as np


@dataclass
class Concept:
    deep_features: np.ndarray      # structural invariants
    surface_templates: list        # list of np.ndarray surface prototypes
    label: int = 0


class ConceptLearningDomain:
    """
    Concept learning domain for HPM Phase 1.

    Observations x_t = [surface_features || deep_features].
    Surface features vary across instances; deep features are invariant per concept.
    Supports §9.1 (deep vs surface perturbation) and §9.2 (near/far transfer probes).
    """

    def __init__(
        self,
        concepts: list[Concept],
        noise: float = 0.1,
        seed: int | None = None,
    ):
        self.concepts = concepts
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._t = 0

    def observe(self) -> np.ndarray:
        concept = self.concepts[self._t % len(self.concepts)]
        self._t += 1
        surface = self.rng.choice(len(concept.surface_templates))
        s = concept.surface_templates[surface] + self.rng.normal(
            0, self.noise, concept.surface_templates[surface].shape
        )
        return np.concatenate([s, concept.deep_features])

    def feature_dim(self) -> int:
        c = self.concepts[0]
        return len(c.surface_templates[0]) + len(c.deep_features)

    def deep_perturb(self) -> 'ConceptLearningDomain':
        """Return copy with structurally altered deep features (§9.1)."""
        new_concepts = [
            Concept(
                deep_features=c.deep_features + self.rng.normal(0, 0.5, c.deep_features.shape),
                surface_templates=c.surface_templates,
                label=c.label,
            )
            for c in self.concepts
        ]
        return ConceptLearningDomain(new_concepts, self.noise)

    def surface_perturb(self) -> 'ConceptLearningDomain':
        """Return copy with altered surface templates but preserved deep structure (§9.1)."""
        new_concepts = [
            Concept(
                deep_features=c.deep_features.copy(),
                surface_templates=[
                    s + self.rng.normal(0, 0.5, s.shape) for s in c.surface_templates
                ],
                label=c.label,
            )
            for c in self.concepts
        ]
        return ConceptLearningDomain(new_concepts, self.noise)

    def transfer_probe(self, near: bool) -> list[tuple[np.ndarray, int]]:
        """
        Generate labelled test observations.
        near=True: familiar surface + correct deep features
        near=False (far): novel random surface + correct deep features
        """
        probes = []
        n_per_concept = 10
        for concept in self.concepts:
            for _ in range(n_per_concept):
                if near:
                    surface = self.rng.choice(len(concept.surface_templates))
                    s = concept.surface_templates[surface] + self.rng.normal(
                        0, self.noise * 0.5, concept.surface_templates[surface].shape
                    )
                else:
                    s = self.rng.normal(0, 2.0, concept.surface_templates[0].shape)
                x = np.concatenate([s, concept.deep_features])
                probes.append((x, int(concept.label)))
        return probes
