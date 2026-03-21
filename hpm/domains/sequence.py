import numpy as np


class SequenceDomain:
    """
    Markov-chain symbol sequence domain with one-hot observations.

    Deep structure  = _transition (row-stochastic V×V matrix).
    Surface structure = _label_map (permutation of range(V)).

    Observations: one-hot vector of dim vocab_size.
    Labels (for transfer_probe): internal symbol index (pre label_map).

    RNG design — three independent streams from SeedSequence:
      _rng        : observation RNG — used only by observe()
      _perturb_rng: perturbation RNG — used by deep_perturb(), surface_perturb(), transfer_probe()
      _init_rng   : initialisation RNG — used only to generate _transition at construction
    This ensures deep_perturb()/surface_perturb()/transfer_probe() never advance _rng.
    """

    def __init__(
        self,
        vocab_size: int = 8,
        order: int = 1,
        seed=None,
        transition=None,
        label_map=None,
    ):
        if order != 1:
            raise ValueError("only order=1 is supported")
        self._vocab_size = vocab_size
        ss = np.random.SeedSequence(seed)
        obs_seed, perturb_seed, init_seed = ss.spawn(3)
        self._rng = np.random.default_rng(obs_seed)           # observe() only
        self._perturb_rng = np.random.default_rng(perturb_seed)  # perturb/probe only
        self._transition = (
            transition
            if transition is not None
            else self._random_transition(np.random.default_rng(init_seed), vocab_size)
        )
        self._label_map = (
            label_map
            if label_map is not None
            else np.arange(vocab_size)
        )
        self._current = 0

    # ------------------------------------------------------------------
    # Domain protocol
    # ------------------------------------------------------------------

    def observe(self) -> np.ndarray:
        """Sample next symbol, apply label map, return one-hot."""
        next_sym = int(self._rng.choice(self._vocab_size, p=self._transition[self._current]))
        self._current = next_sym
        x = np.zeros(self._vocab_size)
        x[self._label_map[next_sym]] = 1.0
        return x

    def feature_dim(self) -> int:
        return self._vocab_size

    def deep_perturb(self) -> 'SequenceDomain':
        """Return new domain with randomised transition matrix.
        Uses _perturb_rng so self._rng (observation RNG) is not mutated."""
        new_transition = self._random_transition(self._perturb_rng, self._vocab_size)
        return SequenceDomain(
            self._vocab_size, seed=None,
            transition=new_transition,
            label_map=self._label_map.copy(),
        )

    def surface_perturb(self) -> 'SequenceDomain':
        """Return new domain with different label_map.
        Uses _perturb_rng so self._rng (observation RNG) is not mutated.
        Guaranteed to differ from self._label_map."""
        new_label_map = self._perturb_rng.permutation(self._vocab_size)
        while np.array_equal(new_label_map, self._label_map):
            new_label_map = self._perturb_rng.permutation(self._vocab_size)
        return SequenceDomain(
            self._vocab_size, seed=None,
            transition=self._transition.copy(),
            label_map=new_label_map,
        )

    def transfer_probe(self, near: bool) -> list:
        """
        Generate 200 labelled (observation, internal_symbol) pairs without
        mutating self._current or self._rng.

        near=True:  probe label_map = self._label_map.copy() (same surface as training)
        near=False: probe label_map = reshuffled permutation != self._label_map (far transfer)

        Label = internal symbol index (surface-invariant ground truth).
        """
        if near:
            probe_label_map = self._label_map.copy()
        else:
            probe_label_map = self._perturb_rng.permutation(self._vocab_size)
            while np.array_equal(probe_label_map, self._label_map):
                probe_label_map = self._perturb_rng.permutation(self._vocab_size)

        # Sample 200 steps using a fresh chain starting at internal state 0
        current = 0
        results = []
        for _ in range(200):
            next_sym = int(self._perturb_rng.choice(self._vocab_size, p=self._transition[current]))
            current = next_sym
            x = np.zeros(self._vocab_size)
            x[probe_label_map[next_sym]] = 1.0
            results.append((x, next_sym))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_transition(rng, vocab_size: int, alpha: float = 1.0) -> np.ndarray:
        """Sample a random row-stochastic V×V matrix (Dirichlet rows)."""
        return rng.dirichlet(np.full(vocab_size, alpha), size=vocab_size)
