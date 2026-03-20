class HPMLevelClassifier:
    """
    Assigns an HPM level (1–5) from structural metrics and pattern density.

    Classification is evaluated top-down (Level 5 checked first; Level 1 is
    the default fallback). All comparisons use strict > — a pattern exactly on
    a threshold boundary stays at the lower level.
    """

    def __init__(
        self,
        l5_density: float = 0.85,
        l5_conn: float = 0.80,
        l5_comp: float = 0.70,
        l4_conn: float = 0.70,
        l4_comp: float = 0.60,
        l3_conn: float = 0.50,
        l3_comp: float = 0.40,
        l2_conn: float = 0.30,
    ):
        self.l5_density = l5_density
        self.l5_conn = l5_conn
        self.l5_comp = l5_comp
        self.l4_conn = l4_conn
        self.l4_comp = l4_comp
        self.l3_conn = l3_conn
        self.l3_comp = l3_comp
        self.l2_conn = l2_conn

    def compute_level(self, pattern, density: float) -> int:
        """Return HPM level (1–5) for the given pattern and its current density."""
        conn = pattern.connectivity()
        comp = pattern.compress()
        if density > self.l5_density and conn > self.l5_conn and comp > self.l5_comp:
            return 5
        if conn > self.l4_conn and comp > self.l4_comp:
            return 4
        if conn > self.l3_conn and comp > self.l3_comp:
            return 3
        if conn > self.l2_conn:
            return 2
        return 1
