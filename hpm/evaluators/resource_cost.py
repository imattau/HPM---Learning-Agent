# hpm/evaluators/resource_cost.py


class ResourceCostEvaluator:
    """
    Resource cost evaluator (HPM energy-constraint principle).

    Penalises complex patterns when the system is under memory/CPU pressure.

    E_cost_i(t) = -lambda_cost * description_length(pattern_i) * pressure(t)

    pressure(t) = w_mem * (mem_percent / 100) + w_cpu * (cpu_percent / 100)

    pressure(t) is in [0, 1]: 0 when system is idle, 1 when fully loaded.
    E_cost_i is always <= 0 — it subtracts from Total_i for complex patterns.

    psutil is imported lazily on first call to pressure(), so this class can
    be instantiated without psutil installed (as long as pressure() is not called).
    Agents with delta_cost=0.0 never call pressure(), so they never need psutil.

    Args:
        lambda_cost: penalty scale (default 1.0)
        w_mem: weight for memory pressure in [0, 1] (default 0.5)
        w_cpu: weight for CPU pressure in [0, 1] (default 0.5)
    """

    def __init__(
        self,
        lambda_cost: float = 1.0,
        w_mem: float = 0.5,
        w_cpu: float = 0.5,
    ):
        self.lambda_cost = lambda_cost
        self.w_mem = w_mem
        self.w_cpu = w_cpu
        self._psutil = None  # lazily populated by _get_psutil()

    def _get_psutil(self):
        """Return cached psutil module, importing it on first call."""
        if self._psutil is None:
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                raise ImportError(
                    "psutil is required for ResourceCostEvaluator. "
                    "Install it with: pip install psutil"
                )
        return self._psutil

    def pressure(self) -> float:
        """
        Current system resource pressure in [0, 1].
        Reads real psutil metrics (or injected mock for testing).
        """
        psutil = self._get_psutil()
        mem = psutil.virtual_memory().percent / 100.0
        cpu_val = psutil.cpu_percent()
        cpu = (cpu_val if cpu_val is not None else 0.0) / 100.0
        return self.w_mem * mem + self.w_cpu * cpu

    def evaluate(self, pattern) -> float:
        """Return E_cost_i = -lambda_cost * description_length * pressure. Always <= 0."""
        return -self.lambda_cost * pattern.description_length() * self.pressure()
