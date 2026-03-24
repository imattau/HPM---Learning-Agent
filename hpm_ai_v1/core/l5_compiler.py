def __init__(self, config: AgentConfig, baseline_cost: float, baseline_node_count: int, **kwargs):
    super().__init__(config, **kwargs)
    self.best_cost = baseline_cost
    self.best_node_count = baseline_node_count
    self.meta_monitor = L5MetaMonitor()
    self.generation = 1
    self.stagnation_counter = 0
    self.allow_bloat = False