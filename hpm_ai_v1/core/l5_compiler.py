def __init__(self, baseline_cost: float, baseline_node_count: int, baseline_surprise: float=0.0):
    self.best_cost = baseline_cost
    self.best_node_count = baseline_node_count
    self.monitor = L5MetaMonitor()
    self.resource_eval = ResourceCostEvaluator(lambda_cost=1.0)
    self.generation = 1
    self.stagnation_counter = 0
    self.allow_bloat = False