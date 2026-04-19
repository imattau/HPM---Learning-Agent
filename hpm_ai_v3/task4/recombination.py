import torch
import networkx as nx
from composable_pattern import ComposableRegressionPattern

class RecombinationOperator:
    def __init__(self, insight_boost_weight: float = 0.5):
        self.insight_boost_weight = insight_boost_weight
        
    def recombine(self, p1: ComposableRegressionPattern, p2: ComposableRegressionPattern) -> ComposableRegressionPattern:
        new_p = ComposableRegressionPattern(p1.input_dim, p1.output_dim, p1.z1_dim, p1.z2_dim)
        with torch.no_grad():
            new_p.decoder_part1.load_state_dict(p2.decoder_part1.state_dict())
            new_p.decoder_part2.load_state_dict(p1.decoder_part2.state_dict())
            new_p.z2_loc.data = p1.z2_loc.data.clone()
        new_p.causal_graph = nx.compose(p1.causal_graph, p2.causal_graph)
        return new_p
    
    def compute_insight(self, new_p, p1, p2, acc) -> float:
        return self.insight_boost_weight * max(0.0, acc)
