import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from typing import Dict, Any, Optional
from task1.regression_pattern import RegressionPattern

class ComposableRegressionPattern(RegressionPattern):
    def __init__(self, input_dim: int, output_dim: int = 1, z1_dim: int = 16, z2_dim: int = 8, pattern_id: Optional[str] = None):
        super().__init__(input_dim, output_dim, z1_dim, z2_dim, pattern_id)
        self.decoder_part1 = nn.Sequential(nn.Linear(z1_dim, z1_dim), nn.ReLU(), nn.Linear(z1_dim, z1_dim))
        self.decoder_part2 = nn.Linear(z1_dim, output_dim * 2)
        del self.fc_z1_to_y

    def parameters(self):
        return (list(self.fc_x_to_z1.parameters()) + list(self.fc_z2_to_z1.parameters()) +
                list(self.decoder_part1.parameters()) + list(self.decoder_part2.parameters()) +
                list(self.fc_z1_to_z2.parameters()) + [self.z2_loc, self.z2_scale])

    def forward_decoder(self, z1):
        return self.decoder_part2(self.decoder_part1(z1))

    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        x = context["input"].unsqueeze(0) if context["input"].dim() == 1 else context["input"]
        with torch.no_grad():
            guide_trace = poutine.trace(self.guide).get_trace({"input": x})
            z1 = guide_trace.nodes["z1"]["value"]
            y_p = self.forward_decoder(z1).chunk(2, dim=-1)
            y = dist.Normal(y_p[0], torch.exp(0.5 * y_p[1])).sample((num_samples,))
        return {"y": y.squeeze(0)}

    def model(self, observations: Optional[Dict[str, torch.Tensor]] = None):
        batch_size = observations["input"].shape[0] if observations else 1
        with pyro.plate("batch", batch_size):
            z2 = pyro.sample("z2", dist.Normal(self.z2_loc, torch.exp(self.z2_scale)).to_event(1))
            pyro.factor("z2_sparsity", -self.sparsity_lambda * z2.abs().sum())
            z1_p = self.fc_z2_to_z1(z2).chunk(2, dim=-1)
            z1 = pyro.sample("z1", dist.Normal(z1_p[0], torch.exp(0.5 * z1_p[1])).to_event(1))
            y_p = self.forward_decoder(z1).chunk(2, dim=-1)
            y = pyro.sample("y", dist.Normal(y_p[0], torch.exp(0.5 * y_p[1])).to_event(1),
                            obs=observations["target"] if observations else None)
        return y
