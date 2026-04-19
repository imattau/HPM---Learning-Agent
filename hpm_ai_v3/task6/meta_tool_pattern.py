import torch, torch.nn as nn, pyro, pyro.distributions as dist
from pattern import HPMPattern
class MetaToolPattern(HPMPattern):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.baseline = nn.Linear(input_dim, 1)
        self.optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.baseline.parameters()), lr=0.001)
    def sample(self, ctx):
        logit = self.net(ctx["features"])
        return {"use_tool": torch.bernoulli(torch.sigmoid(logit)).bool()}
    def update_parameters(self, obs, lr=0.01):
        logit = self.net(obs["features"])
        log_prob = torch.distributions.Bernoulli(logits=logit).log_prob(obs["use_tool_taken"].float())
        bl = self.baseline(obs["features"])
        adv = obs["reward"] - bl.detach()
        loss = -(log_prob * adv).mean() + nn.MSELoss()(bl, obs["reward"])
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def log_prob(self, obs): return torch.tensor(0.0)
    def structural_distance(self, other): return 0.0
    def extract_causal_graph(self): return nx.DiGraph()
    def intervene(self, intervention, context): return self.sample(context)
