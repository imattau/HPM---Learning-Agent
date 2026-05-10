import torch
from typing import List, Dict, Any
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.evaluators.epistemic import EpistemicEvaluator
from hpm_ai_v6.hpm_model.evaluators.affective import AffectiveEvaluator
from hpm_ai_v6.hpm_model.evaluators.social import SocialEvaluator
from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule


class HPMLearner:
    """
    Orchestrates the HPM learning loop:
    1. Utility Evaluation (Epistemic, Affective, Social)
    2. Weight Updates (Meta Pattern Rule)
    3. Embedding Updates (Gradient Ascent on Utility)
    """

    def __init__(
        self,
        meta_rule: MetaPatternRule,
        beta_e: float = 1.0,
        beta_a: float = 0.5,
        beta_s: float = 0.5,
        embed_lr: float = 0.05,
    ):
        self.mpr = meta_rule
        self.beta_e = beta_e
        self.beta_a = beta_a
        self.beta_s = beta_s
        self.embed_lr = embed_lr

        self.epistemic = EpistemicEvaluator()
        self.affective = AffectiveEvaluator()
        self.social = SocialEvaluator()

    def compute_total_utility(self, cell: Cell, context: Dict[str, Any]) -> float:
        """Compatibility path used outside the batched update loop."""
        e_res = self.epistemic.evaluate(cell, context)
        a_res = self.affective.evaluate(cell, context)
        s_res = self.social.evaluate(cell, context)

        return (
            self.beta_e * e_res.score
            + self.beta_a * a_res.score
            + self.beta_s * s_res.score
        )

    @staticmethod
    def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor / (torch.norm(tensor, dim=1, keepdim=True) + 1e-9)

    @staticmethod
    def _normalize_vec(tensor: torch.Tensor) -> torch.Tensor:
        return tensor / (torch.norm(tensor) + 1e-9)

    def _build_batch_context(
        self,
        patterns: List[Cell],
        observation_seq: List[Cell],
        population: List[Cell],
        context: Dict[str, Any],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        device = torch.device("cpu")
        pop_index = {id(cell): idx for idx, cell in enumerate(population)}

        population_embeddings = torch.stack(
            [cell.as_tensor(dtype=dtype).to(device=device) for cell in population]
        )

        pattern_embeddings = torch.stack(
            [pattern.as_tensor(dtype=dtype).to(device=device) for pattern in patterns]
        ).detach().clone().requires_grad_(True)

        field_amps = torch.tensor(
            [context.get("field_amplifications", {}).get(pattern.name, 0.0) for pattern in patterns],
            dtype=dtype,
            device=device,
        )

        consensus_vec = context.get("consensus_vec")
        if consensus_vec is None:
            consensus_vec = torch.zeros_like(patterns[0].as_tensor(dtype=dtype))
        consensus = Cell._to_tensor(consensus_vec, dtype=dtype).to(device=device)

        source_embeddings = []
        source_indices = []
        for pattern in patterns:
            if pattern.source is None:
                source_embeddings.append(torch.zeros_like(pattern.as_tensor(dtype=dtype)))
                source_indices.append(-1)
            else:
                source_embeddings.append(pattern.source.as_tensor(dtype=dtype))
                source_indices.append(pop_index.get(id(pattern.source), -1))

        source_embeddings_t = torch.stack([embedding.to(device=device) for embedding in source_embeddings])
        source_indices_t = torch.tensor(source_indices, dtype=torch.long, device=device)

        if len(observation_seq) > 1:
            curr_indices = [pop_index.get(id(cell), -1) for cell in observation_seq[:-1]]
            next_indices = [pop_index.get(id(cell), -1) for cell in observation_seq[1:]]
            next_dims = [cell.dim for cell in observation_seq[1:]]
        else:
            curr_indices = []
            next_indices = []
            next_dims = []

        return {
            "population_embeddings": population_embeddings,
            "pattern_embeddings": pattern_embeddings,
            "field_amps": field_amps,
            "consensus": consensus,
            "source_embeddings": source_embeddings_t,
            "source_indices": source_indices_t,
            "curr_indices": torch.tensor(curr_indices, dtype=torch.long, device=device),
            "next_indices": torch.tensor(next_indices, dtype=torch.long, device=device),
            "next_dims": torch.tensor(next_dims, dtype=torch.long, device=device),
        }

    def _batched_utilities(
        self,
        patterns: List[Cell],
        batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        pattern_embeddings = batch["pattern_embeddings"]
        population_embeddings = batch["population_embeddings"]
        field_amps = batch["field_amps"]
        consensus = batch["consensus"]
        source_embeddings = batch["source_embeddings"]
        source_indices = batch["source_indices"]
        curr_indices = batch["curr_indices"]
        next_indices = batch["next_indices"]
        next_dims = batch["next_dims"]

        pattern_dims = torch.tensor(
            [pattern.dim for pattern in patterns],
            dtype=torch.long,
            device=pattern_embeddings.device,
        )

        base_embeddings = pattern_embeddings.clone()
        dim1_mask = pattern_dims == 1
        if torch.any(dim1_mask):
            base_embeddings[dim1_mask] = source_embeddings[dim1_mask] + pattern_embeddings[dim1_mask]

        base_embeddings = self._normalize_rows(base_embeddings)
        normalized_population = self._normalize_rows(population_embeddings)
        logits = torch.matmul(base_embeddings, normalized_population.T)
        probs = torch.softmax(logits / temperature, dim=1)

        if len(population_embeddings) > 0:
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            max_entropy = torch.log(
                torch.tensor(float(len(population_embeddings)), dtype=pattern_embeddings.dtype)
            )
            target_entropy = self.affective.target_entropy_ratio * max_entropy
            affective_scores = -torch.abs(entropy - target_entropy)
        else:
            affective_scores = torch.zeros(len(patterns), dtype=pattern_embeddings.dtype)

        consensus = self._normalize_vec(consensus)
        social_scores = torch.matmul(base_embeddings, consensus) + field_amps

        epistemic_scores = torch.zeros(len(patterns), dtype=pattern_embeddings.dtype)
        if len(curr_indices) > 0 and len(next_indices) > 0:
            valid_next = next_indices >= 0
            next_indices_safe = next_indices.clamp(min=0)

            gathered = torch.gather(
                probs,
                1,
                next_indices_safe.unsqueeze(0).expand(len(patterns), -1),
            )
            penalty = torch.full_like(gathered, 5.0)
            nll_terms = torch.where(valid_next.unsqueeze(0), -torch.log(gathered + 1e-9), penalty)

            valid_mask = torch.zeros_like(nll_terms, dtype=torch.bool)

            if torch.any(dim1_mask):
                curr_matrix = curr_indices.unsqueeze(0).expand(len(patterns), -1)
                source_matrix = source_indices.unsqueeze(1).expand(-1, len(curr_indices))
                valid_mask = valid_mask | (dim1_mask.unsqueeze(1) & (curr_matrix == source_matrix) & valid_next.unsqueeze(0))

            higher_mask = pattern_dims > 1
            if torch.any(higher_mask):
                target_dims = (pattern_dims - 1).unsqueeze(1)
                valid_mask = valid_mask | (higher_mask.unsqueeze(1) & (next_dims.unsqueeze(0) == target_dims) & valid_next.unsqueeze(0))

            counts = valid_mask.sum(dim=1)
            safe_counts = torch.clamp(counts, min=1)
            avg_nll = torch.where(
                counts > 0,
                (nll_terms * valid_mask).sum(dim=1) / safe_counts,
                torch.full((len(patterns),), 2.0, dtype=pattern_embeddings.dtype),
            )
            epistemic_scores = -avg_nll

        return (
            self.beta_e * epistemic_scores
            + self.beta_a * affective_scores
            + self.beta_s * social_scores
        )

    def _update_pattern_group(
        self,
        patterns: List[Cell],
        observation_seq: List[Cell],
        population: List[Cell],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        if not patterns:
            return torch.zeros(0, dtype=torch.float32)

        if not population:
            return torch.zeros(len(patterns), dtype=torch.float32)

        dtype = torch.float32
        temperature = float(context.get("temperature", 1.0))
        batch = self._build_batch_context(patterns, observation_seq, population, context, dtype=dtype)
        utilities = self._batched_utilities(patterns, batch, temperature=temperature)
        utilities.sum().backward()

        updated_embeddings = batch["pattern_embeddings"].detach() + self.embed_lr * batch["pattern_embeddings"].grad.detach()
        updated_embeddings = updated_embeddings / (torch.norm(updated_embeddings, dim=1, keepdim=True) + 1e-9)
        for pattern, embedding in zip(patterns, updated_embeddings):
            pattern.embedding = embedding.clone().detach()

        return utilities.detach()

    def step(self, observation_seq: List[Cell], population: List[Cell], context: Dict[str, Any]):
        """
        Executes one learning step.
        context should contain: consensus_vec, field_amplifications (dict)
        """
        patterns = self.mpr.patterns
        if not patterns:
            return torch.zeros(0, dtype=torch.float32).cpu().numpy()

        grouped_indices: Dict[tuple, List[int]] = {}
        for idx, pattern in enumerate(patterns):
            key = (pattern.dim, len(pattern.embedding))
            grouped_indices.setdefault(key, []).append(idx)

        scores = torch.zeros(len(patterns), dtype=torch.float32)
        for indices in grouped_indices.values():
            group_patterns = [patterns[idx] for idx in indices]
            group_scores = self._update_pattern_group(
                group_patterns,
                observation_seq,
                population,
                context,
            )
            scores[indices] = group_scores

        field_amps = torch.tensor(
            [context.get("field_amplifications", {}).get(pattern.name, 0.0) for pattern in patterns]
        , dtype=torch.float32)
        self.mpr.update_weights_tensor(scores, field_amps)
        return scores.detach().cpu().numpy()
