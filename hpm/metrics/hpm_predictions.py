import numpy as np
from copy import deepcopy


def _run_accuracy(agent, domain, n_steps: int) -> float:
    """Run a deepcopy of agent on domain for n_steps, return mean accuracy.
    Uses deepcopy to avoid mutating the caller's agent state.
    """
    agent_copy = deepcopy(agent)
    domain_copy = deepcopy(domain)
    results = []
    for _ in range(n_steps):
        x = domain_copy.observe()
        result = agent_copy.step(x)
        results.append(result['mean_accuracy'])
    return float(np.mean(results))


def sensitivity_ratio(agent, domain, n_steps: int = 100) -> float:
    """
    §9.1: ratio of accuracy change under deep vs surface perturbation.
    HPM predicts ratio > 1 (agents more sensitive to deep structure changes).

    Returns: deep_drop / surface_drop
    """
    baseline = _run_accuracy(agent, domain, n_steps)
    deep_acc = _run_accuracy(agent, domain.deep_perturb(), n_steps)
    surface_acc = _run_accuracy(agent, domain.surface_perturb(), n_steps)

    deep_drop = baseline - deep_acc
    surface_drop = baseline - surface_acc

    if abs(surface_drop) < 1e-10:
        return float('inf')
    return float(deep_drop / surface_drop)


def curiosity_complexity_profile(
    agent,
    domains_by_complexity: dict[float, object],
    n_steps: int = 50,
) -> dict[float, float]:
    """
    §9.4: mean affective evaluator engagement per complexity level.
    HPM predicts inverted-U — peaks at intermediate complexity.

    Returns: {complexity_level: mean_e_aff}
    Note: approximates E_aff by tracking accuracy improvement rate per domain.
    """
    profile = {}
    for complexity, domain in domains_by_complexity.items():
        # Fresh agent copy per complexity level — avoids cross-contamination
        agent_copy = deepcopy(agent)
        domain_copy = deepcopy(domain)
        accuracies = []
        for _ in range(n_steps):
            x = domain_copy.observe()
            result = agent_copy.step(x)
            accuracies.append(result['mean_accuracy'])
        # E_aff proxy: mean absolute improvement across steps
        diffs = np.abs(np.diff(accuracies))
        profile[complexity] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    return profile


def social_field_convergence(quality_history: list[dict]) -> float:
    """
    §9.5: linear regression slope of field diversity over time.

    Negative = converging (social field pulling agents toward shared patterns).
    Positive = diverging. Near-zero = stable diversity.

    quality_history: list of dicts from PatternField.field_quality(), one per step.
    Returns: slope (diversity/step).
    Raises ValueError if fewer than 2 steps provided.
    """
    if len(quality_history) < 2:
        raise ValueError("social_field_convergence requires at least 2 steps of history")
    diversities = np.array([q["diversity"] for q in quality_history], dtype=float)
    t = np.arange(len(diversities), dtype=float)
    t_mean, d_mean = t.mean(), diversities.mean()
    slope = float(
        np.sum((t - t_mean) * (diversities - d_mean)) / np.sum((t - t_mean) ** 2)
    )
    return slope
