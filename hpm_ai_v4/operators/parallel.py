# hpm_ai_v4/operators/parallel.py
"""
Pure-function worker and pool wrapper for parallel pattern evaluation.
Uses stdlib multiprocessing only — no Ray, no extra dependencies.
"""
import numpy as np
from multiprocessing import Pool
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Helpers: pattern state serialisation
# ---------------------------------------------------------------------------

def pattern_to_dict(pattern) -> dict:
    """Serialise a HierarchicalPattern or FlatPattern to a plain dict."""
    d = {
        'pattern_id': pattern.id,
        'complexity': pattern.complexity,
        'latent_dim': pattern.latent_dim,
        'obs_dim': pattern.obs_dim,
        'B':    pattern.B.copy(),
        'running_loss': float(pattern.running_loss),
        'weight': float(pattern.weight),
    }
    if pattern.complexity >= 2 or pattern.latent_dim > 1:
        d['A'] = pattern.A.copy()
        d['pi'] = pattern.pi.copy()
    return d


def dict_to_pattern(d: dict):
    """Reconstruct a minimal pattern object from a state dict."""
    from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
    if d['complexity'] == 1 and d['latent_dim'] == 1:
        p = FlatPattern(d['pattern_id'], obs_dim=d['obs_dim'])
        p.B = d['B']
    else:
        p = HierarchicalPattern(d['pattern_id'],
                                latent_dim=d['latent_dim'],
                                obs_dim=d['obs_dim'])
        p.A = d['A']
        p.B = d['B']
        p.pi = d['pi']
        p._refresh_log_cache()
        
    p.running_loss = d['running_loss']
    p.weight = d['weight']
    return p


# ---------------------------------------------------------------------------
# Pure worker function
# ---------------------------------------------------------------------------

def pattern_worker(state_dict: dict, obs_buffer: list,
                   field_freq: dict, params: dict) -> dict:
    """
    Pure function: reconstruct pattern, run observe/adapt/update_running_loss,
    compute scores, return updated state dict + scores.
    """
    from hpm_ai_v4.evaluators.metrics import (
        epistemic_score, affective_score, social_score, total_score,
    )

    p = dict_to_pattern(state_dict)
    result = update_pattern_resident(p, obs_buffer, field_freq, params)
    state = pattern_to_dict(p)
    state.update(result)
    return state


def _worker_context_window(obs_buffer: list, params: dict) -> list:
    """Keep only the tail of the observation buffer needed by the worker path."""
    window = max(30, int(params.get('adapt_window', 100)))
    if not obs_buffer:
        return []
    return list(obs_buffer[-window:])


def update_pattern_resident(pattern, obs_buffer: list,
                            field_freq: dict, params: dict) -> dict:
    """Update a resident pattern object in place and return its scores."""
    from hpm_ai_v4.evaluators.metrics import (
        epistemic_score, affective_score, social_score, total_score,
    )

    ll = None
    if obs_buffer:
        if params.get('do_param_update', True):
            if pattern.complexity >= 2 or pattern.latent_dim > 1:
                ll = pattern.update_parameters_online(obs_buffer, window_size=params.get('adapt_window', 100))
            else:
                if hasattr(pattern, "observe"):
                    pattern.observe(obs_buffer[-1], learning_rate=params['learning_rate'])
                else:
                    ll = pattern.log_likelihood(obs_buffer[-30:])
        if ll is None:
            # Recompute LL every 10 steps; use cached value otherwise
            step = params.get('step_counter', 0)
            cached_ll = getattr(pattern, '_cached_ll', None)
            if cached_ll is None or step % 10 == 0:
                ll = pattern.log_likelihood(obs_buffer[-30:])
                pattern._cached_ll = ll
            else:
                ll = cached_ll
        pattern.update_running_loss(obs_buffer, ll=ll)

    ep = epistemic_score(pattern)
    aff = affective_score(pattern, obs_buffer)
    soc = social_score(pattern, field_freq)
    tot = total_score(
        pattern, obs_buffer, field_freq,
        beta_aff=params['beta_aff'],
        gamma_soc=params['gamma_soc'],
        external_soc=params.get('external_soc', 0.5),
    )

    return {
        'ep_score': float(ep),
        'aff_score': float(aff),
        'soc_score': float(soc),
        'total_score': float(tot),
    }


# ---------------------------------------------------------------------------
# Pool wrapper
# ---------------------------------------------------------------------------

class ParallelPatternPool:
    """
    Wraps multiprocessing.Pool for parallel per-pattern updates.
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self._pool = Pool(processes=num_workers) if num_workers > 1 else None

    def map_patterns(self, patterns: list, obs_buffer: list,
                     field_freq: dict, params: dict) -> list:
        worker_obs = _worker_context_window(obs_buffer, params)
        task_args = [
            (pattern_to_dict(p), list(worker_obs), dict(field_freq),
             {**params, 'external_soc': params.get('external_soc_map', {}).get(p.id, 0.5)})
            for p in patterns
        ]
        if self._pool is None:
            return [pattern_worker(*args) for args in task_args]
        else:
            return self._pool.starmap(pattern_worker, task_args)

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.close()
