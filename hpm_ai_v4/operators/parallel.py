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
    return {
        'pattern_id': pattern.id,
        'complexity': pattern.complexity,
        'latent_dim': pattern.latent_dim,
        'obs_dim': pattern.obs_dim,
        'A3':   pattern.A3.copy(),
        'A32':  pattern.A32.copy(),
        'A21':  pattern.A21.copy(),
        'B':    pattern.B.copy(),
        'pi3':  pattern.pi3.copy(),
        'SS_A3':  pattern.SS_A3.copy(),
        'SS_A32': pattern.SS_A32.copy(),
        'SS_A21': pattern.SS_A21.copy(),
        'SS_B':   pattern.SS_B.copy(),
        'running_loss': float(pattern.running_loss),
        'weight': float(pattern.weight),
        'statistics_decay': float(pattern.statistics_decay),
    }


def dict_to_pattern(d: dict):
    """Reconstruct a minimal pattern object from a state dict."""
    from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
    if d['complexity'] == 1:
        p = FlatPattern(d['pattern_id'], obs_dim=d['obs_dim'])
    else:
        p = HierarchicalPattern(d['pattern_id'],
                                latent_dim=d['latent_dim'],
                                obs_dim=d['obs_dim'])
    p.A3   = d['A3']
    p.A32  = d['A32']
    p.A21  = d['A21']
    p.B    = d['B']
    p.pi3  = d['pi3']
    p.SS_A3  = d['SS_A3']
    p.SS_A32 = d['SS_A32']
    p.SS_A21 = d['SS_A21']
    p.SS_B   = d['SS_B']
    p.running_loss = d['running_loss']
    p.weight = d['weight']
    p.statistics_decay = d['statistics_decay']
    return p


# ---------------------------------------------------------------------------
# Pure worker function
# ---------------------------------------------------------------------------

def pattern_worker(state_dict: dict, obs_buffer: list,
                   field_freq: dict, params: dict) -> dict:
    """
    Pure function: reconstruct pattern, run observe/adapt/update_running_loss,
    compute scores, return updated state dict + scores.

    No shared state.  Safe to call from multiprocessing.Pool workers.
    """
    from hpm_ai_v4.evaluators.metrics import (
        epistemic_score, affective_score, social_score, total_score,
    )

    p = dict_to_pattern(state_dict)

    # --- per-step update (mirrors HPMAgent.perceive_and_learn inner loop) ---
    if obs_buffer:
        obs = obs_buffer[-1]
        p.observe(obs, learning_rate=params['learning_rate'])

        adapt_window = params.get('adapt_window', 20)
        if p.complexity >= 2 and len(obs_buffer) >= 10:
            p.adapt(obs_buffer[-adapt_window:])

        p.update_running_loss(obs_buffer, lambda_l=params['lambda_l'])

    # --- evaluator scores ---
    ep  = epistemic_score(p)
    aff = affective_score(p, obs_buffer)
    soc = social_score(p, field_freq)
    tot = total_score(
        p, obs_buffer, field_freq,
        beta_aff=params['beta_aff'],
        gamma_soc=params['gamma_soc'],
        external_soc=params.get('external_soc', 0.5),
    )

    # --- return updated state + scores ---
    result = pattern_to_dict(p)
    result['ep_score']    = float(ep)
    result['aff_score']   = float(aff)
    result['soc_score']   = float(soc)
    result['total_score'] = float(tot)
    return result


# ---------------------------------------------------------------------------
# Pool wrapper
# ---------------------------------------------------------------------------

class ParallelPatternPool:
    """
    Wraps multiprocessing.Pool for parallel per-pattern updates.

    num_workers=1  -> sequential fallback (no pool spawned, no IPC overhead).
    num_workers>1  -> pool created once at construction, reused each step.
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self._pool = Pool(processes=num_workers) if num_workers > 1 else None

    def map_patterns(self, patterns: list, obs_buffer: list,
                     field_freq: dict, params: dict) -> list:
        """
        Run pattern_worker for each pattern.  Returns list of result dicts
        in the same order as `patterns`.
        """
        task_args = [
            (pattern_to_dict(p), list(obs_buffer), dict(field_freq),
             {**params, 'external_soc': params.get('external_soc_map', {}).get(p.id, 0.5)})
            for p in patterns
        ]
        if self._pool is None:
            # Sequential fallback
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
