"""
common.py — Shared utilities for HPM benchmark scripts.

Provides:
  - BENCH_CONFIG: base AgentConfig kwargs for all benchmarks
  - make_agent(): construct an Agent with benchmark defaults
  - print_results_table(): print an ASCII table to stdout
"""

import sys
import os

# Allow running benchmarks from the repo root as: python benchmarks/xxx.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hpm.config import AgentConfig
from hpm.agents.agent import Agent

# ---------------------------------------------------------------------------
# Shared benchmark configuration defaults
# ---------------------------------------------------------------------------

BENCH_CONFIG = dict(
    feature_dim=16,
    beta_comp=0.1,
    T_recomb=50,
    recomb_cooldown=25,
    kappa_0=0.5,
    init_sigma=2.0,
    conflict_threshold=0.1,
)


def make_agent(feature_dim: int = 16, agent_id: str = "bench", **overrides) -> Agent:
    """
    Construct an Agent with benchmark defaults.
    Pass keyword overrides to customise any AgentConfig field.
    """
    cfg_kwargs = dict(BENCH_CONFIG)
    cfg_kwargs["feature_dim"] = feature_dim
    cfg_kwargs["agent_id"] = agent_id
    cfg_kwargs.update(overrides)
    config = AgentConfig(**cfg_kwargs)
    return Agent(config)


# ---------------------------------------------------------------------------
# ASCII table printer
# ---------------------------------------------------------------------------

def print_results_table(title: str, cols: list[str], rows: list[dict]) -> None:
    """
    Print a titled ASCII table to stdout.

    Args:
        title: Table heading string.
        cols:  List of column header strings (defines column order).
        rows:  List of dicts, each keyed by column name.
    """
    # Compute column widths
    col_widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            val = str(row.get(c, ""))
            col_widths[c] = max(col_widths[c], len(val))

    # Build format string
    sep = "   "
    header = sep.join(c.ljust(col_widths[c]) for c in cols)
    total_width = max(len(title), len(header))

    print()
    print(title)
    print("─" * total_width)
    print(header)
    print("─" * total_width)
    for row in rows:
        line = sep.join(str(row.get(c, "")).ljust(col_widths[c]) for c in cols)
        print(line)
    print()
