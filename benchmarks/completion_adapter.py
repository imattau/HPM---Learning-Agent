
"""Helpers for thin completion-aware benchmark adapters.

These utilities monkeypatch existing benchmark factory call sites so we can
compare the current baseline with completion-aware agent configurations
without copying the benchmark logic.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Any, Callable


@contextmanager
def patch_attr(module, attr: str, replacement):
    original = getattr(module, attr)
    setattr(module, attr, replacement)
    try:
        yield original
    finally:
        setattr(module, attr, original)


def make_agent_factory(original: Callable[..., Any], *, seed: int | None = None, overrides: dict[str, Any] | None = None):
    overrides = dict(overrides or {})

    def _factory(*args, **kwargs):
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        for key, value in overrides.items():
            kwargs.setdefault(key, value)
        return original(*args, **kwargs)

    return _factory


def make_orchestrator_factory(original: Callable[..., Any], *, seed: int | None = None, overrides: dict[str, Any] | None = None):
    overrides = dict(overrides or {})

    def _factory(*args, **kwargs):
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        for key, value in overrides.items():
            kwargs.setdefault(key, value)
        if seed is not None and "agent_seeds" not in kwargs:
            try:
                bound = inspect.signature(original).bind_partial(*args, **kwargs)
                n_agents = bound.arguments.get("n_agents")
                if n_agents is not None:
                    kwargs["agent_seeds"] = [seed + i * 101 for i in range(int(n_agents))]
            except Exception:
                pass
        return original(*args, **kwargs)

    return _factory
