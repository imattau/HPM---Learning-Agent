"""
EmpiricalOracle and CountingOracle.

Ported from:
- experiment_unified_perception_action.py (EmpiricalOracle)
- experiment_meta_strategy_controller.py (CountingOracle)
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Optional

from hpm_ai_v2.utils.state import S_DIM


class EmpiricalOracle:
    """Computes a 20D empirical state vector from execution outputs."""

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
    ) -> np.ndarray:
        s = np.zeros(S_DIM)
        valid_outputs = [o for o, e in zip(outputs, errors) if e is None]
        if not valid_outputs:
            s[0] = 0.0
            s[9] = 1.0
            return s
        s[0] = 1.0

        def safe_float(x: Any) -> float:
            try:
                return float(max(min(x, 1e100), -1e100))
            except (OverflowError, TypeError):
                return 0.0

        is_list, lens, means, mins, maxs, firsts, lasts, is_int = [], [], [], [], [], [], [], []
        for out in valid_outputs:
            if isinstance(out, list):
                is_list.append(1.0)
                lens.append(len(out))
                num_out = [safe_float(x) for x in out if isinstance(x, (int, float))]
                if num_out:
                    means.append(float(np.mean(num_out)))
                    mins.append(float(np.min(num_out)))
                    maxs.append(float(np.max(num_out)))
                    firsts.append(num_out[0])
                    lasts.append(num_out[-1])
            elif isinstance(out, (int, float)):
                is_list.append(0.0)
                is_int.append(1.0)
                sf = safe_float(out)
                means.append(sf)
                mins.append(sf)
                maxs.append(sf)
                firsts.append(sf)
                lasts.append(sf)
        s[1] = float(np.mean(is_list)) if is_list else 0.0
        s[2] = float(np.mean(lens)) if lens else 0.0
        s[3] = float(np.mean(means)) if means else 0.0
        s[4] = float(np.mean(mins)) if mins else 0.0
        s[5] = float(np.mean(maxs)) if maxs else 0.0
        s[6] = float(np.mean(firsts)) if firsts else 0.0
        s[7] = float(np.mean(lasts)) if lasts else 0.0
        s[8] = float(np.mean(is_int)) if is_int else 0.0
        s[10] = 1.0 if 'for ' in code else 0.0
        s[11] = 1.0 if '.append(' in code else 0.0
        s[12] = 1.0 if 'if ' in code else 0.0
        s[13] = 1.0 if ('+=' in code or '-=' in code or '*=' in code) else 0.0
        s[14] = 1.0 if 'x = inp' in code else 0.0
        s[15] = 1.0 if 'val = item' in code else 0.0
        s[16] = 1.0 if 'res = []' in code else 0.0
        return s


class CountingOracle(EmpiricalOracle):
    """Wraps EmpiricalOracle with a per-task call counter."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
    ) -> np.ndarray:
        self.call_count += 1
        return super().compute_state(outputs, errors, code)
