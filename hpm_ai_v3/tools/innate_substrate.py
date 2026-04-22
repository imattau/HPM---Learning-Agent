# hpm_ai_v3/tools/innate_substrate.py
"""
InnateCognitiveSubstrate - Permanent cognitive infrastructure for HPM agents.
Handles signature inspection, type coercion, and pattern extraction.
Never in the population. Never subject to replicator dynamics.
"""
import re
import math
import inspect
import importlib
from typing import Any, Dict, List, Optional, Tuple


class InnateCognitiveSubstrate:
    """Permanent singleton. Wraps every tool call with correct argument resolution."""

    # ── Group A: Introspection ──────────────────────────────────────────────

    def inspect_signature(self, module: str, function: str) -> Optional[Dict]:
        """Return {params: [{name, type, default}]} for module.function."""
        try:
            mod = importlib.import_module(module)
            # Handle dotted names like str.upper
            if "." in function:
                parts = function.split(".", 1)
                type_obj = getattr(mod, parts[0], None) or getattr(__builtins__ if isinstance(__builtins__, dict) else __import__("builtins"), parts[0], None)
                fn = getattr(type_obj, parts[1], None) if type_obj else None
            else:
                fn = getattr(mod, function, None)
            if fn is None:
                return None
            sig = inspect.signature(fn)
            params = []
            for name, param in sig.parameters.items():
                if name in ("self", "cls"):
                    continue
                ann = param.annotation
                type_hint = ann.__name__ if ann != inspect.Parameter.empty and hasattr(ann, "__name__") else None
                default = None if param.default is inspect.Parameter.empty else param.default
                params.append({"name": name, "type": type_hint, "default": default})
            return {"params": params}
        except Exception:
            return None

    def get_type(self, value: Any) -> str:
        """Return string type name of value."""
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "float"
        if isinstance(value, str): return "str"
        if isinstance(value, list): return "list"
        if isinstance(value, dict): return "dict"
        return type(value).__name__

    def describe_value(self, value: Any) -> Dict:
        """Return {type, length, preview, numeric_value}."""
        result = {"type": self.get_type(value), "preview": str(value)[:50]}
        if isinstance(value, (list, str, dict)):
            result["length"] = len(value)
        numeric = self.to_float(value)
        result["numeric_value"] = numeric
        return result

    def list_callable_functions(self, module: str) -> List[Dict]:
        """Return [{name, signature}] for all callable public functions in module."""
        try:
            mod = importlib.import_module(module)
            result = []
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                obj = getattr(mod, name)
                if callable(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = "(...)"
                    result.append({"name": name, "signature": f"{name}{sig}"})
            return result
        except Exception:
            return []

    # ── Group B: Type Operations ────────────────────────────────────────────

    def coerce(self, value: Any, target_type: Optional[str]) -> Optional[Any]:
        """Convert value to target_type. Returns None on failure."""
        if target_type is None:
            return value
        try:
            if target_type == "int":
                return int(float(str(value).strip()))
            if target_type == "float":
                return float(str(value).strip())
            if target_type == "str":
                return str(value)
            if target_type == "list":
                return self.to_list(value)
            if target_type == "bool":
                return bool(value)
        except Exception:
            return None
        return value

    def to_int(self, value: Any) -> Optional[int]:
        return self.coerce(value, "int")

    def to_float(self, value: Any) -> Optional[float]:
        try:
            return float(str(value).strip())
        except Exception:
            return None

    def to_str(self, value: Any) -> str:
        return str(value)

    def to_list(self, value: Any) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return list(value)
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    def safe_call(self, module: str, function: str, *args) -> Any:
        """Call module.function(*args). Returns error dict instead of raising."""
        try:
            mod = importlib.import_module(module)
            fn = getattr(mod, function)
            return fn(*args)
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    # ── Group C: Pattern Matching / Perception ──────────────────────────────

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text as floats."""
        matches = re.findall(r"-?\d+\.?\d*", str(text))
        result = []
        for m in matches:
            try:
                result.append(float(m))
            except ValueError:
                pass
        return result

    def match_regex(self, pattern: str, text: str) -> List[str]:
        """Return all regex matches in text."""
        try:
            return re.findall(pattern, str(text))
        except Exception:
            return []

    def find_in_list(self, value: Any, lst: list) -> Optional[int]:
        """Return index of value in lst, or None."""
        try:
            return lst.index(value)
        except (ValueError, TypeError):
            return None

    def split_text(self, text: str, delimiter: Optional[str] = None) -> List[str]:
        """Split text by delimiter (default: whitespace)."""
        if delimiter:
            return str(text).split(delimiter)
        return str(text).split()

    def compare_values(self, a: Any, b: Any) -> str:
        """Compare two values. Returns 'equal', 'greater', 'less', 'incomparable'."""
        try:
            fa, fb = float(a), float(b)
            if fa == fb: return "equal"
            return "greater" if fa > fb else "less"
        except (TypeError, ValueError):
            if a == b: return "equal"
            return "incomparable"

    # ── Group D: Structural / Relational Reasoning ─────────────────────────

    def build_mapping(self, keys, values) -> dict:
        """Zip keys and values into a dict. Truncates to shorter list."""
        try:
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            if not isinstance(values, (list, tuple)):
                values = [values]
            return dict(zip(keys, values))
        except Exception:
            return {}

    def invert_mapping(self, mapping: dict) -> dict:
        """Swap keys and values. Skips unhashable values. Duplicate values: last key wins."""
        result = {}
        try:
            for k, v in mapping.items():
                try:
                    result[v] = k
                except TypeError:
                    pass
        except Exception:
            pass
        return result

    def get_nested(self, obj, path: str):
        """Safe deep access using dot-separated path. Numeric segments used as list indices."""
        if path == "":
            return obj
        try:
            parts = path.split(".")
            current = obj
            for part in parts:
                if current is None:
                    return None
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, (list, tuple)):
                    try:
                        current = current[int(part)]
                    except (ValueError, IndexError):
                        return None
                else:
                    return None
            return current
        except Exception:
            return None

    def flatten(self, nested) -> list:
        """Recursively flatten nested lists. Dicts and non-list scalars kept as-is."""
        result = []
        try:
            if not isinstance(nested, list):
                return [nested]
            for item in nested:
                if isinstance(item, list):
                    result.extend(self.flatten(item))
                else:
                    result.append(item)
        except Exception:
            pass
        return result

    def group_by(self, items: list, key_fn) -> dict:
        """Partition items by key_fn(item). Items where key_fn raises are skipped."""
        result = {}
        try:
            for item in items:
                try:
                    key = key_fn(item)
                    if key not in result:
                        result[key] = []
                    result[key].append(item)
                except Exception:
                    pass
        except Exception:
            pass
        return result

    # ── Group E: Temporal / Sequential Reasoning ───────────────────────────

    def detect_trend(self, series: list) -> str:
        """Characterise direction: 'increasing'|'decreasing'|'stable'|'volatile'."""
        try:
            nums = []
            for v in series:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    pass
            if len(nums) < 2:
                return "stable"
            diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
            if all(d == 0 for d in diffs):
                return "stable"
            if all(d > 0 for d in diffs):
                return "increasing"
            if all(d < 0 for d in diffs):
                return "decreasing"
            return "volatile"
        except Exception:
            return "stable"

    def diff_sequence(self, series: list) -> list:
        """First-order differences: series[i+1] - series[i]."""
        try:
            nums = []
            for v in series:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    return []
            if len(nums) < 2:
                return []
            return [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
        except Exception:
            return []

    def find_repeating(self, sequence: list):
        """Return smallest repeating sub-list, or None."""
        try:
            n = len(sequence)
            if n < 2:
                return None
            for period in range(1, n // 2 + 1):
                unit = sequence[:period]
                tiles = (n // period)
                remainder = n % period
                if unit * tiles + unit[:remainder] == sequence:
                    return unit
            return None
        except Exception:
            return None

    def sliding_window(self, sequence: list, n: int) -> list:
        """All contiguous windows of size n."""
        try:
            if n <= 0 or n > len(sequence):
                return []
            return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]
        except Exception:
            return []

    # ── Core: resolve_call ──────────────────────────────────────────────────

    def resolve_call(
        self,
        module: str,
        function: str,
        pool: List[Any],
        task_text: str
    ) -> Tuple[str, str, List[Any]]:
        """
        Resolve arguments for module.function from pool and task_text.
        Returns (module, function, resolved_args_list).
        """
        sig = self.inspect_signature(module, function)
        if not sig or not sig["params"]:
            # No signature info — pass task_text as sole arg
            nums = self.extract_numbers(task_text)
            return (module, function, nums[:1] if nums else [task_text])

        resolved = []
        for i, param in enumerate(sig["params"]):
            # Only resolve required arguments (no default)
            # or first 2 if all have defaults (heuristic)
            if param.get("has_default") and i >= 2:
                break
                
            val = self._match_param(param, pool, task_text, function)
            resolved.append(val)

        return (module, function, resolved)

    def _match_param(
        self,
        param: Dict,
        pool: List[Any],
        task_text: str,
        function_name: str = ""
    ) -> Any:
        """Find best value from pool for a parameter."""
        target_type = param.get("type")
        param_name = param.get("name", "")

        # Special case: regex pattern parameter
        if param_name == "pattern" or "pattern" in param_name.lower():
            return r"\d+"

        # Special case: string/text parameter
        if target_type == "str" or param_name in ("string", "text", "s", "seq"):
            # Prefer task_text for string params
            for val in pool:
                if isinstance(val, str):
                    return val
            return task_text

        # Try pool values, coerce to target type
        for val in pool:
            coerced = self.coerce(val, target_type)
            if coerced is not None:
                return coerced

        # Fallback: extract numbers from task_text
        nums = self.extract_numbers(task_text)
        if nums:
            coerced = self.coerce(nums[0], target_type)
            if coerced is not None:
                return coerced

        # Last resort: task_text
        return task_text
