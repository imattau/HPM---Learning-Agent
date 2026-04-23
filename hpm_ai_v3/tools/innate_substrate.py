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
                if fn is None and "innate_substrate" in module:
                    # Heuristic: if not found at module level, try InnateCognitiveSubstrate class
                    cls = getattr(mod, "InnateCognitiveSubstrate", None)
                    if cls:
                        fn = getattr(cls, function, None)
            
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
        if not isinstance(lst, list):
            raise TypeError(
                f"find_in_list requires 'lst' argument to be a list, got {type(lst).__name__}"
            )
        try:
            return lst.index(value)
        except (ValueError, TypeError):
            return None

    def list_index(self, value: Any, lst: list) -> Optional[int]:
        """Alias for find_in_list. Return index of value in lst, or None."""
        return self.find_in_list(value, lst)

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

    # ── Group F: Uncertainty / Confidence ──────────────────────────────────

    def normalize(self, values: list) -> list:
        """Convert list of non-negative numerics to probability distribution."""
        try:
            if not values:
                return []
            floats = []
            for v in values:
                try:
                    floats.append(float(v))
                except (TypeError, ValueError):
                    return []
            total = sum(floats)
            if total == 0:
                n = len(floats)
                return [1.0 / n] * n
            return [v / total for v in floats]
        except Exception:
            return []

    def entropy(self, probs: list) -> float:
        """Shannon entropy in bits. Zeros skipped (0*log2(0) = 0 by convention)."""
        try:
            if not probs:
                return 0.0
            result = 0.0
            for p in probs:
                try:
                    p = float(p)
                except (TypeError, ValueError):
                    continue
                if p > 0:
                    result -= p * math.log2(p)
            return result
        except Exception:
            return 0.0

    def argmax(self, values: list) -> int:
        """Index of maximum value. Returns -1 for empty list. Ties: lowest index."""
        try:
            if not values:
                return -1
            best_idx = 0
            best_val = values[0]
            for i in range(1, len(values)):
                if values[i] > best_val:
                    best_val = values[i]
                    best_idx = i
            return best_idx
        except Exception:
            return -1

    def clamp(self, value, lo, hi) -> float:
        """Constrain value to [lo, hi]. Swaps lo/hi if lo > hi."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            try:
                return float(lo)
            except Exception:
                return 0.0
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except (TypeError, ValueError):
            return v
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        if v < lo_f:
            return lo_f
        if v > hi_f:
            return hi_f
        return v

    # ── Group G: Goal / Task Decomposition ─────────────────────────────────

    _DECOMPOSE_VERBS = {
        "find", "compute", "calculate", "sort", "filter", "group", "count",
        "sum", "detect", "compare", "get", "list", "check", "estimate",
        "build", "flatten", "split", "match",
    }
    _DECOMPOSE_STOPS = {"a", "an", "the", "in", "of", "for", "with", "from", "to", "that", "which"}
    _DECOMPOSE_MODS = {"by", "with", "from", "greater", "less", "above", "below", "than", "where", "between"}

    def decompose_text(self, text: str) -> dict:
        """Extract {verb, object, modifier} from natural language goal text."""
        empty = {"verb": "", "object": "", "modifier": ""}
        try:
            if not text or not text.strip():
                return empty
            tokens = text.strip().split()
            if not tokens:
                return empty

            # Find verb
            verb = ""
            verb_idx = -1
            for i, tok in enumerate(tokens):
                if tok.lower().rstrip(".,!?") in self._DECOMPOSE_VERBS:
                    verb = tok.lower().rstrip(".,!?")
                    verb_idx = i
                    break
            if not verb:
                verb = tokens[0].lower().rstrip(".,!?")
                verb_idx = 0

            # Find object: first token after verb not in stopwords
            obj = ""
            obj_idx = -1
            for i in range(verb_idx + 1, len(tokens)):
                tok = tokens[i].lower().rstrip(".,!?")
                if tok not in self._DECOMPOSE_STOPS:
                    obj = tok
                    obj_idx = i
                    break

            # Find modifier: token after object starting with a modifier word
            modifier = ""
            if obj_idx >= 0:
                for i in range(obj_idx + 1, len(tokens)):
                    tok = tokens[i].lower().rstrip(".,!?")
                    if tok in self._DECOMPOSE_MODS:
                        modifier = " ".join(tokens[i:]).lower().rstrip(".,!?")
                        break

            return {"verb": verb, "object": obj, "modifier": modifier}
        except Exception:
            return empty

    def estimate_progress(self, current, target) -> float:
        """Normalised progress current/target clamped to [0, 1]."""
        try:
            c = float(current)
            t = float(target)
        except (TypeError, ValueError):
            return 0.0
        try:
            if t == 0:
                return 1.0 if c == 0 else 0.0
            return self.clamp(c / t, 0.0, 1.0)
        except Exception:
            return 0.0

    def check_constraint(self, value, constraint_str: str) -> bool:
        """Evaluate a constraint string against value. Unknown constraints return True."""
        try:
            cs = constraint_str.strip()

            # type:typename
            m = re.match(r"^type:(\w+)$", cs)
            if m:
                type_name = m.group(1)
                type_map = {
                    "int": int, "float": float, "str": str,
                    "list": list, "dict": dict, "bool": bool,
                }
                t = type_map.get(type_name)
                if t is None:
                    return True
                # int check: bool is subclass of int, exclude
                if type_name == "int":
                    return isinstance(value, int) and not isinstance(value, bool)
                return isinstance(value, t)

            # len comparisons: len > N etc.
            m = re.match(r"^len\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$", cs)
            if m:
                op, n_str = m.group(1), m.group(2)
                try:
                    length = len(value)
                    n = float(n_str)
                    return self._apply_op(float(length), op, n)
                except (TypeError, ValueError):
                    return False

            # in [a, b, c]
            m = re.match(r"^in\s*\[(.+)\]$", cs)
            if m:
                parts = [p.strip() for p in m.group(1).split(",")]
                candidates = []
                for p in parts:
                    try:
                        candidates.append(float(p))
                    except ValueError:
                        candidates.append(p.strip("'\""))
                try:
                    return float(value) in candidates or value in candidates
                except (TypeError, ValueError):
                    return value in candidates

            # numeric comparisons: >= N, > N, etc.
            m = re.match(r"^(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$", cs)
            if m:
                op, n_str = m.group(1), m.group(2)
                try:
                    v = float(value)
                    n = float(n_str)
                    return self._apply_op(v, op, n)
                except (TypeError, ValueError):
                    return False

            # Unknown constraint: permissive
            return True
        except Exception:
            return True

    def _apply_op(self, a: float, op: str, b: float) -> bool:
        """Apply a comparison operator string."""
        if op == ">":  return a > b
        if op == ">=": return a >= b
        if op == "<":  return a < b
        if op == "<=": return a <= b
        if op == "==": return a == b
        if op == "!=": return a != b
        return True

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
            # If there's a string in the pool that looks like a regex, use it
            # Otherwise return a default digit matcher
            for val in pool:
                if isinstance(val, str) and any(c in val for c in r"\[].*+?^$|"):
                    return val
            return r"\d+"

        # Special case: list parameter
        if target_type == "list" or param_name in ("lst", "obj", "items", "values"):
            for val in pool:
                if isinstance(val, list):
                    return val
            # Fallback: extract list from task_text if possible
            # (Very basic extraction of [a, b, c] strings)
            m = re.search(r"\[(.*)\]", task_text)
            if m:
                try:
                    return [x.strip().strip("'\"") for x in m.group(1).split(",")]
                except: pass
            return self.to_list(task_text)

        # Special case: string/text parameter
        if target_type == "str" or param_name in ("string", "text", "s", "seq"):
            # Prefer task_text for string params IF it's likely the target
            # but if there's a string in the pool that ISN'T task_text, use it
            for val in pool:
                if isinstance(val, str) and val != task_text:
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
