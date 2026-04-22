"""
discovery_agent.py - Unified HPM discovery agent.
Learns to use different tools for different curriculum phases.
"""

import torch
import numpy as np
import random
import sys
from typing import Dict, Any, List, Optional

# Increase limit for integer string conversion to support math.factorial discovery
try:
    sys.set_int_max_str_digits(10000)
except:
    pass

from .base_discovery import PureAgnosticDiscoveryAgent, ActionPattern
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.online_buffer import OnlineLearningBuffer
from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern


class UnifiedDiscoveryAgent(PureAgnosticDiscoveryAgent):
    """
    Unified HPM Agent for Cold-Start Discovery.
    Learns to use different tools for different curriculum phases.
    Uses OnlineLearningBuffer to fetch domain knowledge on failure.
    """
    def __init__(self, context_dim: int = 64, lm: Optional[LanguageModelPattern] = None):
        from hpm_ai_v3.tools.python_substrate import register_python_substrate
        from hpm_ai_v3.tools.innate import register_innate_tools
        register_python_substrate()
        register_innate_tools()

        super().__init__(context_feature_dim=context_dim)
        
        # Track hidden environment for formula discovery
        self.hidden_fn = None
        self.points = []
        self.confidence = 0.0
        
        # Online Learning Infrastructure
        self.online_buffer = OnlineLearningBuffer()
        self.current_phase_id = "initial"

        if lm is not None:
            from hpm_ai_v3.tools.tool_selector import ToolSelector
            self.tool_selector = ToolSelector(lm, alpha=0.5)

    def initialize_task(self, task: Dict) -> Dict:
        """Set up agent state for a new task."""
        self.current_task = task
        self.hidden_fn = task.get("true_fn")
        self.points = []
        self.confidence = 0.0
        self.predicted_solution = None
        self.current_phase_id = str(task.get("phase", "unknown"))
        
        # Initial context: the task text
        text = task.get("text", "")
        ctx = {"text": text}
        if "inputs" in task: ctx["inputs"] = task["inputs"]
        if "x" in task: ctx["x"] = task["x"]
        
        # CURRICULUM ADAPTATION: Handle 'hint' via weight-boost or unbound pattern
        hint = task.get("hint")
        if hint and "." in hint:
            mod, func = hint.split(".", 1)
            # Boost existing or add unbound pattern
            existing = next((p for p in self.population.patterns if getattr(p, 'module', None) == mod and getattr(p, 'function', None) == func), None)
            if existing:
                existing.weight = max(existing.weight * 2.0, 0.5)
            else:
                p = ActionPattern("python_call", module=mod, function=func)
                p.weight = 0.3 # Moderate weight for discoverability
                self.population.patterns.append(p)

        # AGNOSTIC BOOTSTRAP: Only list_modules and list_builtins (set in parent)
        if self.hidden_fn:
            has_eval = any(getattr(p, 'action_type', None) == "evaluate_at" for p in self.population.patterns)
            if not has_eval:
                self.population.patterns.append(ActionPattern("evaluate_at"))

        return ctx

    def on_step_complete(self, reward: float, pattern: Any, result: Dict):
        """Track consecutive failures per task type/domain."""
        if self.current_task is None: return
        
        # Domain key based on task type or hint
        domain = self.current_task.get("type", "generic")
        if self.current_task.get("hint"):
            domain += "_" + self.current_task["hint"]
            
        if reward > 0.9:
            self.online_buffer.record_success(domain)
        elif reward <= 0.0:
            self.online_buffer.record_failure(domain)
            
    def run_episode(self, task: Dict[str, Any], max_steps: int = 50) -> Optional[Any]:
        # Domain key for failure check
        domain = task.get("type", "generic")
        if task.get("hint"): domain += "_" + task["hint"]
        
        # CHECK FOR TRIGGERED ONLINE LEARNING
        if self.online_buffer.can_fetch(self.current_phase_id, domain):
            print(f"  [Failure Trigger] Stuck on {domain}. Attempting online fetch...")
            keywords = self.online_buffer.extract_keywords(task.get("text", ""))
            text = self.online_buffer.fetch_wikipedia(keywords)
            
            if text:
                self.online_buffer.add_to_buffer(text)
                self.online_buffer.mark_fetched(self.current_phase_id)
                
                # Find LM pattern in population
                lm = next((p for p in self.population.patterns if isinstance(p, LanguageModelPattern)), None)
                if lm:
                    print(f"  [Fine-Tune] Training LM on {len(text)} chars from Wikipedia...")
                    lm.fine_tune(text, epochs=2)
                else:
                    # Inject LM if missing but requested by curriculum
                    print("  [Fine-Tune] Injecting LanguageModelPattern for online learning.")
                    lm = LanguageModelPattern()
                    lm.fine_tune(text, epochs=2)
                    self.population.patterns.append(lm)

        return super().run_episode(task, max_steps)

    def evaluate_solution(self, solution: Any) -> float:
        """
        Generic evaluation based on task type.
        """
        if solution is None: return -1.0
        task = self.current_task
        if not task: return -1.0

        try:
            # 0. TOOL FAMILIARIZATION (Child-like practice)
            if task.get("type") == "practice":
                # For execution practice (demo), any successful return is a win
                if "demo" in task:
                    return 1.0 if solution is not None else 0.0

                # For listing tasks, check if the expected items are present
                if isinstance(task.get("answer"), list) and isinstance(solution, list):
                    # Check if answer items are present in solution results
                    # (Solution is often a list of dicts from list_functions)
                    sol_names = []
                    for item in solution:
                        if isinstance(item, dict) and "name" in item:
                            sol_names.append(item["name"])
                        else:
                            sol_names.append(str(item))
                    
                    matches = 0
                    for a in task["answer"]:
                        if any(str(a).lower() == str(sn).lower() for sn in sol_names):
                            matches += 1

                    if not task["answer"]: return 0.0
                    score = matches / len(task["answer"])
                    return 1.0 if score >= 0.8 else score
                return 0.0

            if task["type"] in ["arithmetic", "function_eval", "pretraining"]:
                # 1. Handle Dict-based targets (like Dissection)
                if isinstance(task["answer"], dict):
                    if not isinstance(solution, dict): return 0.0
                    match_count = 0
                    for k, v in task["answer"].items():
                        if k in solution and solution[k] == v:
                            match_count += 1
                    return float(match_count) / len(task["answer"])

                # 2. Handle list-based targets
                if isinstance(task["answer"], list):
                    if not isinstance(solution, list): return 0.0
                    if len(solution) == len(task["answer"]):
                        try:
                            if all(abs(float(a) - float(s)) < 1e-6 for a, s in zip(task["answer"], solution)):
                                return 1.0
                        except: pass
                    return 0.0

                # 3. Numeric match
                ans = float(task["answer"])
                sol = float(solution)
                error = abs(ans - sol)
                if error < 1e-6:
                    return 1.0
                else:
                    return 0.0
                
            elif task["type"] == "formula_discovery":
                if not self.points: return -1.0
                errors = []
                for px, py in self.points:
                    pred = np.polyval(solution, px)
                    errors.append((py - pred)**2)
                mse = float(np.mean(errors))
                self.confidence = 1.0 - min(np.sqrt(mse) / 0.1, 1.0)
                return 1.0 - min(mse, 2.0)

        except:
            return -0.5
        return -1.0

    def extract_features(self) -> torch.Tensor:
        """Truly agnostic: The agent only sees the raw numbers and environment state."""
        f = []

        # 1. Numeric Inputs
        inputs = self.context.get("inputs", [])
        if len(inputs) >= 2:
            f.extend([float(inputs[0])/100.0, float(inputs[1])/100.0])
        else:
            f.extend([0.0, 0.0])

        # 2. Local Environment State
        f.append(self.context.get("x", 0.0) / 10.0)
        f.append(len(self.points) / 50.0)
        f.append(self.confidence)

        # 3. CONTEXT PERCEPTION (Agnostic Hash)
        task_text = str(self.context.get("text", "")) # ROBUSTNESS FIX
        if task_text:
            import hashlib
            h = int(hashlib.md5(task_text.encode()).hexdigest(), 16)
            for _ in range(8):
                f.append((h % 100) / 100.0)
                h //= 100
        
        # 4. EPISODIC PERCEPTION (Robust Agnostic Flattening)
        mem_res = ToolRegistry.call("episodic_get_recent", n=5)
        for event in mem_res.get("events", []):
            res = event.get("result")
            
            # AGNOSTIC SUMMARY: Extract any numeric signal from the tool result
            signals = []
            if isinstance(res, (int, float)):
                signals.append(float(res))
            elif isinstance(res, (list, tuple, np.ndarray, torch.Tensor)):
                # Flatten nested structures safely
                for item in list(res)[:10]:
                    try:
                        if isinstance(item, (int, float)):
                            signals.append(float(item))
                        elif isinstance(item, str):
                            # Try to parse number from string
                            signals.append(float(item))
                    except: pass
            
            if signals:
                # Incorporate first 8 signals into context
                f.extend(signals[:8])
                break
        
        while len(f) < self.context_dim: f.append(0.0)
        return torch.tensor(f[:self.context_dim], dtype=torch.float32)

    def act(self, step_idx: int = 0, step_population: bool = True) -> Dict[str, Any]:
        """Inject environment tool if we have a hidden function."""
        if self.hidden_fn and not ToolRegistry.get_tool_info("evaluate_at"):
            ToolRegistry.register("evaluate_at", self._env_evaluate, ["x"], "result", 0.1)
        return super().act(step_idx, step_population=step_population)

    def _env_evaluate(self, x: float) -> Dict[str, Any]:
        if self.hidden_fn:
            y = self.hidden_fn(x)
            return {"x": x, "y": y, "status": "success"}
        return {"status": "failed", "error": "No hidden function"}
