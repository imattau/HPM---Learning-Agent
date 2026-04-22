"""
base_discovery.py - Pure HPM agent that discovers solutions via evaluator-driven
population dynamics. Uses InnateCognitiveSubstrate for argument resolution.
"""

import torch
import numpy as np
import networkx as nx
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from hpm_ai_v3.pattern import HPMPattern
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.evaluators import EvaluatorManager
from hpm_ai_v3.compiler import SubstrateCompiler
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.operators.pipeline_recombination import PipelineRecombinationOperator
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate


class ActionPattern(HPMPattern):
    """
    A minimal pattern that wraps a tool call. 
    Agnostic: Argument resolution is delegated to the InnateCognitiveSubstrate.
    """
    def __init__(self, 
                 action_type: str, 
                 module: str = None, 
                 function: str = None,
                 pattern_id: str = None):
        super().__init__(pattern_id)
        self.action_type = action_type
        self.module = module
        self.function = function
        self.substrate_type = "functional_action"
        self.required_observation_keys = ["reward", "outcome"]
        self.cost = 0.01
        self.accuracy = 0.0
        self.exploration_temperature: float = 1.0
        self.recent_use_count: int = 0

    @property
    def tool_name(self) -> str:
        return f"{self.module}.{self.function}" if self.module and self.function else self.action_type

    @property
    def tool_description(self) -> str:
        if self.module and self.function:
            return f"{self.module}.{self.function}: call {self.function} from {self.module}"
        return self.action_type

    @property
    def output_key(self) -> str:
        return "result"

    @property
    def input_keys(self) -> List[str]:
        """Agnostic patterns don't specify keys, they use the pool."""
        return ["pool", "text"]

    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        reward = observations.get("reward", torch.tensor(0.0, device=self._device))
        return reward.to(self._device)

    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, Any]:
        """Execute the action using resolved_args provided in context."""
        resolved_args = context.get("resolved_args", [])
        
        try:
            if self.action_type == "python_call":
                result = ToolRegistry.call("python_call", 
                                           module=self.module, 
                                           function=self.function, 
                                           args=resolved_args)
            else:
                # Builtin tools: map positional resolved_args to registered input_keys
                info = ToolRegistry.get_tool_info(self.action_type)
                if info and info.get("input_keys") and isinstance(resolved_args, list):
                    kwargs = {}
                    for i, key in enumerate(info["input_keys"]):
                        if i < len(resolved_args):
                            kwargs[key] = resolved_args[i]
                    result = ToolRegistry.call(self.action_type, **kwargs)
                else:
                    result = ToolRegistry.call(self.action_type)
            
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.sample(context)

    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        pass

    def mark_used(self):
        super().mark_used()
        self.recent_use_count += 1

    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, ActionPattern):
            return 1.0
        if self.action_type != other.action_type:
            return 1.0
        if self.module != other.module or self.function != other.function:
            return 0.8
        return 0.0

    def extract_causal_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(self.id, action=self.tool_name)
        return g


class PureAgnosticDiscoveryAgent(ABC):
    """
    HPM discovery agent with minimal innate tools (priors).
    Uses InnateCognitiveSubstrate for stable argument resolution.
    """
    def __init__(self, context_feature_dim: int = 64):
        self.context_dim = context_feature_dim

        # 1. CORE INNATE TOOLS (The agent's 'body')
        patterns = [
            ActionPattern("arithmetic", pattern_id="innate_calc"),
            ActionPattern("float", pattern_id="innate_float"),
            ActionPattern("split", pattern_id="innate_split"),
            ActionPattern("re_findall", pattern_id="innate_regex"),
            ActionPattern("index", pattern_id="innate_index"),
        ]
        patterns.append(ActionPattern("list_modules", pattern_id="discovery_mod"))

        self.population = PatternPopulation(patterns)

        self.evaluator_mgr = EvaluatorManager()
        self.compiler = SubstrateCompiler()
        self.pipeline_recomb = PipelineRecombinationOperator(
            co_occurrence_threshold=0.2,
            min_weight=0.1,
            max_pipeline_length=3
        )
        self.substrate = InnateCognitiveSubstrate()
        self.tool_selector = None  # Set externally with ToolSelector instance

        self.context = {}
        self.discovered_modules = set()
        self.discovered_functions = defaultdict(set)
        self.current_task = None
        self.predicted_solution = None
        self.episode_sequence = []

    @abstractmethod
    def evaluate_solution(self, solution: Any) -> float:
        """Task-specific evaluation."""
        pass

    @abstractmethod
    def extract_features(self) -> torch.Tensor:
        """Extract context features for the population step."""
        pass

    def act(self, step_idx: int = 0, step_population: bool = True, resample_count: int = 0) -> Dict[str, Any]:
        """Select and execute a pattern based purely on population weights."""
        if not self.population.patterns:
            print("  [Warning] Population extinct. Re-injecting innate tools.")
            self._reinject_innate_tools()
            if not self.population.patterns: return {"status": "failed", "error": "No patterns"}

        # 1. SELECTION (Evaluator-Gated Replicator)
        weights = np.array([p.weight for p in self.population.patterns])
        if self.tool_selector is not None and self.current_task:
            weights = self.tool_selector.apply(
                self.current_task.get("text", ""),
                weights,
                self.population.patterns
            )
        total = weights.sum()
        probs = weights / total if total > 1e-6 else np.ones(len(weights)) / len(weights)
        chosen = np.random.choice(len(self.population.patterns), p=probs)
        action_pattern = self.population.patterns[chosen]
        
        # 2. SUBSTRATE ARGUMENT RESOLUTION
        pool = self._get_pool()
        task_text = self.current_task.get("text", "") if self.current_task else ""

        # Handle both python_call patterns and specific innate tools
        if isinstance(action_pattern, ActionPattern):
            mod = action_pattern.module
            func = action_pattern.function
            if not mod or not func:
                # Builtin tool: find its underlying Python mapping if possible
                info = ToolRegistry.get_tool_info(action_pattern.action_type)
                # For builtins like 'arithmetic', we just pass task_text or first pool item
                resolved_args = [pool[0]] if pool else [task_text]
            else:
                _, _, resolved_args = self.substrate.resolve_call(mod, func, pool, task_text)
        else:
            resolved_args = [task_text] if task_text else []

        action_pattern.mark_used()
        if isinstance(action_pattern, ActionPattern):
            self.episode_sequence.append(action_pattern.tool_name)

        context = {
            "pool": pool,
            "text": task_text,
            "resolved_args": resolved_args,
            "context_features": self.extract_features()
        }
        
        # 3. EXECUTION
        result = action_pattern.sample(context)

        # RECORD IN EPISODIC MEMORY
        if result.get("status") == "success":
            ToolRegistry.call("episodic_append", event={
                "tool": getattr(action_pattern, 'tool_name', getattr(action_pattern, 'agent_name', action_pattern.id)),
                "args": resolved_args,
                "result": result.get("result")
            })

        # 4. EVALUATE OUTCOME
        status = result.get("status", "failed")
        val = result.get("result")
        
        # BINARY SUCCESS MODEL: Correct Answer = 1.0, Everything else = -0.5
        is_valid = self._is_solution_valid(val)
        reward = 1.0 if is_valid else -0.5

        # 5. REINFORCE & ABSORB
        obs = {
            "reward": torch.tensor([reward], device=action_pattern._device),
            "outcome": result,
            "input": context["context_features"].unsqueeze(0),
            "context_features": context["context_features"].unsqueeze(0)
        }
        
        if step_population:
            self.evaluator_mgr.update_epistemic(action_pattern, obs)
            self.evaluator_mgr.update_affective(action_pattern, reward)
            
            # Pattern Field (Attention Signal)
            field_signal = [float(p.weight * (1.0 + p.coherence_score)) for p in self.population.patterns]
            self.population.step(self.evaluator_mgr, obs, self.compiler, pattern_field_signal=field_signal)

        # GROUNDING: If an unbound pattern produced the CORRECT ANSWER, absorb it as a FACT
        if reward == 1.0 and isinstance(action_pattern, ActionPattern) and action_pattern.module and action_pattern.function:
            new_fact = ActionPattern("python_call", action_pattern.module, action_pattern.function)
            new_fact.weight = 0.1 # High weight for verified facts
            new_fact.accuracy = 1.0 
            new_fact.affective_score = 1.0
            self.population.patterns.append(new_fact)
            print(f"    [Discovery] Absorbed VERIFIED FACT: {new_fact.tool_name}")

        # LOGGING
        if status == "success" and reward > 0:
            print(f"    -> Result: {val}")
            self._absorb_discovery(action_pattern, result)
        elif status == "failed":
            err = result.get("error", "Unknown")
            print(f"    -> Error: {str(err)[:50]}...")

        if reward > 0.9:
            self.pipeline_recomb.record_sequence(self.episode_sequence)

        # Hook for subclasses to track failures/successes
        if hasattr(self, 'on_step_complete'):
            self.on_step_complete(reward, action_pattern, result)

        # Defensive property access for ActionPattern/ToolPattern/Composite compatibility
        action_name = getattr(action_pattern, 'tool_name', 
                             getattr(action_pattern, 'agent_name', action_pattern.id))
        return {"status": status, "result": val, "reward": reward, "action": action_name}

    def _get_pool(self) -> List[Any]:
        """Collect candidate values from episodic memory and task context."""
        pool = []
        mem_res = ToolRegistry.call("episodic_get_recent", n=10)
        for event in mem_res.get("events", []):
            res = event.get("result")
            if res is not None and isinstance(res, (int, float, str, list, dict)):
                pool.append(res)
        if self.current_task:
            if "inputs" in self.current_task:
                pool.extend(self.current_task["inputs"])
            text = self.current_task.get("text", "")
            if text:
                pool.append(text)
        # Deduplicate
        seen = set()
        unique = []
        for item in pool:
            key = (type(item).__name__, str(item))
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return unique

    def _reinject_innate_tools(self):
        """Emergency re-injection of innate tools if population vanishes."""
        patterns = [
            ActionPattern("arithmetic", pattern_id="innate_calc"),
            ActionPattern("float", pattern_id="innate_float"),
            ActionPattern("split", pattern_id="innate_split"),
            ActionPattern("re_findall", pattern_id="innate_regex"),
            ActionPattern("index", pattern_id="innate_index"),
            ActionPattern("list_modules", pattern_id="discovery_mod")
        ]
        for p in patterns:
            if not any(op.id == p.id for op in self.population.patterns):
                p.weight = 0.01
                self.population.patterns.append(p)

    def run_episode(self, task: Dict[str, Any], max_steps: int = 50) -> Optional[Any]:
        self.current_task = task
        self.episode_sequence = []
        self.predicted_solution = None

        # Clear episodic memory between tasks
        ToolRegistry.call("episodic_clear")

        print(f"\n[Episode] Task: {task.get('text', 'No Text')}")

        # Pure HPM: Treat demo as a pre-defined interaction step
        if "demo" in task:
            demo = task["demo"]
            res_val = None
            if isinstance(demo, list):
                for d in demo: res_val = self._execute_demo_as_step(d)
            else:
                res_val = self._execute_demo_as_step(demo)
            
            # Short-circuit for practice tasks
            if task.get("type") == "practice" and res_val is not None:
                self.predicted_solution = res_val
                return self.predicted_solution

        # Recombination Step
        composite = self.pipeline_recomb.should_recombine(self.population)
        if composite:
            print(f"  [Recombination] New Composite Pattern Discovered: {composite.id}")
            self.population.patterns.append(composite)

        for step in range(max_steps):
            step_res = self.act(step)
            if step_res["reward"] > 0.9:
                self.predicted_solution = step_res["result"]
                return self.predicted_solution

        return None

    def _execute_demo_as_step(self, demo: Dict) -> Any:
        """Pure HPM: Treat demo as a privileged observed episode."""
        mod = demo.get("module")
        func = demo.get("function")
        args = demo.get("args", [])
        
        print(f"  [Observation] Observing demo step: {mod}.{func}({args})")
        res = ToolRegistry.call("python_call", module=mod, function=func, args=args)
        
        if res.get("status") == "success":
            # Record in episodic memory
            ToolRegistry.call("episodic_append", event={
                "tool": f"{mod}.{func}",
                "args": args,
                "result": res.get("result")
            })

            # Absorbing a demo is like cultural inheritance
            p = ActionPattern("python_call", module=mod, function=func)
            p.weight = 0.1 # High initial weight for demonstrated facts
            p.accuracy = 1.0 # Maximum accuracy prior
            p.affective_score = 1.0
            self.population.patterns.append(p)
            self._absorb_discovery(p, res)
            return res.get("result")
        return None

    def _is_solution_valid(self, val: Any) -> bool:
        if self.current_task is None: return False
        if hasattr(self, 'evaluate_solution'):
            return self.evaluate_solution(val) > 0.9
        if "answer" not in self.current_task: return False
        return self._answers_match(val, self.current_task["answer"])

    def _answers_match(self, val: Any, answer: Any) -> bool:
        if val is None: return False
        try:
            if isinstance(answer, (list, tuple)):
                if not isinstance(val, (list, tuple)) or len(val) != len(answer): return False
                try:
                    return all(abs(float(a) - float(v)) < 1e-6 for a, v in zip(answer, val))
                except:
                    return all(str(a).strip().lower() == str(v).strip().lower() for a, v in zip(answer, val))
            try:
                return abs(float(val) - float(answer)) < 1e-6
            except:
                return str(val).strip().lower() == str(answer).strip().lower()
        except:
            return val == answer

    def _absorb_discovery(self, pattern: ActionPattern, result: Dict):
        """When list_modules or list_functions succeeds, add discovered items as new patterns."""
        res_val = result.get("result")
        if not res_val: return

        if pattern.action_type == "list_modules":
            for mod in res_val:
                if mod not in self.discovered_modules:
                    self.discovered_modules.add(mod)
                    new_pat = ActionPattern("list_functions", module=mod)
                    new_pat.weight = 0.01
                    self.population.patterns.append(new_pat)
                    print(f"    [Absorption] New Module Pattern: {mod}")

        elif pattern.action_type == "list_functions":
            mod = result.get("module") or pattern.module
            if not mod: return
            funcs = res_val
            for f in funcs:
                fname = f.get("name")
                if fname and fname not in self.discovered_functions[mod]:
                    self.discovered_functions[mod].add(fname)
                    new_pat = ActionPattern("python_call", module=mod, function=fname)
                    new_pat.weight = 0.01
                    self.population.patterns.append(new_pat)
