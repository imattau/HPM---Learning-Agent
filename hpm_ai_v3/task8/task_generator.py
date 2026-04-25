"""
task_generator.py - Auto-generate hundreds of training examples for Phase 1 discovery.
"""

import random
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Tuple, Callable

class TaskGenerator:
    """Generates diverse tasks with known answers for supervised discovery."""
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Template definitions: (weight, generator_function)
        self.templates: List[Tuple[float, Callable[[], Dict[str, Any]]]] = [
            (0.15, self._gen_discovery_practice), # New practice type
            (0.20, self._gen_arithmetic),
            (0.15, self._gen_function_eval),
            (0.15, self._gen_parsing_number),
            (0.15, self._gen_parsing_with_unit),
            (0.15, self._gen_simple_physics),
            (0.05, self._gen_comparison),
        ]

    def generate(self, n: int) -> List[Dict[str, Any]]:
        """Generate n tasks."""
        # 1. Always start with some guaranteed discovery practice
        tasks = []
        practice_count = min(n // 5, 20)
        for _ in range(practice_count):
            tasks.append(self._gen_discovery_practice())
            
        # 2. Generate remaining random tasks
        for _ in range(n - len(tasks)):
            # Select template by weight
            weights, templates = zip(*self.templates)
            template = random.choices(templates, weights=weights)[0]
            tasks.append(template())
        return tasks

    def _gen_discovery_practice(self) -> Dict[str, Any]:
        """Tasks that reward tool exploration with unique context signals."""
        import importlib
        
        # 1. CORE MODULES
        core_mods = ["math", "numpy", "sympy", "scipy.constants", "re", "json", "spacy", "operator", "builtins", "textblob"]
        
        choices = [
            ("CORE_MODULES", core_mods),
            ("FUNCS:builtins", [f['name'] for f in self._get_module_funcs("builtins")]),
            ("FUNCS:math", [f['name'] for f in self._get_module_funcs("math")]),
            ("FUNCS:operator", [f['name'] for f in self._get_module_funcs("operator")]),
        ]
        
        text, answer = random.choice(choices)
        return {
            "text": text,
            "answer": answer,
            "type": "practice"
        }

    def _get_module_funcs(self, module_name: str) -> List[Dict]:
        """Helper to get function names for ground truth."""
        try:
            import importlib
            import inspect
            mod = importlib.import_module(module_name)
            functions = []
            for name, func in inspect.getmembers(mod, inspect.isroutine):
                if not name.startswith("_"):
                    functions.append({"name": name})
            return functions
        except:
            return []

    def _gen_arithmetic(self) -> Dict[str, Any]:
        """Generate basic arithmetic: '3 + 5', '10 / 2', etc."""
        ops = [
            ('+', lambda a, b: a + b),
            ('-', lambda a, b: a - b),
            ('*', lambda a, b: a * b),
            ('/', lambda a, b: a / b if b != 0 else 0),
        ]
        op_symbol, op_fn = random.choice(ops)
        
        # Integer or float
        if random.random() < 0.7:
            a = random.randint(1, 100)
            b = random.randint(1, 100) if op_symbol != '/' else random.randint(1, 20)
        else:
            a = round(random.uniform(1, 50), 2)
            b = round(random.uniform(1, 50), 2)
            if op_symbol == '/' and b == 0: b = 1
            
        answer = op_fn(a, b)
        
        # Format variations
        formats = [
            f"{a} {op_symbol} {b}",
            f"What is {a} {op_symbol} {b}?",
            f"Calculate {a} {op_symbol} {b}",
            f"{a}{op_symbol}{b}",
        ]
        text = random.choice(formats)
        return {"text": text, "answer": answer, "type": "arithmetic"}

    def _gen_function_eval(self) -> Dict[str, Any]:
        """Generate function evaluation: 'sin(1.2)', 'sqrt(16)', etc."""
        funcs = [
            ('sin', np.sin),
            ('cos', np.cos),
            ('tan', np.tan),
            ('exp', np.exp),
            ('log', np.log),
            ('sqrt', np.sqrt),
            ('abs', abs),
        ]
        name, fn = random.choice(funcs)
        
        # Choose appropriate argument range
        if name == 'log':
            x = round(random.uniform(0.1, 10), 3)
        elif name == 'tan':
            x = round(random.uniform(-1.4, 1.4), 3)
        elif name == 'sqrt':
            x = round(random.uniform(0, 25), 3)
        else:
            x = round(random.uniform(-5, 5), 3)
            
        answer = float(fn(x))
        formats = [
            f"{name}({x})",
            f"Evaluate {name}({x})",
            f"What is {name}({x})?",
        ]
        text = random.choice(formats)
        return {"text": text, "answer": answer, "type": "function_eval"}

    def _gen_parsing_number(self) -> Dict[str, Any]:
        """Extract number from sentence."""
        value = random.randint(1, 1000)
        if random.random() < 0.3:
            value = round(random.uniform(1, 100), 2)
        
        templates = [
            f"The value is {value}",
            f"Result: {value}",
            f"Found {value} items",
            f"Speed = {value}",
            f"Temperature is {value} degrees",
        ]
        text = random.choice(templates)
        return {"text": text, "answer": value, "type": "parsing"}

    def _gen_parsing_with_unit(self) -> Dict[str, Any]:
        """Extract number with unit."""
        value = random.randint(1, 200)
        units = ['m', 'kg', 's', 'm/s', 'N', 'J', 'W', 'Hz', 'Pa']
        unit = random.choice(units)
        
        templates = [
            f"The length is {value}{unit}",
            f"Mass: {value} {unit}",
            f"Force of {value} {unit} applied",
            f"Velocity = {value} {unit}",
        ]
        text = random.choice(templates)
        return {"text": text, "answer": value, "type": "parsing"}

    def _gen_simple_physics(self) -> Dict[str, Any]:
        """Generate simple physics word problems with known answers."""
        if random.random() > 0.5:
            # Kinematics: s = u*t + 0.5*a*t^2
            u = random.randint(0, 20)
            a = random.randint(1, 10)
            t = random.randint(1, 15)
            s = u * t + 0.5 * a * t**2
            templates = [
                f"A car accelerates from {u} m/s at {a} m/s² for {t} seconds. How far does it travel?",
                f"Starting at {u} m/s, accelerating at {a} m/s² for {t} s. Distance?",
            ]
            text = random.choice(templates)
            return {"text": text, "answer": s, "type": "physics", "tolerance": 1.0}
        else:
            # Newton's second law: F = m*a
            m = random.randint(1, 50)
            a = random.randint(1, 20)
            F = m * a
            templates = [
                f"A force of {F} N acts on a mass of {m} kg. What is the acceleration?",
                f"Mass = {m} kg, Force = {F} N. Find acceleration.",
            ]
            text = random.choice(templates)
            return {"text": text, "answer": a, "type": "physics", "tolerance": 0.5}

    def _gen_comparison(self) -> Dict[str, Any]:
        """Generate comparison questions."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        if a > b:
            answer = a
        elif b > a:
            answer = b
        else:
            answer = a
            
        templates = [
            f"Which is larger: {a} or {b}?",
            f"Max of {a} and {b}",
        ]
        text = random.choice(templates)
        return {"text": text, "answer": answer, "type": "comparison"}


def generate_phase1_dataset(n: int = 500, output_file: str = None) -> List[Dict]:
    """Generate a dataset of n tasks for Phase 1 training."""
    gen = TaskGenerator(seed=42)
    tasks = gen.generate(n)
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f"Saved {n} tasks to {output_file}")
    return tasks


if __name__ == "__main__":
    tasks = generate_phase1_dataset(200, "phase1_tasks.json")
    print(f"Generated {len(tasks)} tasks.")
    print("Sample tasks:")
    for task in tasks[:5]:
        print(f"  {task['text']} -> {task['answer']}")
