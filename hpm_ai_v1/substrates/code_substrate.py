"""Code Substrate for HPM AI v1.

Parses local Python files into AST nodes, serving as the L1/L2 sensory environment
for the self-refactoring agent.
"""
import ast
import os
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
from hpm.substrate.base import hash_vectorise, ExternalSubstrate

class LocalCodeSubstrate:
    """
    Extends ExternalSubstrate to ingest Python codebase and expose ASTs.
    Implements the core HPM ExternalSubstrate protocol.
    """

    def __init__(self, root_dir: str, feature_dim: int = 32):
        self.root_dir = root_dir
        self.feature_dim = feature_dim
        self._cache: Dict[str, List[np.ndarray]] = {}

    def fetch(self, query: str) -> List[np.ndarray]:
        """Fetch vectorised snippets of code matching the query."""
        if query in self._cache:
            return self._cache[query]

        results = []
        for filepath in self.get_all_python_files():
            source = self.read_file(filepath)
            if source and query.lower() in source.lower():
                results.append(hash_vectorise(source, self.feature_dim))

        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        """Alignment between pattern and the codebase density."""
        # Standard HPM frequency logic: how many files match this pattern?
        vecs = self.fetch(getattr(pattern, 'label', 'code'))
        if not vecs: return 0.0
        return 1.0 # Simplified for prototype

    def stream(self) -> Iterator[np.ndarray]:
        """Stream the entire codebase as vectorised files."""
        for filepath in self.get_all_python_files():
            source = self.read_file(filepath)
            if source:
                yield hash_vectorise(source, self.feature_dim)

    def get_all_python_files(self) -> List[str]:
        """Recursively find all .py files in the target directory."""
        py_files = []
        for root, _, files in os.walk(self.root_dir):
            # Ignore some standard folders
            if any(part in root for part in ['.git', '.venv', '__pycache__']):
                continue
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))
        return py_files

    def read_file(self, filepath: str) -> Optional[str]:
        """Read raw text of a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def parse_ast(self, filepath: str) -> Optional[ast.AST]:
        """Parse a Python file into an AST."""
        source = self.read_file(filepath)
        if source is None:
            return None
        try:
            return ast.parse(source, filename=filepath)
        except SyntaxError as e:
            print(f"Syntax error in {filepath}: {e}")
            return None

    def extract_functions(self, tree: ast.AST) -> List[ast.FunctionDef]:
        """Extract all function definitions from an AST (L2 Structural Anatomy)."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        return functions

    def extract_classes(self, tree: ast.AST) -> List[ast.ClassDef]:
        """Extract all class definitions from an AST (L2 Structural Anatomy)."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node)
        return classes
