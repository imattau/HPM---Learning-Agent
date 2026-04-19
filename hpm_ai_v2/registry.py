"""
KnowledgeRegistry: A global manifest tracking HPM knowledge bases across the workspace.
Prevents knowledge fragmentation by mapping domain/topic to stable forest paths.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

class KnowledgeRegistry:
    """
    Singleton-like registry to track all learned knowledge bases.
    Persists to data/knowledge_base/registry.json.
    """
    DEFAULT_REGISTRY_PATH = Path("data/knowledge_base/registry.json")

    def __init__(self, registry_path: Optional[str | Path] = None):
        self.path = Path(registry_path) if registry_path else self.DEFAULT_REGISTRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.manifest, f, indent=4)

    def register_forest(self, domain_id: str, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register or update a forest in the global manifest."""
        path_str = str(Path(path).absolute())
        entry = self.manifest.get(domain_id, {})
        
        # Update entry
        entry.update({
            "path": path_str,
            "last_updated": time.time(),
            "last_updated_str": time.ctime(),
        })
        if metadata:
            entry.update(metadata)
            
        self.manifest[domain_id] = entry
        self._save()
        print(f"      [REGISTRY] Registered domain '{domain_id}' at {path_str}")

    def get_forest_path(self, domain_id: str) -> Optional[Path]:
        """Retrieve the absolute path to a domain's forest."""
        entry = self.manifest.get(domain_id)
        if entry:
            return Path(entry["path"])
        return None

    def list_domains(self) -> List[str]:
        """Return all registered domain IDs."""
        return list(self.manifest.keys())

    def get_all_entries(self) -> Dict[str, Dict[str, Any]]:
        """Return the full manifest."""
        return dict(self.manifest)

# Global instance for easy access
_global_registry: Optional[KnowledgeRegistry] = None

def get_registry() -> KnowledgeRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = KnowledgeRegistry()
    return _global_registry
