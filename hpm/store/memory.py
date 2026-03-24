from typing import Optional, List, Tuple

class InMemoryStore:
    """In-memory PatternStore. Default backend for Phase 1."""

    def __init__(self):
        # (pattern_id, agent_id) -> (pattern, weight, agent_id)
        self._data: dict = {}

    def save(self, pattern, weight: float, agent_id: str = "default_agent") -> None:
        self._data[(pattern.id, agent_id)] = (pattern, weight, agent_id)

    def _find_key(self, pattern_id: str, agent_id: str) -> Optional[Tuple[str, str]]:
        """Helper to find key, with fallback to any agent if agent_id is default."""
        key = (pattern_id, agent_id)
        if key in self._data:
            return key
        
        if agent_id == "default_agent":
            # Search across all agents for this pattern_id
            for (pid, aid) in self._data.keys():
                if pid == pattern_id:
                    return (pid, aid)
        return None

    def load(self, pattern_id: str, agent_id: str = "default_agent") -> Tuple[object, float]:
        key = self._find_key(pattern_id, agent_id)
        if not key:
            raise KeyError(f"Pattern {pattern_id} for agent {agent_id} not found.")
        pattern, weight, _ = self._data[key]
        return pattern, weight

    def query(self, agent_id: str = "default_agent") -> List[Tuple[object, float]]:
        return [
            (p, w)
            for (pid, aid), (p, w, _) in self._data.items()
            if aid == agent_id
        ]

    def query_all(self) -> List[Tuple[object, float, str]]:
        return list(self._data.values())

    def delete(self, pattern_id: str, agent_id: str = "default_agent") -> None:
        key = self._find_key(pattern_id, agent_id)
        if key:
            self._data.pop(key, None)

    def has(self, pattern_id: str, agent_id: str = "default_agent") -> bool:
        return self._find_key(pattern_id, agent_id) is not None

    def update_weight(self, pattern_id: str, agent_id: str, weight: Optional[float] = None) -> None:
        if weight is None:
            actual_weight = float(agent_id)
            actual_agent_id = "default_agent"
        else:
            actual_weight = weight
            actual_agent_id = agent_id

        key = self._find_key(pattern_id, actual_agent_id)
        if key:
            p, _, aid = self._data[key]
            self._data[key] = (p, actual_weight, aid)
