class InMemoryStore:
    """In-memory PatternStore. Default backend for Phase 1."""

    def __init__(self):
        # (pattern_id, agent_id) -> (pattern, weight, agent_id)
        self._data: dict = {}

    def save(self, pattern, weight: float, agent_id: str) -> None:
        self._data[(pattern.id, agent_id)] = (pattern, weight, agent_id)

    def load(self, pattern_id: str, agent_id: str) -> tuple:
        pattern, weight, _ = self._data[(pattern_id, agent_id)]
        return pattern, weight

    def query(self, agent_id: str) -> list:
        return [
            (p, w)
            for (pid, aid), (p, w, _) in self._data.items()
            if aid == agent_id
        ]

    def query_all(self) -> list:
        return list(self._data.values())

    def delete(self, pattern_id: str, agent_id: str) -> None:
        self._data.pop((pattern_id, agent_id), None)

    def has(self, pattern_id: str, agent_id: str) -> bool:
        return (pattern_id, agent_id) in self._data

    def update_weight(self, pattern_id: str, agent_id: str, weight: float) -> None:
        key = (pattern_id, agent_id)
        if key in self._data:
            p, _, aid = self._data[key]
            self._data[key] = (p, weight, aid)
