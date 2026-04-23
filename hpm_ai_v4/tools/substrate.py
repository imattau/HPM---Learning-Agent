class ExternalSubstrate:
    """A symbolic notebook where patterns can be persisted and shared as descriptions."""
    def __init__(self):
        # id -> description (e.g., symbolic rule string)
        self.storage = {}

    def write(self, pattern_id, description):
        """Write a pattern's symbolic representation to the substrate."""
        self.storage[pattern_id] = description

    def read(self, pattern_id):
        """Read a pattern's representation from the substrate."""
        return self.storage.get(pattern_id, None)

    def broadcast(self, patterns, threshold=0.1):
        """Automatically persist high-weight patterns to the shared environment."""
        for p in patterns:
            if p.weight > threshold and p.id not in self.storage:
                desc = f"pattern_{p.id}_complexity_{p.complexity}_compressed"
                self.write(p.id, desc)
