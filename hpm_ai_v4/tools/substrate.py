import random
import copy

class ExternalSubstrate:
    """
    A shared memory space (collective memory) where agents can publish and retrieve patterns.
    This enables cultural transmission and pattern gossip across the population.
    """
    def __init__(self):
        # id -> HierarchicalPattern object
        self.storage = {}

    def write(self, pattern_id, pattern):
        """Publish a pattern to the substrate (deep copy to avoid shared references)."""
        self.storage[pattern_id] = copy.deepcopy(pattern)

    def read(self, pattern_id):
        """Retrieve a specific pattern from the substrate."""
        return self.storage.get(pattern_id, None)

    def broadcast(self, patterns, threshold=0.1):
        """Automatically publish high-weight patterns to the shared environment."""
        for p in patterns:
            if p.weight > threshold:
                self.write(p.id, p)

    def get_random_pattern(self):
        """Retrieve a random pattern from the substrate (for gossip)."""
        if not self.storage:
            return None
        return random.choice(list(self.storage.values()))
