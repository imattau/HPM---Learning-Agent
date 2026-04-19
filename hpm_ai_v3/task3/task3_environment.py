import numpy as np

class NonStationaryEnvironment:
    def __init__(self, stable_duration=100, volatile_duration=50, volatile_change_freq=10, input_dim=2, noise_std=0.05):
        self.stable_duration = stable_duration
        self.volatile_duration = volatile_duration
        self.volatile_change_freq = volatile_change_freq
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.regime_schedule = self._generate_schedule()
        self.current_step = 0
        
    def _generate_schedule(self):
        schedule = []
        for _ in range(self.stable_duration): schedule.append(('stable', None))
        w = np.random.randn(self.input_dim)
        for i in range(self.volatile_duration):
            if i % self.volatile_change_freq == 0: w = np.random.randn(self.input_dim)
            schedule.append(('volatile', w.copy()))
        return schedule
    
    def reset(self): self.current_step = 0
    
    def step(self):
        if self.current_step >= len(self.regime_schedule): return None, None, None
        regime, w = self.regime_schedule[self.current_step]
        x = np.random.uniform(-1, 1, self.input_dim)
        y = np.dot(x, (np.array([0.8, -0.4]) if regime == 'stable' else w)) + np.random.normal(0, self.noise_std)
        self.current_step += 1
        return x, y, {'regime': regime}
