import math
import numpy as np
from hpm_ai_v5.planning.acrobot import AcrobotEnv

def diagnostic():
    env = AcrobotEnv()
    obs = env.reset()
    print(f"Initial obs: {obs}")
    
    max_height = -2.0
    for i in range(500):
        # Try simple heuristic: torque in direction of v2
        action = 1.0 if obs["theta2_dot"] > 0 else -1.0
        obs, reward, done = env.step(action)
        
        theta1 = obs["theta1"]
        theta2 = obs["theta2"]
        tip_y = -math.cos(theta1) - math.cos(theta1 + theta2)
        max_height = max(max_height, tip_y)
        
        if i % 50 == 0:
            print(f"Step {i}: tip_y={tip_y:.2f}, obs={obs}")
        
        if done:
            print(f"Goal reached at step {i}!")
            break
    else:
        print(f"Goal NOT reached. Max height: {max_height:.2f}")

if __name__ == "__main__":
    diagnostic()
