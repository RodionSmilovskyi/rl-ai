import gymnasium as gym
import numpy as np

class RoundActionWrapper(gym.ActionWrapper):
    def __init__(self, env, decimals=3):
        super().__init__(env)
        self.decimals = decimals

    def action(self, action: np.ndarray) -> np.ndarray:
        # Rounds the float32 array to the specified decimal places
        return np.round(action, decimals=self.decimals)
