import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def make_env(env_id, seed, rank):
    """
    Utility for creating a gymnasium environment with a unique seed and rank.
    """
    def _init():
        env = Monitor(gym.make(env_id, render_mode="rgb_array"))
        env.reset(seed=seed + rank)
        return env
    return _init

def ensure_directory(path):
    """
    Ensures that a directory exists, creating it if necessary.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
