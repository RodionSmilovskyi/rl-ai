from typing import Optional
from drone_env import DroneEnv
from drone_hrl_wrapper import DroneHRLWrapper
from round_action_wrapper import RoundActionWrapper
from settings import SUB_EPISODE_LIMIT, K_STEPS

def make_drone_env(rank: int, seed: int = 0, render_mode: Optional[str] = "rgb_array"):
    """
    Utility for creating the specialized Drone environment with HRL wrapper and action rounding.
    """
    def _init():
        env = DroneEnv(render_mode=render_mode)
        env = RoundActionWrapper(DroneHRLWrapper(env, k_steps=K_STEPS, sub_episode_limit=SUB_EPISODE_LIMIT), 2)
        env.reset(seed=seed + rank)
        return env
    return _init

