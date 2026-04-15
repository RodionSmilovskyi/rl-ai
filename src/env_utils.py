from typing import Optional
from drone_env import DroneEnv
from basic_hover_curriculum_wrapper import BasicHoverCurriculumWrapper
from round_action_wrapper import RoundActionWrapper
from settings import SUB_EPISODE_LIMIT

def make_drone_env(rank: int, seed: int = 0, render_mode: Optional[str] = "rgb_array"):
    """
    Utility for creating the specialized Drone environment with curriculum and action rounding.
    """
    def _init():
        env = DroneEnv(render_mode=render_mode)
        # Using SUB_EPISODE_LIMIT from settings
        env = RoundActionWrapper(BasicHoverCurriculumWrapper(env, k_steps=20, sub_episode_limit=SUB_EPISODE_LIMIT), 2)
        env.reset(seed=seed + rank)
        return env
    return _init
