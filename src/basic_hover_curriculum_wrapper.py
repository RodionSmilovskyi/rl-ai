import gymnasium as gym
import numpy as np
from typing import Any, Tuple, Dict, Optional, Union
from drone_wrappers import DroneHRLWrapper

class BasicHoverCurriculumWrapper(DroneHRLWrapper):
    """
    Wrapper for DroneEnv to learn altitude control, extending DroneHRLWrapper.
    Uses the same observation and action space as DroneHRLWrapper.
    """
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, **kwargs)

    # Removed manual step override to allow dynamic axis unlocking via locked_axes mechanism
