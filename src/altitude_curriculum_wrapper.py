import gymnasium as gym
import numpy as np
from typing import Any, Tuple, Dict, Optional, Union
from drone_wrappers import DroneHRLWrapper

class AltitudeCurriculumWrapper(DroneHRLWrapper):
    """
    Wrapper for DroneEnv to learn altitude control, extending DroneHRLWrapper.
    Uses the same observation and action space as DroneHRLWrapper.
    Locks high-level roll, pitch, and yaw actions to 0.
    """
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, **kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # action: [desired_alt, desired_roll, desired_pitch, desired_yaw_rate]
        # Lock roll, pitch, and yaw rate to 0 for this curriculum
        locked_action = np.array([action[0], 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Call the parent's step with the locked action
        return super().step(locked_action)
