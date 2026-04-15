import gymnasium as gym
import numpy as np
import os
from typing import Any, Tuple, Dict, Optional, Union
from PIL import Image, ImageDraw, ImageFont

from settings import K_STEPS, SUB_EPISODE_LIMIT, PHYSICS_FREQ, GAMMA, ASSETS_DIRECTORY
from flight_controller import FlightController

class DroneHRLWrapper(gym.Wrapper):
    """
    HRL Wrapper for DroneEnv.
    Executes K-steps of low-level control for each high-level action.
    """
    def __init__(self, env: gym.Env, k_steps: int = K_STEPS, sub_episode_limit: int = SUB_EPISODE_LIMIT, gamma: float = GAMMA):
        super().__init__(env)
        self.k_steps = k_steps
        self.sub_episode_limit = sub_episode_limit
        self.gamma = gamma
        self.fc = FlightController()
        self.physics_freq = PHYSICS_FREQ
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.goal_alt = 0.5
        self.sub_episode_count = 0
        self.episode_start_state = None
        self.last_full_obs = None
        self.last_obs = None
        self.next_episode_params = {}
        self.locked_axes = []
        self.initial_pos = [0.0, 0.0, 0.05]
        self.pos_debug_id = None

    def set_next_episode_params(self, **params):
        """
        Set parameters for the next episode and beyond.
        Supported params: goal_alt, initial_pos, locked_axes
        """
        self.next_episode_params.update(params)

    def _get_obs(self, full_obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            full_obs["altitude"][0],
            full_obs["shift_x"][0],
            full_obs["shift_y"][0],
            full_obs["velocity_x"][0],
            full_obs["velocity_y"][0],
            self.goal_alt
        ], dtype=np.float32)

    def _get_full_state_for_fc(self, full_obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            full_obs["altitude"][0],
            full_obs["roll"][0],
            full_obs["pitch"][0],
            full_obs["yaw"][0]
        ], dtype=np.float32)

    def calculate_potential(self, state_goal: np.ndarray) -> float:
        current_alt = state_goal[0]
        goal_alt = state_goal[5]
        alt_error = abs(current_alt - goal_alt)
        drift_error = np.sqrt(state_goal[1]**2 + state_goal[2]**2)
        return -(alt_error + 2 * drift_error) * 3

    def calculate_sparse_reward(self, state_goal: np.ndarray) -> float:
        current_alt = state_goal[0]
        goal_alt = state_goal[5]
        alt_error = abs(current_alt - goal_alt)
        drift_error = np.sqrt(state_goal[1]**2 + state_goal[2]**2)
        return 5.0 if alt_error < 0.1 and drift_error < 0.15 else 0.0

    def is_crashed(self, start_state: np.ndarray, end_state: np.ndarray) -> bool:
        current_alt = end_state[0]
        goal_alt = end_state[5]
        start_goal_distance = abs(start_state[0] - start_state[5])
        alt_error = abs(current_alt - goal_alt)
        drift_error = np.sqrt(end_state[1]**2 + end_state[2]**2)
        return drift_error >= 0.5 or alt_error > start_goal_distance + 0.06

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if options is None:
            options = {}
        
        # Apply persistent updates from next_episode_params
        if "locked_axes" in self.next_episode_params:
            self.locked_axes = self.next_episode_params["locked_axes"]
        if "initial_pos" in self.next_episode_params:
            self.initial_pos = self.next_episode_params["initial_pos"]
        
        # goal_alt is usually one-time, let's extract it if present
        target_goal_alt = self.next_episode_params.pop("goal_alt", None)
        if target_goal_alt is not None:
            options["goal_alt"] = target_goal_alt
        
        # Apply persistent values to options for this reset
        if "locked_axes" not in options:
            options["locked_axes"] = self.locked_axes
        if "initial_pos" not in options:
            options["initial_pos"] = self.initial_pos
            
        full_obs, info = self.env.reset(seed=seed, options=options)
        self.last_full_obs = full_obs
        self.fc.reset()
        self.sub_episode_count = 0
        
        # Goal altitude: prioritize options, otherwise use last known or 0.5
        if "goal_alt" in options:
            self.goal_alt = options["goal_alt"]
        # else: self.goal_alt remains what it was (e.g. from previous episode or init)
            
        obs = self._get_obs(full_obs)
        self.episode_start_state = obs
        self.last_obs = obs

        # Display debug text in human mode
        if getattr(self.env.unwrapped, "use_gui", False):
            client = self.env.unwrapped.client
            client.addUserDebugText(
                f"Start Pos: {self.initial_pos}",
                [0, 0, 1.2],
                textColorRGB=[1, 0, 0],
                textSize=1.2,
            )
            client.addUserDebugText(
                f"Goal Altitude: {self.goal_alt:.2f}",
                [0, 0, 1.0],
                textColorRGB=[1, 0, 0],
                textSize=1.2,
            )
            
            pos, _ = client.getBasePositionAndOrientation(self.env.unwrapped.drone_id)
            self.pos_debug_id = client.addUserDebugText(
                f"Current Pos: {[round(c, 2) for c in pos]}",
                [0, 0, 0.8],
                textColorRGB=[1, 0, 0],
                textSize=1.2,
            )

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        p_start = self.calculate_potential(self.last_obs)

        # Apply axis locking by zeroing out action components
        if 'roll' in self.locked_axes:
            action[1] = 0.0
        if 'pitch' in self.locked_axes:
            action[2] = 0.0
        if 'yaw' in self.locked_axes:
            action[3] = 0.0

        crashed = False
        terminated = False
        truncated = False
        
        for _ in range(self.k_steps):
            fc_state = self._get_full_state_for_fc(self.last_full_obs)
            rc_commands = self.fc.compute_rc_commands(action, fc_state, 1.0/self.physics_freq)
            
            full_obs, _, env_terminated, env_truncated, info = self.env.step(rc_commands)
            self.last_full_obs = full_obs
            
            # Update Current Pos display
            if getattr(self.env.unwrapped, "use_gui", False) and self.pos_debug_id is not None:
                client = self.env.unwrapped.client
                pos, _ = client.getBasePositionAndOrientation(self.env.unwrapped.drone_id)
                client.addUserDebugText(
                    f"Current Pos: {[round(c, 2) for c in pos]}",
                    [0, 0, 0.8],
                    textColorRGB=[1, 0, 0],
                    textSize=1.2,
                    replaceItemUniqueId=self.pos_debug_id
                )
            
            if env_terminated or env_truncated:
                crashed = True
                break
        
        self.sub_episode_count += 1
        self.last_obs = self._get_obs(self.last_full_obs)
        is_success = False
        if not crashed:
            if self.is_crashed(self.episode_start_state, self.last_obs):
                crashed = True
            else:
                # Check if it reached the goal (sparse reward > 0)
                if self.calculate_sparse_reward(self.last_obs) > 0:
                    is_success = True
        
        p_end = self.calculate_potential(self.last_obs)
        shaping_reward = self.gamma * p_end - p_start
        sparse_reward = self.calculate_sparse_reward(self.last_obs)
        
        steps_remaining = self.sub_episode_limit - self.sub_episode_count
        
        if crashed:
            reward = -2.0 * steps_remaining
            terminated = True
        else:
            reward = shaping_reward + sparse_reward
            if self.sub_episode_count >= self.sub_episode_limit:
                truncated = True
        
        info["is_success"] = is_success
        return self.last_obs, float(reward), terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(os.path.join(ASSETS_DIRECTORY, "arial.ttf"), 16)
        except Exception:
            font = ImageFont.load_default()
        
        draw.text((10, 10), f"Start Pos: {self.initial_pos}", fill=(255, 0, 0), font=font)
        draw.text((10, 30), f"Goal Alt: {self.goal_alt:.2f}", fill=(255, 0, 0), font=font)
        
        client = self.env.unwrapped.client
        pos, _ = client.getBasePositionAndOrientation(self.env.unwrapped.drone_id)
        draw.text((10, 50), f"Current Pos: {[round(c, 2) for c in pos]}", fill=(255, 0, 0), font=font)
        
        return np.array(img)
