import gymnasium as gym
import numpy as np
import time
from typing import Any, Tuple, Dict, Optional, Union

from settings import K_STEPS, SUB_EPISODE_LIMIT, PHYSICS_FREQ, GAMMA

class PIDController:
    """PID Controller with Derivative-on-Measurement to prevent setpoint kicks."""
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_measurement = 0.0

    def reset(self):
        self.last_measurement = 0.0
        self.integral = 0.0

    def compute(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self.integral += error * dt
        if dt > 0:
            derivative = -(measurement - self.last_measurement) / dt
        else:
            derivative = 0.0
        self.last_measurement = measurement
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class FlightController:
    """Low-level controller translating high-level actions to RC commands."""
    def __init__(self):
        self.throttle_pid = PIDController(Kp=4, Ki=0, Kd=2)
        self.roll_pid = PIDController(Kp=2, Ki=0.1, Kd=0.5)
        self.pitch_pid = PIDController(Kp=2, Ki=0.1, Kd=0.5)
        self.yaw_pid = PIDController(Kp=2, Ki=1, Kd=0)
        
        self.hover_throttle = 1280
        self.min_throttle = 1200
        self.max_throttle = 1350
        self.min_roll_pitch = 1300
        self.max_roll_pitch = 1700
        
        self.reset()

    def reset(self):
        self.throttle_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
    
    def compute_rc_commands(self, high_level_action: np.ndarray, state_goal: np.ndarray, dt: float) -> np.ndarray:
        # state_goal: [altitude, roll, pitch, yaw_rate]
        current_alt_norm, current_roll_norm, current_pitch_norm, current_yaw_rate_norm = state_goal
        # high_level_action: [desired_alt, desired_roll, desired_pitch, desired_yaw_rate]
        desired_alt_norm, desired_roll_norm, desired_pitch_norm, desired_yaw_rate_norm = high_level_action
        
        self.throttle_pid.setpoint = desired_alt_norm
        throttle_pid_out = self.throttle_pid.compute(current_alt_norm, dt)
        
        rc_throttle = self.hover_throttle + (100 * throttle_pid_out)

        self.roll_pid.setpoint = desired_roll_norm
        roll_command = self.roll_pid.compute(current_roll_norm, dt)
        
        self.pitch_pid.setpoint = desired_pitch_norm
        pitch_command = self.pitch_pid.compute(current_pitch_norm, dt)
        
        self.yaw_pid.setpoint = desired_yaw_rate_norm
        yaw_command = self.yaw_pid.compute(current_yaw_rate_norm, dt)
        
        rc_roll = 1500 + 20 * roll_command
        rc_pitch = 1500 + 20 * pitch_command
        rc_yaw = 1500 + 20 * yaw_command
        
        rc_roll = np.clip(rc_roll, self.min_roll_pitch, self.max_roll_pitch)
        rc_pitch = np.clip(rc_pitch, self.min_roll_pitch, self.max_roll_pitch)

        rc_commands = np.clip([rc_throttle, rc_roll, rc_pitch, rc_yaw], 1000, 2000)
        rc_commands[0] = np.clip(rc_commands[0], self.min_throttle, self.max_throttle)
        
        return rc_commands.astype(np.float32)

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
            low=np.array([0, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.goal_alt = 0.5
        self.sub_episode_count = 0
        self.episode_start_state = None
        self.last_full_obs = None
        self.last_obs = None

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
        return 5.0 if alt_error < 0.05 and drift_error < 0.15 else 0.0

    def is_crashed(self, start_state: np.ndarray, end_state: np.ndarray) -> bool:
        current_alt = end_state[0]
        goal_alt = end_state[5]
        start_goal_distance = abs(start_state[0] - start_state[5])
        alt_error = abs(current_alt - goal_alt)
        drift_error = np.sqrt(end_state[1]**2 + end_state[2]**2)
        return drift_error >= 0.7 or alt_error > start_goal_distance + 0.06

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if options is None:
            options = {}
        
        if "initial_pos" not in options:
            start_alt = options.get("start_alt", 0.05)
            options["initial_pos"] = [0.0, 0.0, start_alt]
            
        full_obs, info = self.env.reset(seed=seed, options=options)
        self.last_full_obs = full_obs
        self.fc.reset()
        self.sub_episode_count = 0
        
        if "goal_alt" in options:
            self.goal_alt = options["goal_alt"]
        else:
            self.goal_alt = np.random.uniform(0.1, 0.9)
            
        obs = self._get_obs(full_obs)
        self.episode_start_state = obs
        self.last_obs = obs
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        p_start = self.calculate_potential(self.last_obs)

        crashed = False
        terminated = False
        truncated = False
        
        for _ in range(self.k_steps):
            fc_state = self._get_full_state_for_fc(self.last_full_obs)
            rc_commands = self.fc.compute_rc_commands(action, fc_state, 1.0/self.physics_freq)
            
            full_obs, _, env_terminated, env_truncated, info = self.env.step(rc_commands)
            self.last_full_obs = full_obs
            
            if env_terminated or env_truncated:
                crashed = True
                break
        
        self.sub_episode_count += 1
        self.last_obs = self._get_obs(self.last_full_obs)
        
        if not crashed:
            if self.is_crashed(self.episode_start_state, self.last_obs):
                crashed = True
        
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
        
        return self.last_obs, float(reward), terminated, truncated, info
