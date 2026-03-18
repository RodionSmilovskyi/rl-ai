import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict, Any, Optional, Tuple

class AltitudeCurriculumCallback(BaseCallback):
    """
    Custom callback for Stable-Baselines3 to manage a dynamic altitude curriculum.
    Phases:
    1. Fixed start position, random goal altitude, locked roll, pitch, and yaw.
    2. Unlock Roll.
    3. Unlock Pitch.
    4. Unlock Yaw.
    """
    def __init__(
        self,
        eval_env: Any,
        success_threshold: float = 0.8,
        eval_freq: int = 2000,
        n_eval_episodes: int = 10,
        max_phase: int = 4,
        verbose: int = 1,
        export_callback: Optional[Any] = None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_phase = max_phase
        self.export_callback = export_callback
        
        self.current_phase = 1
        
        # Initial parameters for Phase 1
        self.locked_axes = ['roll', 'pitch', 'yaw']
        self.initial_pos = [0.0, 0.0, 0.05]
        
    def _on_training_start(self) -> None:
        """
        Called when training starts. Set initial environment parameters and goals.
        """
        self._apply_phase_params()
        # Set initial random goals and start positions for all training environments
        for i in range(self.training_env.num_envs):
            start_alt, goal_alt = self._get_random_start_and_goal()
            self.training_env.env_method('set_next_episode_params', 
                                         goal_alt=goal_alt,
                                         initial_pos=[0.0, 0.0, start_alt],
                                         indices=i)

    def _get_random_start_and_goal(self) -> Tuple[float, float]:
        """
        Generate random start and goal altitudes with a minimum gap of 0.15.
        If they are closer than 0.15, push start altitude down.
        """
        start_alt = float(np.random.uniform(0.05, 0.95))
        # goal_alt = float(np.random.uniform(0.1, 0.9))
        skewed_random = np.random.beta(2, 1) 
        goal_alt = round(0.1 + (skewed_random * 0.8), 3)
        
        if abs(start_alt - goal_alt) < 0.15:
            # Push start altitude down
            start_alt = goal_alt - 0.15
            # Ensure it doesn't go below minimum start altitude
            if start_alt < 0.05:
                # If pushing down is not possible, push it up
                start_alt = goal_alt + 0.15
                
        return start_alt, goal_alt

    def _apply_phase_params(self) -> None:
        """
        Apply the current phase's parameters to all vectorized environments.
        """
        if self.verbose > 0:
            print(f"\n[Curriculum] >>> Entering Phase {self.current_phase} <<<")
            print(f"[Curriculum] Locked axes: {self.locked_axes}")
            print(f"[Curriculum] Fixed initial pos (Eval): {self.initial_pos}")

        # Update name prefix for video recording if VecVideoRecorder is used
        if hasattr(self.eval_env, 'name_prefix'):
            self.eval_env.name_prefix = f"eval_altitude_phase_{self.current_phase}"

        # Update training environments - Only locked_axes, randomization handled in _on_step
        self.training_env.env_method('set_next_episode_params', 
                                     locked_axes=self.locked_axes.copy())
        
        # Update evaluation environment
        self.eval_env.env_method('set_next_episode_params', 
                                 locked_axes=self.locked_axes.copy(), 
                                 initial_pos=self.initial_pos.copy())

    def _on_step(self) -> bool:
        """
        Check if it's time to evaluate and potentially advance the curriculum.
        Also handle per-episode goal randomization for training environments.
        """
        # Randomize goal and start pos for the NEXT reset for any env that just finished an episode
        dones = self.locals.get("dones")
        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    start_alt, goal_alt = self._get_random_start_and_goal()
                    self.training_env.env_method('set_next_episode_params', 
                                                 goal_alt=goal_alt,
                                                 initial_pos=[0.0, 0.0, start_alt],
                                                 indices=i)

        if self.n_calls % self.eval_freq == 0:
            success_rate = self._evaluate_success_rate()
            
            # Log metrics with phase-specific prefix
            prefix = f"curriculum/phase_{self.current_phase}"
            self.logger.record(f"{prefix}/success_rate", success_rate)
            
            if self.verbose > 0:
                print(f"[Curriculum] Phase {self.current_phase} Evaluation: Success Rate = {success_rate:.2f}")

            if success_rate >= self.success_threshold:
                if self.verbose > 0:
                    print(f"[Curriculum] Phase {self.current_phase} success threshold ({self.success_threshold}) reached!")
                
                # Export model for the finished phase
                self._export_phase_model()

                if self.current_phase >= self.max_phase:
                    if self.verbose > 0:
                        print("[Curriculum] All phases completed. Stopping training.")
                    return False # Stop training
                
                self.current_phase += 1
                self._update_phase_params()
                self._apply_phase_params()
                
        return True

    def _export_phase_model(self) -> None:
        """
        Export the current model to ONNX and PyTorch with phase-specific naming via the export callback.
        """
        if self.export_callback is None:
            if self.verbose > 0:
                print("[Curriculum] Warning: export_callback is None, skipping export.")
            return

        phase_name = f"phase_{self.current_phase}"
        if self.current_phase == self.max_phase:
            filename = f"model_{phase_name}_final"
        else:
            filename = f"model_{phase_name}"
            
        self.export_callback.trigger_export(filename=filename, model=self.model)

    def _evaluate_success_rate(self) -> float:
        """
        Run evaluation episodes and calculate the success rate.
        """
        successes = []
        eval_goals = np.linspace(0.1, 0.9, self.n_eval_episodes)
        
        for goal_alt in eval_goals:
            # Set a random goal and ensure current phase parameters are applied
            # Use fixed initial position [0.0, 0.0, 0.05] for evaluation
            self.eval_env.env_method('set_next_episode_params', 
                                     goal_alt=goal_alt,
                                     locked_axes=self.locked_axes.copy(),
                                     initial_pos=[0.0, 0.0, 0.05])
            
            obs = self.eval_env.reset()
            done = False
            episode_success = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # VecEnv.step returns (obs, reward, done, info)
                # where done is terminated | truncated
                obs, reward, dones, info = self.eval_env.step(action)
                done = dones[0]
                # Check for success in info. Since eval_env is vectorized, info is a list of dicts
                for i in info:
                    if i.get("is_success", False):
                        episode_success = True
            successes.append(float(episode_success))
        
        return np.mean(successes)

    def _update_phase_params(self) -> None:
        """
        Update the locked axes based on the current phase.
        Phase 1: roll, pitch, yaw locked
        Phase 2: pitch, yaw locked (roll unlocked)
        Phase 3: yaw locked (pitch unlocked)
        Phase 4: none locked (yaw unlocked)
        """
        if self.current_phase == 2:
            self.locked_axes = ['pitch', 'yaw']
        elif self.current_phase == 3:
            self.locked_axes = ['yaw']
        elif self.current_phase == 4:
            self.locked_axes = []
