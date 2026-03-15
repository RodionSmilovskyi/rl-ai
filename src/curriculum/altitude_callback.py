import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict, Any, Optional

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
        verbose: int = 1,
        onnx_export_callback: Optional[Any] = None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.onnx_export_callback = onnx_export_callback
        
        self.current_phase = 1
        self.max_phase = 4
        
        # Initial parameters for Phase 1
        self.locked_axes = ['roll', 'pitch', 'yaw']
        self.initial_pos = [0.0, 0.0, 0.05]
        
    def _on_training_start(self) -> None:
        """
        Called when training starts. Set initial environment parameters and goals.
        """
        self._apply_phase_params()
        # Set initial random goals for all training environments
        for i in range(self.training_env.num_envs):
            self.training_env.env_method('set_next_episode_params', 
                                         goal_alt=self._get_random_goal(), 
                                         indices=i)

    def _get_random_goal(self) -> float:
        """
        Generate a random normalized goal altitude.
        """
        return float(np.random.uniform(0.1, 0.9))

    def _apply_phase_params(self) -> None:
        """
        Apply the current phase's parameters to all vectorized environments.
        """
        if self.verbose > 0:
            print(f"\n[Curriculum] >>> Entering Phase {self.current_phase} <<<")
            print(f"[Curriculum] Locked axes: {self.locked_axes}")
            print(f"[Curriculum] Initial pos: {self.initial_pos}")

        # Update training environments
        self.training_env.env_method('set_next_episode_params', 
                                     locked_axes=self.locked_axes.copy(), 
                                     initial_pos=self.initial_pos.copy())
        
        # Update evaluation environment
        self.eval_env.env_method('set_next_episode_params', 
                                 locked_axes=self.locked_axes.copy(), 
                                 initial_pos=self.initial_pos.copy())

    def _on_step(self) -> bool:
        """
        Check if it's time to evaluate and potentially advance the curriculum.
        Also handle per-episode goal randomization for training environments.
        """
        # Randomize goal for the NEXT reset for any env that just finished an episode
        dones = self.locals.get("dones")
        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    self.training_env.env_method('set_next_episode_params', 
                                                 goal_alt=self._get_random_goal(), 
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
        Export the current model to ONNX with phase-specific naming via the OnnxExportCallback.
        """
        if self.onnx_export_callback is None:
            if self.verbose > 0:
                print("[Curriculum] Warning: onnx_export_callback is None, skipping export.")
            return

        phase_name = f"phase_{self.current_phase}"
        if self.current_phase == self.max_phase:
            filename = f"model_{phase_name}_final.onnx"
        else:
            filename = f"model_{phase_name}.onnx"
            
        self.onnx_export_callback.trigger_export(filename=filename, model=self.model)

    def _evaluate_success_rate(self) -> float:
        """
        Run evaluation episodes and calculate the success rate.
        """
        successes = []
        for _ in range(self.n_eval_episodes):
            # Set a random goal and ensure current phase parameters are applied
            random_goal = self._get_random_goal()
            self.eval_env.env_method('set_next_episode_params', 
                                     goal_alt=random_goal,
                                     locked_axes=self.locked_axes.copy(),
                                     initial_pos=self.initial_pos.copy())
            
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
