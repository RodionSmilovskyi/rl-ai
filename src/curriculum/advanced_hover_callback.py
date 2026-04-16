import numpy as np
import torch as th
from typing import Any, Optional

from curriculum.basic_hover_callback import BasicHoverCallback

class AdvancedHoverCallback(BasicHoverCallback):
    """
    Custom callback that extends BasicHoverCallback but uses a stricter
    success evaluation criterion (requires total reward >= 20 over 20 episodes)
    and restricts training to a single phase.
    """
    def __init__(
        self,
        eval_env: Any,
        success_threshold: float = 20.0,
        eval_freq: int = 2000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
        export_callback: Optional[Any] = None,
    ):
        super().__init__(
            eval_env=eval_env,
            success_threshold=success_threshold,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            max_phase=1,
            verbose=verbose,
            export_callback=export_callback
        )
        # Advanced hover has no locked axes. It starts directly with everything unlocked.
        self.locked_axes = []
        self.current_phase = 1

    def _evaluate_success_rate(self) -> float:
        """
        Run evaluation episodes and calculate the total reward across all of them.
        Returns the sum of rewards.
        """
        total_reward = 0.0
        
        for _ in range(self.n_eval_episodes):
            start_alt, goal_alt = self._get_random_start_and_goal()
            
            # Set random start and goal and ensure current phase parameters are applied
            self.eval_env.env_method('set_next_episode_params', 
                                     goal_alt=goal_alt,
                                     locked_axes=self.locked_axes.copy(),
                                     initial_pos=[0.0, 0.0, start_alt])
            
            obs = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # VecEnv.step returns (obs, reward, dones, info)
                obs, rewards, dones, info = self.eval_env.step(action)
                done = dones[0]
                total_reward += rewards[0]
        
        return total_reward / self.n_eval_episodes
