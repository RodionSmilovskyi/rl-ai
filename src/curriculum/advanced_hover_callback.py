import numpy as np
import torch as th
from typing import Any, Optional

from curriculum.altitude_callback import AltitudeCurriculumCallback

class AdvancedHoverCallback(AltitudeCurriculumCallback):
    """
    Custom callback that extends AltitudeCurriculumCallback but uses a stricter
    success evaluation criterion (requires at least 5 successful steps in an episode)
    and restricts training to a single phase.
    """
    def __init__(
        self,
        eval_env: Any,
        success_threshold: float = 0.8,
        eval_freq: int = 2000,
        n_eval_episodes: int = 10,
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

    def _on_training_start(self) -> None:
        super()._on_training_start()
        
        # Reset entropy coefficient and optimizer for SAC
        if hasattr(self.model, "log_ent_coef"):
            if self.verbose > 0:
                print(f"[Curriculum] Resetting entropy coefficient and optimizer on training start")
                
            with th.no_grad():
                # Setting log(1.0) = 0.0. This forces the entropy multiplier back to 1.0 (max exploration)
                self.model.log_ent_coef.fill_(0.0) 
            
            # CRITICAL: Clear the Adam optimizer's momentum state!
            if hasattr(self.model, "ent_coef_optimizer"):
                self.model.ent_coef_optimizer.state.clear()
        
    def _evaluate_success_rate(self, tail_window: int = 6, required_successes: int = 4) -> float:
        """
        Run evaluation episodes and calculate the success rate.
        An episode is considered successful only if it has at least 5 successful steps.
        """
        successes = []
        eval_goals = np.linspace(0.1, 0.9, self.n_eval_episodes)
        np.random.shuffle(eval_goals)
        
        for goal_alt in eval_goals:
            # Set a random goal and ensure current phase parameters are applied
            # Use fixed initial position [0.0, 0.0, 0.05] for evaluation
            self.eval_env.env_method('set_next_episode_params', 
                                     goal_alt=goal_alt,
                                     locked_axes=self.locked_axes.copy(),
                                     initial_pos=[0.0, 0.0, 0.05])
            
            obs = self.eval_env.reset()
            done = False
            success_steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # VecEnv.step returns (obs, reward, dones, info)
                obs, reward, dones, info = self.eval_env.step(action)
                done = dones[0]
                
                # Check for success in info. Since eval_env is vectorized, info is a list of dicts
                for i in info:
                    if i.get("is_success", False):
                        success_steps += 1
                        
            # Episode is successful only if it has at least 5 successful steps
            episode_success = success_steps >= 5
            successes.append(float(episode_success))
        
        return np.mean(successes)
