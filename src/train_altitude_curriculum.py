#!/usr/bin/env python
import os
import sys
import argparse
import random
from typing import Optional
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.monitor import Monitor
from common import ensure_directory
from drone_env import DroneEnv
from altitude_curriculum_wrapper import AltitudeCurriculumWrapper

class OnnxablePolicy(th.nn.Module):
    def __init__(self, actor: th.nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # NOTE: Post-processing (clipping/unscaling actions) is NOT included by default.
        # SAC actor output is usually squash by tanh.
        # SB3 SAC actor returns (action, log_std) during forward or just action if deterministic.
        # For inference, we use deterministic=True.
        return self.actor(observation, deterministic=True)

class OnnxExportCallback(BaseCallback):
    """
    Callback for exporting the model to ONNX format when a new best model is found.
    """
    def __init__(self, save_path: str, verbose: int = 0):
        super(OnnxExportCallback, self).__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        This method is called by the parent callback (EvalCallback) when a new best model is found.
        """
        if self.verbose > 0:
            print(f"Exporting new best model to ONNX: {self.save_path}")
        
        # Wrap the policy for ONNX export
        # model.policy.actor is the network we want for inference
        onnxable_model = OnnxablePolicy(self.model.policy.actor)
        
        # Define dummy input
        observation_size = self.model.observation_space.shape
        dummy_input = th.randn(1, *observation_size)
        
        # Export to ONNX
        th.onnx.export(
            onnxable_model,
            dummy_input,
            self.save_path,
            opset_version=15,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        return True

def make_env(rank: int, seed: int = 0, goal_alt: Optional[float] = None, render_mode: Optional[str] = "rgb_array"):
    def _init():
        env = DroneEnv(render_mode=render_mode)
        env = AltitudeCurriculumWrapper(env, k_steps=20, sub_episode_limit=24)
        # env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train(params):
    print(f"Initialized SummaryWriter at {params['tensorboard_dir']}")
    num_cpu = os.cpu_count() or 1
    print(f"Dynamically scaling training to {num_cpu} CPUs using SubprocVecEnv.")
    # Vectorized environments for Parallel SAC
    env = SubprocVecEnv([make_env(i, params["seed"]) for i in range(num_cpu)])
    env = VecMonitor(env)
    
    # Evaluation environment
    # Each evaluation episode record video
    eval_video_dir = os.path.join(params["output_dir"], "eval_videos")
    ensure_directory(eval_video_dir)
    
    # We use a single environment for evaluation to make video recording easier
    eval_env = SubprocVecEnv([make_env(0, params["seed"] + 1000, render_mode="rgb_array")])
    eval_env = VecMonitor(eval_env)
    
    # Wrap eval_env in VecVideoRecorder
    # record_video_trigger=lambda x: True means it records every episode in this env
    eval_env = VecVideoRecorder(
        eval_env, 
        eval_video_dir, 
        record_video_trigger=lambda x: True, 
        video_length=200,
        name_prefix="eval_altitude"
    )
    
    # Callback for ONNX export on new best
    onnx_path = os.path.join(params["model_dir"], "best_model.onnx")
    onnx_callback = OnnxExportCallback(onnx_path, verbose=1)
    
    # Callback to stop training when reward threshold is reached
    # The user wants to train until avg evaluation return reaches 1.0
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.1, verbose=1)
    
    # Combine callbacks to be executed when a new best model is found
    callback_on_best = CallbackList([onnx_callback, stop_callback])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=params["model_dir"],
        log_path=params["output_dir"],
        eval_freq=max(2000 // num_cpu, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=params["lr"],
        batch_size=params["batch_size"],
        seed=params["seed"],
        verbose=1,
        tensorboard_log=params["tensorboard_dir"],
    )

    print(f"Starting training for {params['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=params["total_timesteps"],
        callback=[eval_callback]
    )
    print("Training complete.")

    model.save(os.path.join(params["model_dir"], "final_model.zip"))
    
    # Close evaluation environment to flush any remaining video recordings
    eval_env.close()
    env.close()

    # Final video recording
    # print("Recording final evaluation video...")
    # final_video_dir = os.path.join(params["output_dir"], "final_video")
    # ensure_directory(final_video_dir)
    
    # final_video_env = SubprocVecEnv([make_env(0, params["seed"] + 2000)])
    # final_video_env = VecMonitor(final_video_env)
    # final_video_env = VecVideoRecorder(
    #     final_video_env, 
    #     final_video_dir, 
    #     record_video_trigger=lambda x: x == 0, 
    #     video_length=1000,
    #     name_prefix="final_eval_altitude"
    # )
    
    # obs = final_video_env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = final_video_env.step(action)
    # final_video_env.close()
    
    # display.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="sac-altitude")
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sm_output_dir = os.environ.get("SM_OUTPUT_DIR", "output")
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "checkpoints/sac-altitude")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/sac-altitude")
    tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "output/tensorboard")

    for d in [sm_output_dir, sm_model_dir, checkpoint_dir, tensorboard_dir]:
        ensure_directory(d)

    train({
        "prefix": args.prefix,
        "output_dir": sm_output_dir,
        "model_dir": sm_model_dir,
        "checkpoint_dir": checkpoint_dir,
        "tensorboard_dir": tensorboard_dir,
        "batch_size": args.batch_size,
        "total_timesteps": args.total_timesteps,
        "lr": args.lr,
        "seed": args.seed,
    })
