#!/usr/bin/env python
import os
import sys
import argparse
import random
from typing import Any, Optional
import numpy as np
import torch as th

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecMonitor
from common import ensure_directory
from curriculum.altitude_callback import AltitudeCurriculumCallback
from export_utils import PPOOnnxablePolicy, PPOExportCallback
from env_utils import make_drone_env

def train(params):
    print(f"Initialized SummaryWriter at {params['tensorboard_dir']}")
    num_cpu = os.cpu_count() or 1
    print(f"Dynamically scaling training to {num_cpu} CPUs using SubprocVecEnv.")
    
    # Vectorized environments for Parallel PPO
    env = SubprocVecEnv([make_drone_env(i, params["seed"]) for i in range(num_cpu)])
    env = VecMonitor(env)

    # Evaluation environment
    # Each evaluation episode record video
    eval_video_dir = os.path.join(params["output_dir"], "data", "eval_videos")
    ensure_directory(eval_video_dir)

    # We use a single environment for evaluation to make video recording easier
    eval_env = SubprocVecEnv([make_drone_env(0, params["seed"] + 1000, render_mode="rgb_array")])    
    eval_env = VecMonitor(eval_env)
    
    # Wrap eval_env in VecVideoRecorder
    eval_env = VecVideoRecorder(
        eval_env, 
        eval_video_dir, 
        record_video_trigger=lambda x: x == 0, 
        video_length=200,
        name_prefix="eval_altitude_ppo"
    )
    
    # Callback for ONNX and PyTorch export on new best
    export_callback = PPOExportCallback(model_dir=params["model_dir"], verbose=1)
    
    # Curriculum Callback
    curriculum_callback = AltitudeCurriculumCallback(
        eval_env=eval_env,
        success_threshold=0.8,
        eval_freq=max(2000 // num_cpu, 1),
        n_eval_episodes=10,
        max_phase=params.get("max_phase", 4),
        verbose=1,
        export_callback=export_callback
    )
    
    # PPO Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params["lr"],
        n_steps=2048,
        batch_size=params["batch_size"],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=params["seed"],
        verbose=1,
        tensorboard_log=params["tensorboard_dir"],
    )

    print(f"Starting training for {params['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=params["total_timesteps"],
        callback=[curriculum_callback]
    )
    print("Training complete.")
   
    # Close evaluation environment to flush any remaining video recordings
    eval_env.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="ppo-altitude")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64) # PPO often uses smaller batches
    parser.add_argument("--max-phase", type=int, default=4)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sm_output_dir = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output")
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints")
    tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "/opt/ml/output/tensorboard")

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
        "max_phase": args.max_phase,
    })
