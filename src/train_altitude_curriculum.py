#!/usr/bin/env python
import os
import sys
import argparse
import random
from typing import Any, Optional
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.monitor import Monitor
from common import ensure_directory
from curriculum.altitude_callback import AltitudeCurriculumCallback
from export_utils import OnnxablePolicy, ExportCallback
from env_utils import make_drone_env

def train(params):
    print(f"Initialized SummaryWriter at {params['tensorboard_dir']}")
    num_cpu = os.cpu_count() or 1
    print(f"Dynamically scaling training to {num_cpu} CPUs using SubprocVecEnv.")
    # Vectorized environments for Parallel SAC
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
    # record_video_trigger=lambda x: True means it records every episode in this env
    eval_env = VecVideoRecorder(
        eval_env, 
        eval_video_dir, 
        record_video_trigger=lambda x: x == 0, 
        video_length=200,
        name_prefix="eval_altitude"
    )
    
    # Callback for ONNX and PyTorch export on new best
    export_callback = ExportCallback(model_dir=params["model_dir"], verbose=1)
    
    # Curriculum Callback
    # This callback manages the dynamic altitude curriculum and stops training when finished
    curriculum_callback = AltitudeCurriculumCallback(
        eval_env=eval_env,
        success_threshold=0.6,
        eval_freq=max(2000 // num_cpu, 1),
        n_eval_episodes=10,
        verbose=1,
        export_callback=export_callback
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
        callback=[curriculum_callback]
    )
    print("Training complete.")
   
    # Close evaluation environment to flush any remaining video recordings
    eval_env.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="sac-altitude")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
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
    })
