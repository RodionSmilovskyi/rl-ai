#!/usr/bin/env python
import os
import sys
import argparse
import random
import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from common import make_env, ensure_directory


def train(params):
    print(f"Initialized SummaryWriter at {params['tensorboard_dir']}")

    env_id = params["env_id"]
    num_cpu = os.cpu_count() or 1
    print(f"Dynamically scaling PPO to {num_cpu} CPUs using SubprocVecEnv.")
    
    # Implementing Parallel PPO logic via vectorized environments
    env = SubprocVecEnv([make_env(env_id, params["seed"], i) for i in range(num_cpu)])
    eval_env = gym.make(env_id, render_mode="rgb_array")
    
    # Replicating evaluation frequency from object-detection
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=params["model_dir"],
        log_path=params["output_dir"],
        eval_freq=max(2000 // num_cpu, 1),
        deterministic=True,
        render=False,
    )

    # Standard PPO instantiation using MlpPolicy for continuous control tasks like Pendulum
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params["lr"],
        n_steps=params["n_steps"],
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

    model.save(os.path.join(params["model_dir"], "final_model_ppo.zip"))

    # Recording final evaluation video as per FR5
    print("Recording evaluation video to output/videos...")
    video_dir = os.path.join(params["output_dir"], "videos")
    ensure_directory(video_dir)
    
    video_env = gym.make(env_id, render_mode="rgb_array")
    video_env = gym.wrappers.RecordVideo(video_env, video_folder=video_dir, name_prefix="final_eval_ppo")
    
    obs, info = video_env.reset(seed=params["seed"])
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        if terminated or truncated:
            break
    video_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="ppo-pendulum")
    parser.add_argument("--total-timesteps", type=int, default=100000) # PPO often needs more steps than SAC
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4) # PPO default LR is usually lower
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    T.manual_seed(args.seed)

    # Replicating SageMaker/Local path logic from object-detection/src/train.py
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
        "n_steps": args.n_steps,
        "total_timesteps": args.total_timesteps,
        "lr": args.lr,
        "env_id": args.env_id,
        "seed": args.seed,
    })
    sys.exit(0)
