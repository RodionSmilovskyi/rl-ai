#!/usr/bin/env python
import os
import sys
import argparse
import random
import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from common import make_env, ensure_directory

class CustomLoggingCallback(BaseCallback):
    """
    Maps SB3 metrics to the object-detection style logging format (e.g., train/loss, eval/ep_rew_mean).
    """
    def __init__(self, writer, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.last_logged = {}

    def _on_step(self) -> bool:
        if self.logger is not None:
            for key, value in self.logger.name_to_value.items():
                if not isinstance(value, (int, float, np.number)):
                    continue
                if key not in self.last_logged or self.last_logged[key] != value:
                    # Replicate object-detection grouping: train/ and eval/
                    if "loss" in key.lower() or "train" in key:
                        metric_name = key.split("/")[-1]
                        self.writer.add_scalar(f"train/{metric_name}", float(value), self.num_timesteps)
                    elif "rollout" in key or "eval" in key:
                        metric_name = key.split("/")[-1]
                        self.writer.add_scalar(f"eval/{metric_name}", float(value), self.num_timesteps)
                    self.last_logged[key] = value
        return True

def train(params):
    writer = SummaryWriter(os.path.join(params["tensorboard_dir"], params["prefix"]))
    print(f"Initialized SummaryWriter at {params['tensorboard_dir']}")

    env_id = params["env_id"]
    num_cpu = os.cpu_count() or 1
    print(f"Dynamically scaling ParallelSAC to {num_cpu} CPUs using SubprocVecEnv.")
    
    # Implementing ParallelSAC logic via vectorized environments
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

    custom_logger = CustomLoggingCallback(writer)
    
    # Standard SAC instantiation using MlpPolicy for continuous control tasks like Pendulum
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=params["lr"],
        batch_size=params["batch_size"],
        seed=params["seed"],
        verbose=0
    )

    print(f"Starting training for {params['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=params["total_timesteps"],
        callback=[eval_callback, custom_logger]
    )
    print("Training complete.")

    model.save(os.path.join(params["model_dir"], "final_model.zip"))

    # Recording final evaluation video as per FR5
    print("Recording evaluation video to output/videos...")
    video_dir = os.path.join(params["output_dir"], "videos")
    ensure_directory(video_dir)
    
    video_env = gym.make(env_id, render_mode="rgb_array")
    video_env = gym.wrappers.RecordVideo(video_env, video_folder=video_dir, name_prefix="final_eval")
    
    obs, info = video_env.reset(seed=params["seed"])
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        if terminated or truncated:
            break
    video_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="sac-pendulum")
    parser.add_argument("--total-timesteps", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
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
        "total_timesteps": args.total_timesteps,
        "lr": args.lr,
        "env_id": args.env_id,
        "seed": args.seed,
    })
    sys.exit(0)
