#!/usr/bin/env python
import os
import sys
import argparse
import random
import glob
import json
from typing import Any, Optional
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.monitor import Monitor
from common import ensure_directory
from curriculum.advanced_hover_callback import AdvancedHoverCallback
from export_utils import SACOnnxablePolicy, SACExportCallback
from env_utils import make_drone_env
from settings import K_STEPS, SUB_EPISODE_LIMIT

class ExpertDataset(Dataset):
    def __init__(self, data_dir):
        self.observations = []
        self.actions = []
        
        json_files = glob.glob(os.path.join(data_dir, "expert_episode_*.json"))
        if not json_files:
            print(f"Warning: No expert data found in {data_dir}")
            return
            
        print(f"Loading {len(json_files)} expert episodes from {data_dir}...")
        for file_path in json_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                for step_data in data:
                    if "action" in step_data:
                        self.observations.append([float(x) for x in step_data["obs"]])
                        self.actions.append([float(x) for x in step_data["action"]])
        
        self.observations = th.tensor(self.observations, dtype=th.float32)
        self.actions = th.tensor(self.actions, dtype=th.float32)
        print(f"Loaded {len(self.observations)} samples.")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

def train(params):
    print(f"Initialized SummaryWriter at {params['tensorboard_dir']}")
    num_cpu = params.get("num_cpus") or os.cpu_count() or 1
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
        record_video_trigger=lambda x: True,
        video_length=SUB_EPISODE_LIMIT * K_STEPS,
        name_prefix="eval_advanced_hover"
    )

    # Callback for ONNX and PyTorch export on new best
    export_callback = SACExportCallback(model_dir=params["model_dir"], verbose=1)

    # Curriculum Callback
    # This callback manages the dynamic altitude curriculum and stops training when finished
    curriculum_callback = AdvancedHoverCallback(
        eval_env=eval_env,
        success_threshold=params["success_threshold"],
        eval_freq=max(2000 // num_cpu, 1),
        n_eval_episodes=20,
        verbose=1,
        export_callback=export_callback
    )
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"--- Hardware Check: Training on device: {device} ---")

    print("Starting training from scratch...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=params["lr"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        train_freq=(params["train_freq"], "step"),
        gradient_steps=params["gradient_steps"],
        seed=params["seed"],
        verbose=1,
        device=device,
        tensorboard_log=params["tensorboard_dir"],
    )

    # 1. Behavior Cloning Pretraining (Optional)
    if params.get("bc_data_dir") and os.path.exists(params["bc_data_dir"]):
        print(f"Starting Behavior Cloning pretraining using data from {params['bc_data_dir']}...")
        dataset = ExpertDataset(params["bc_data_dir"])
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=params["bc_batch_size"], shuffle=True)
            actor = model.policy.actor
            optimizer = th.optim.Adam(actor.parameters(), lr=params["lr"])
            criterion = th.nn.MSELoss()
            
            actor.train()
            for epoch in range(params["bc_epochs"]):
                losses = []
                for obs, target_actions in loader:
                    obs, target_actions = obs.to(device), target_actions.to(device)
                    optimizer.zero_grad()
                    pred_actions = actor(obs)
                    loss = criterion(pred_actions, target_actions)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                if (epoch + 1) % 10 == 0:
                    print(f"BC Epoch {epoch+1}/{params['bc_epochs']}, Loss: {np.mean(losses):.6f}")
            print("Behavior Cloning pretraining finished.")
        else:
            print("Expert dataset is empty, skipping BC.")

    # 2. Critic Warmup (Optional)
    if params.get("critic_warmup_steps", 0) > 0:
        print(f"Starting Critic warmup for {params['critic_warmup_steps']} steps...")
        # Freeze the actor for warm-up
        for param in model.policy.actor.parameters():
            param.requires_grad = False
        
        model.learn(total_timesteps=params["critic_warmup_steps"])
        
        # Unfreeze the actor for fine-tuning
        for param in model.policy.actor.parameters():
            param.requires_grad = True
        print("Critic warmup finished.")

    # 3. Main Training Loop
    print(f"Starting main training for {params['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=params["total_timesteps"],
        callback=[curriculum_callback],
        reset_num_timesteps=False
    )
    print("Training complete.")
   
    # Close evaluation environment to flush any remaining video recordings
    eval_env.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="sac-advanced-hover")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-cpus", type=int, default=None, help="Number of CPUs to use")
    
    # New arguments for BC and fine-tuned SAC
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--train-freq", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=64)
    parser.add_argument("--critic-warmup-steps", type=int, default=0)
    parser.add_argument("--bc-data-dir", type=str, default=None)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--success-threshold", type=float, default=25.0)
    
    args = parser.parse_args()

    sm_hps_str = os.environ.get("SM_HPS", "{}")
    sm_hps = json.loads(sm_hps_str)
    if sm_hps:
        if "total-timesteps" in sm_hps: args.total_timesteps = int(sm_hps["total-timesteps"])
        if "seed" in sm_hps: args.seed = int(sm_hps["seed"])
        if "lr" in sm_hps: args.lr = float(sm_hps["lr"])
        if "batch-size" in sm_hps: args.batch_size = int(sm_hps["batch-size"])
        if "prefix" in sm_hps: args.prefix = sm_hps["prefix"]
        if "num-cpus" in sm_hps: args.num_cpus = int(sm_hps["num-cpus"])
        if "ent-coef" in sm_hps: args.ent_coef = float(sm_hps["ent-coef"])
        if "train-freq" in sm_hps: args.train_freq = int(sm_hps["train-freq"])
        if "gradient-steps" in sm_hps: args.gradient_steps = int(sm_hps["gradient-steps"])
        if "critic-warmup-steps" in sm_hps: args.critic_warmup_steps = int(sm_hps["critic-warmup-steps"])
        if "bc-data-dir" in sm_hps: args.bc_data_dir = sm_hps["bc-data-dir"]
        if "bc-epochs" in sm_hps: args.bc_epochs = int(sm_hps["bc-epochs"])
        if "bc-batch-size" in sm_hps: args.bc_batch_size = int(sm_hps["bc-batch-size"])
        if "success-threshold" in sm_hps: args.success_threshold = float(sm_hps["success-threshold"])

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
        "num_cpus": args.num_cpus,
        "ent_coef": args.ent_coef,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "critic_warmup_steps": args.critic_warmup_steps,
        "bc_data_dir": args.bc_data_dir,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "success_threshold": args.success_threshold,
    })
