import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import torch as th

from drone_wrappers import DroneHRLWrapper

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from drone_env import DroneEnv


def main():
    parser = argparse.ArgumentParser(description="Test an ONNX or PyTorch model for drone altitude control.")
    parser.add_argument("--model-path", type=str, default="output/model/model_phase_1.onnx", help="Path to the model (.onnx or .pt).")
    parser.add_argument("--goal-alt", type=float, default=0.5, help="Goal altitude for the test.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=24, help="Maximum steps per episode (sub-episodes for HRL).")
    parser.add_argument("--locked-axes", nargs="*", default=[], help="Axes to lock (roll, pitch, yaw).")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    is_onnx = args.model_path.endswith(".onnx")
    is_torch = args.model_path.endswith(".pt") or args.model_path.endswith(".pth")

    if not is_onnx and not is_torch:
        print(f"Error: Unsupported model format for {args.model_path}. Use .onnx, .pt, or .pth")
        sys.exit(1)

    # Initialize model
    if is_onnx:
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime not found. Please install it with 'pip install onnxruntime' or 'pip install onnxruntime-cpu'.")
            sys.exit(1)
            
        print(f"Loading ONNX model from {args.model_path}...")
        session = ort.InferenceSession(args.model_path)
        input_name = session.get_inputs()[0].name
        
        def predict(obs):
            obs_input = obs.astype(np.float32).reshape(1, -1)
            onnx_outputs = session.run(None, {input_name: obs_input})
            return onnx_outputs[0][0]
    else:
        print(f"Loading PyTorch model from {args.model_path}...")
        # Load the model
        model = th.jit.load(args.model_path)
        model.eval()
        
        def predict(obs):
            obs_input = th.from_numpy(obs.astype(np.float32)).reshape(1, -1)
            with th.no_grad():
                action = model(obs_input)
            return action.numpy()[0]

    # Initialize environment
    render_mode = "human" if args.render else None
    base_env = DroneEnv(render_mode=render_mode)
    env = DroneHRLWrapper(base_env, k_steps=20, sub_episode_limit=args.max_steps)

    for ep in range(args.episodes):
        print(f"Starting Episode {ep + 1} with Goal Altitude: {args.goal_alt}, Locked Axes: {args.locked_axes}")
        
        # Reset with specific goal altitude and locked axes
        obs, info = env.reset(options={"goal_alt": args.goal_alt, "locked_axes": args.locked_axes, "initial_pos": [0, 0, 0.05]})
        alt = obs[0]
        dist = abs(alt - args.goal_alt)
        total_reward = 0
        done = False
        step = 0
        print(f"Step {step}: Alt: {alt:.3f}, Goal: {args.goal_alt:.1f}, Dist: {dist:.3f}, Reward: 0")
        
        while not done and step < args.max_steps:
            # Run inference
            action = predict(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            done = terminated or truncated
            
            # Print current state
            alt = obs[0]
            dist = abs(alt - args.goal_alt)
            print(f"Step {step}: Alt: {alt:.3f}, Goal: {args.goal_alt:.1f}, Dist: {dist:.3f}, Reward: {reward:.3f}")
            
            if args.render:
                env.render()
        
        print(f"Episode {ep + 1} finished. Total Reward: {total_reward:.3f}, Final Alt: {obs[0]:.3f}")

    env.close()

if __name__ == "__main__":
    main()
