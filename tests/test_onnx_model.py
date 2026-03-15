import os
import sys
import argparse
import numpy as np
import gymnasium as gym

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from drone_env import DroneEnv
from altitude_curriculum_wrapper import AltitudeCurriculumWrapper

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found. Please install it with 'pip install onnxruntime' or 'pip install onnxruntime-cpu'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test an ONNX model for drone altitude control.")
    parser.add_argument("--model-path", type=str, default="output/model/model_phase_1.onnx", help="Path to the ONNX model.")
    parser.add_argument("--goal-alt", type=float, default=0.5, help="Goal altitude for the test.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=24, help="Maximum steps per episode (sub-episodes for HRL).")
    parser.add_argument("--locked-axes", nargs="*", default=[], help="Axes to lock (roll, pitch, yaw).")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    # Initialize ONNX runtime session
    print(f"Loading ONNX model from {args.model_path}...")
    session = ort.InferenceSession(args.model_path)
    input_name = session.get_inputs()[0].name

    # Initialize environment
    render_mode = "human" if args.render else None
    base_env = DroneEnv(render_mode=render_mode)
    env = AltitudeCurriculumWrapper(base_env, k_steps=20, sub_episode_limit=args.max_steps)
    env.set_next_episode_params(
        goal_alt=args.goal_alt, 
        locked_axes=args.locked_axes, 
        initial_pos=[0, 0, 0.05]
    )
    for ep in range(args.episodes):
        print(f"Starting Episode {ep + 1} with Goal Altitude: {args.goal_alt}, Locked Axes: {args.locked_axes}")
        
        # Reset with specific goal altitude and locked axes
        obs, info = env.reset(options={"goal_alt": args.goal_alt, "locked_axes": args.locked_axes})
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < args.max_steps:
            # Prepare input for ONNX (add batch dimension)
            obs_input = obs.astype(np.float32).reshape(1, -1)
            
            # Run inference
            # SB3 SAC ONNX export returns [action]
            onnx_outputs = session.run(None, {input_name: obs_input})
            action = onnx_outputs[0][0]
            
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
