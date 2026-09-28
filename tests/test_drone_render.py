import os
import sys
import argparse
import gymnasium as gym
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from drone_env import DroneEnv
from drone_hrl_wrapper import DroneHRLWrapper
from settings import SUB_EPISODE_LIMIT, MAX_ALTITUDE

def main():
    parser = argparse.ArgumentParser(description="Test drone visualization and rendering.")
    parser.add_argument("--goal-alt", type=float, default=0.5, help="Goal altitude in meters (default: 0.5m)")
    parser.add_argument("--start-alt", type=float, default=0.1, help="Start altitude in meters (default: 0.1m)")
    parser.add_argument("--max-steps", type=int, default=120, help="Number of HRL sub-episodes (default: 120 = ~10s)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    # Create the base environment
    use_gui = not args.headless
    render_mode = "human" if use_gui else None
    base_env = DroneEnv(render_mode=render_mode, use_gui=use_gui)
    
    goal_alt = args.goal_alt
    start_alt = args.start_alt
    initial_pos = [0.5, 0.5, start_alt]

    # Wrap with HRL wrapper
    env = DroneHRLWrapper(base_env, sub_episode_limit=args.max_steps)
    env.set_next_episode_params(        
        goal_alt=goal_alt, 
        locked_axes=['roll', 'pitch', 'yaw'], 
        initial_pos=initial_pos
    )
    obs, info = env.reset()
    print(f"Environment reset successful. Goal Altitude: {goal_alt}m, Start Altitude: {start_alt}m")
    print(f"Initial Observation (7D: alt, front_dist, sx, sy, vx, vy, goal): {obs}")
    
    # Map desired altitude (0..MAX_ALTITUDE) to high-level action [-1, 1]
    goal_alt_norm = np.clip(goal_alt / MAX_ALTITUDE, 0.0, 1.0)
    action_alt = float(goal_alt_norm * 2.0 - 1.0)
    # High-level action: [desired_alt_action, desired_roll, desired_pitch, desired_yaw_rate]
    action = np.array([action_alt, 0.0, 0.0, 0.0], dtype=np.float32)

    # Run for specified sub-episodes
    for i in range(args.max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        
        alt_m = obs[0] * MAX_ALTITUDE
        front_m = obs[1] * MAX_ALTITUDE
        
        if i % 10 == 0 or i < 5:
            print(f"Sub-Episode {i:3d}: Alt={alt_m:.3f}m, Front={front_m:.3f}m, Reward={reward:.4f}")
            print(f"  Obs (7D): {obs}")
            
        if terminated or truncated:
            print(f"Episode finished at step {i}. Reason: {'Terminated' if terminated else 'Truncated'}")
            obs, info = env.reset(options={"goal_alt": goal_alt, "initial_pos": initial_pos})
            
    env.close()
    print("Test finished successfully.")

if __name__ == "__main__":
    main()
