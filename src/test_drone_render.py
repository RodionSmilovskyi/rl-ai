import gymnasium as gym
import numpy as np
from drone_env import DroneEnv
from drone_wrappers import DroneHRLWrapper
from settings import SUB_EPISODE_LIMIT

def main():
    # Create the base environment
    base_env = DroneEnv(render_mode="human", use_gui=True)
    
    # Wrap with HRL wrapper
    # Uses defaults from settings.py (K_STEPS, SUB_EPISODE_LIMIT, etc.)
    env = DroneHRLWrapper(base_env)
    
    goal_alt = 0.5
    start_alt = 0.5
    obs, info = env.reset(options={"goal_alt": goal_alt, "start_alt": start_alt})
    print(f"Environment reset successful. Goal Altitude: {goal_alt}, Start Altitude: {start_alt}")
    print(f"Initial Observation: {obs}")
    
    # Run for some sub-episodes
    for i in range(SUB_EPISODE_LIMIT):
        # High-level action: [desired_alt, desired_roll, desired_pitch, desired_yaw_rate]
        action = np.array([goal_alt, 0.0, 0.0, 0.0], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 1 == 0:
            print(f"Sub-Episode {i}:")
            print(f"  Obs (alt, sx, sy, vx, vy, goal): {obs}")
            print(f"  Reward: {reward:.4f}")
            
        if terminated or truncated:
            print(f"Episode finished. Reason: {'Terminated' if terminated else 'Truncated'}")
            obs, info = env.reset(options={"goal_alt": goal_alt, "start_alt": start_alt})
            
    env.close()
    print("Test finished")

if __name__ == "__main__":
    main()
