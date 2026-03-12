import gymnasium as gym
import time
import numpy as np
from drone_env import DroneEnv

def main():
    # Create the environment with human render mode to see the GUI
    env = DroneEnv(render_mode="human", use_gui=True)
    
    obs, info = env.reset()
    print("Environment reset successful")
    
    # Run for a few steps with a simple hover-like command
    # rc_command: [throttle, roll, pitch, yaw] in [1000, 2000]
    # 1500 is neutral for roll, pitch, yaw.
    # We'll try a slightly higher throttle to see if it moves.
    
    for i in range(200):
        # Slightly above hover throttle (assuming ~1280 is hover for 280g drone with TWR 4)
        # Let's try 1400 to see some movement
        action = np.array([1400, 1500, 1500, 1500], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Altitude={obs['altitude'][0]:.4f}, Velocity_Z={info['vertical_velocity']:.4f}")
            
        if terminated or truncated:
            print("Episode finished")
            obs, info = env.reset()
            
    env.close()
    print("Test finished")

if __name__ == "__main__":
    main()
