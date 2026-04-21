import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from drone_env import DroneEnv
from settings import PHYSICS_FREQ

def test_angle_mode_stabilization():
    env = DroneEnv(use_gui=True)
    obs, info = env.reset()

    # --- SET DESIRED TARGETS HERE ---
    desired_roll_deg = 0.0
    desired_pitch_deg = 21.0
    desired_yaw_rate_deg = 20
    # --------------------------------

    from settings import TILT_LIMIT, MAX_YAW_RATE_RADS
    TILT_LIMIT_DEG = np.rad2deg(TILT_LIMIT)
    MAX_YAW_RATE_DEG = np.rad2deg(MAX_YAW_RATE_RADS)
    
    print(f"TILT_LIMIT is {TILT_LIMIT_DEG:.2f}°")
    
    # Check if it's close to targets
    expected_roll = np.clip(desired_roll_deg, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)
    expected_pitch = np.clip(desired_pitch_deg, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)

    # Map targets to [-1, 1] normalized range
    desired_roll_norm = desired_roll_deg / TILT_LIMIT_DEG
    desired_pitch_norm = desired_pitch_deg / TILT_LIMIT_DEG
    desired_yaw_rate_norm = desired_yaw_rate_deg / MAX_YAW_RATE_DEG
    
    # Map normalized values to RC [1000, 2000]
    rc_roll = 1500 + 500 * desired_roll_norm
    rc_pitch = 1500 + 500 * desired_pitch_norm
    rc_yaw = 1500 + 500 * desired_yaw_rate_norm
    
    # rc_command: [throttle, roll, pitch, yaw]
    # Use 1450 for small angles, 1600 only for very large ones
    throttle = 1450 if (abs(desired_roll_deg) < 30 and abs(desired_pitch_deg) < 30) else 1600
    rc_command = np.array([throttle, rc_roll, rc_pitch, rc_yaw], dtype=np.float32)
    
    print(f"Requested: roll={desired_roll_deg}°, pitch={desired_pitch_deg}°, yaw_rate={desired_yaw_rate_deg}°/s")
    print(f"Using throttle: {throttle}")
    
    for i in range(int(PHYSICS_FREQ * 3)): # 3 seconds
        obs, reward, terminated, truncated, info = env.step(rc_command)
        r_deg = obs['roll'][0] * TILT_LIMIT_DEG
        p_deg = obs['pitch'][0] * TILT_LIMIT_DEG
        alt = obs['altitude'][0]
        
        if i % 120 == 0: # Print every 0.5s
            print(f"  Step {i:4d}: roll={r_deg:6.2f}°, pitch={p_deg:6.2f}°, alt={alt:.4f}")
            
        # Stop early if target reached and stable
        if i > 60 and abs(r_deg - expected_roll) < 1.0 and abs(p_deg - expected_pitch) < 1.0:
            print(f"Target reached at step {i}")
            break

        if terminated or truncated:
            print(f"Terminated at step {i}. Reason: {'Ceiling/Floor/Tilt' if terminated else 'Timeout'}")
            break
            
    final_roll_deg = obs["roll"][0] * TILT_LIMIT_DEG
    final_pitch_deg = obs["pitch"][0] * TILT_LIMIT_DEG
    print(f"Final: roll={final_roll_deg:.2f}°, pitch={final_pitch_deg:.2f}°")
    
    assert abs(final_roll_deg - expected_roll) < 2.0, f"Expected roll {expected_roll}, got {final_roll_deg}"
    assert abs(final_pitch_deg - expected_pitch) < 2.0, f"Expected pitch {expected_pitch}, got {final_pitch_deg}"
    
    if terminated:
        print("Re-resetting for level test...")
        obs, info = env.reset()

    # Now return to level
    rc_command = np.array([1450, 1500, 1500, 1500], dtype=np.float32)
    print("\nReturning to level...")
    for i in range(int(PHYSICS_FREQ * 2)):
        obs, reward, terminated, truncated, info = env.step(rc_command)
        if i % 120 == 0: # Print every 0.5s
            r_deg = obs['roll'][0] * TILT_LIMIT_DEG
            p_deg = obs['pitch'][0] * TILT_LIMIT_DEG
            print(f"  Step {i:4d}: roll={r_deg:6.2f}°, pitch={p_deg:6.2f}°")
        if terminated or truncated:
            break
            
    final_roll_deg = obs["roll"][0] * TILT_LIMIT_DEG
    final_pitch_deg = obs["pitch"][0] * TILT_LIMIT_DEG
    print(f"Final Level: roll={final_roll_deg:.2f}°, pitch={final_pitch_deg:.2f}°")
    assert abs(final_roll_deg) < 1.0
    assert abs(final_pitch_deg) < 1.0

    
    env.close()

if __name__ == "__main__":
    test_angle_mode_stabilization()
