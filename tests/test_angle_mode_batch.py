import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from drone_env import DroneEnv
from settings import PHYSICS_FREQ

def run_scenario(env, roll_deg, pitch_deg, yaw_rate_deg, label):
    print(f"\n=== Scenario: {label} ===")
    obs, info = env.reset()
    
    from settings import TILT_LIMIT, MAX_YAW_RATE_RADS
    TILT_LIMIT_DEG = np.rad2deg(TILT_LIMIT)
    MAX_YAW_RATE_DEG = np.rad2deg(MAX_YAW_RATE_RADS)
    
    expected_roll = np.clip(roll_deg, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)
    expected_pitch = np.clip(pitch_deg, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)

    # Map targets
    desired_roll_norm = roll_deg / TILT_LIMIT_DEG
    desired_pitch_norm = pitch_deg / TILT_LIMIT_DEG
    desired_yaw_rate_norm = yaw_rate_deg / MAX_YAW_RATE_DEG
    
    rc_roll = 1500 + 500 * desired_roll_norm
    rc_pitch = 1500 + 500 * desired_pitch_norm
    rc_yaw = 1500 + 500 * desired_yaw_rate_norm
    
    # Very careful throttle to stay within 0.05-1.0m altitude
    throttle = 1450 if (abs(roll_deg) < 35 and abs(pitch_deg) < 35) else 1600
    rc_command = np.array([throttle, rc_roll, rc_pitch, rc_yaw], dtype=np.float32)
    
    print(f"Target: roll={roll_deg:5.1f}°, pitch={pitch_deg:5.1f}°, yaw_rate={yaw_rate_deg:5.1f}°/s")
    
    for i in range(int(PHYSICS_FREQ * 3)): # 3 seconds
        obs, reward, terminated, truncated, info = env.step(rc_command)
        r_deg = obs['roll'][0] * TILT_LIMIT_DEG
        p_deg = obs['pitch'][0] * TILT_LIMIT_DEG
        y_rate_deg = obs['yaw'][0] * MAX_YAW_RATE_DEG
        
        # Stop early if target reached and stable
        if i > 60 and abs(r_deg - expected_roll) < 2.0 and abs(p_deg - expected_pitch) < 2.0:
            if abs(yaw_rate_deg) < 1.0 or abs(y_rate_deg - yaw_rate_deg) < 10.0:
                break

        if terminated or truncated:
            break
            
    final_roll_deg = obs["roll"][0] * TILT_LIMIT_DEG
    final_pitch_deg = obs["pitch"][0] * TILT_LIMIT_DEG
    final_yaw_rate_deg = obs["yaw"][0] * MAX_YAW_RATE_DEG
    print(f"Reached: roll={final_roll_deg:6.2f}°, pitch={final_pitch_deg:6.2f}°, yaw_rate={final_yaw_rate_deg:6.2f}°/s, alt={obs['altitude'][0]:.2f}")
    
    assert abs(final_roll_deg - expected_roll) < 3.5
    assert abs(final_pitch_deg - expected_pitch) < 3.5
    if abs(yaw_rate_deg) > 1.0:
        assert abs(final_yaw_rate_deg - yaw_rate_deg) < 15.0

    # Reset for level-off test
    obs, info = env.reset()
    # Briefly tilt the drone
    for _ in range(60): env.step(rc_command)
    
    # Return to level
    rc_command_level = np.array([1450, 1500, 1500, 1500], dtype=np.float32)
    for _ in range(int(PHYSICS_FREQ * 2)):
        obs, reward, terminated, truncated, info = env.step(rc_command_level)
        if terminated or truncated: break
            
    final_roll_deg = obs["roll"][0] * TILT_LIMIT_DEG
    final_pitch_deg = obs["pitch"][0] * TILT_LIMIT_DEG
    print(f"Level:   roll={final_roll_deg:6.2f}°, pitch={final_pitch_deg:6.2f}°, alt={obs['altitude'][0]:.2f}")
    assert abs(final_roll_deg) < 2.0
    assert abs(final_pitch_deg) < 2.0

def main():
    env = DroneEnv(use_gui=False)
    
    scenarios = [
        (25.0, 0.0, 0.0, "Roll Only"),
        (0.0, 25.0, 0.0, "Pitch Only"),
        (0.0, 0.0, 90.0, "Yaw Rate Only"),
        (15.0, 15.0, 45.0, "Combined Motion")
    ]
    
    for r, p, y, label in scenarios:
        try:
            run_scenario(env, r, p, y, label)
        except AssertionError as e:
            print(f"FAILED: {e}")
            sys.exit(1)
            
    print("\nALL SCENARIOS PASSED!")
    env.close()

if __name__ == "__main__":
    main()
