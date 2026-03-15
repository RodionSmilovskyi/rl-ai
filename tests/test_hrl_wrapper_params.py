import os
import sys
import numpy as np
import gymnasium as gym

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from drone_env import DroneEnv
from drone_wrappers import DroneHRLWrapper

def test_hrl_wrapper_params():
    # Initialize environment
    env = DroneEnv(render_mode="rgb_array")
    wrapper = DroneHRLWrapper(env)
    
    print("--- Episode 1 (Defaults) ---")
    obs, info = wrapper.reset()
    print(f"Goal altitude: {wrapper.goal_alt}")
    
    # Queue changes for Episode 2
    print("\nQueueing changes for Episode 2: goal_alt=0.8, locked_axes=['roll', 'pitch', 'yaw'], initial_pos=[0.5, 0.5, 0.1]")
    wrapper.set_next_episode_params(
        goal_alt=0.8, 
        locked_axes=['roll', 'pitch', 'yaw'], 
        initial_pos=[0.5, 0.5, 0.1]
    )
    
    # Test step with action that should be zeroed
    action = np.array([0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    wrapper.step(action)
    
    print("\n--- Episode 2 (Applied) ---")
    obs, info = wrapper.reset()
    print(f"Goal altitude (expected 0.8): {wrapper.goal_alt}")
    assert abs(wrapper.goal_alt - 0.8) < 1e-5
    
    # Check if locked_axes are applied in the wrapper
    print(f"Locked axes (expected ['roll', 'pitch', 'yaw']): {wrapper.locked_axes}")
    assert wrapper.locked_axes == ['roll', 'pitch', 'yaw']
    
    # Check if initial_pos is applied in the underlying env
    print(f"Initial position (expected [0.5, 0.5, 0.1]): {env.initial_pos_np}")
    assert np.allclose(env.initial_pos_np, np.array([0.5, 0.5, 0.1]))
    
    # Test if actions are zeroed in Episode 2
    action_to_test = np.array([0.5, 0.7, 0.8, 0.9], dtype=np.float32)
    # We need to monkey patch or observe the action passed to env.step or just check if it was modified
    # Since action is passed by reference, it will be modified in place
    action_copy = action_to_test.copy()
    wrapper.step(action_to_test)
    print(f"Action after step (expected [0.5, 0.0, 0.0, 0.0]): {action_to_test}")
    assert abs(action_to_test[0] - 0.5) < 1e-6
    assert abs(action_to_test[1]) < 1e-6
    assert abs(action_to_test[2]) < 1e-6
    assert abs(action_to_test[3]) < 1e-6

    # Queue changes for Episode 3
    print("\nQueueing changes for Episode 3: goal_alt=0.2, locked_axes=['yaw'], initial_pos=[0, 0, 0.05]")
    wrapper.set_next_episode_params(
        goal_alt=0.2, 
        locked_axes=['yaw'], 
        initial_pos=[0.0, 0.0, 0.05]
    )

    wrapper.step(action)

    print("\n--- Episode 3 (Applied) ---")
    obs, info = wrapper.reset()
    print(f"Goal altitude (expected 0.2): {wrapper.goal_alt}")
    assert abs(wrapper.goal_alt - 0.2) < 1e-5

    print(f"Locked axes (expected ['yaw']): {wrapper.locked_axes}")
    assert wrapper.locked_axes == ['yaw']

    action_to_test_3 = np.array([0.5, 0.7, 0.8, 0.9], dtype=np.float32)
    wrapper.step(action_to_test_3)
    print(f"Action after step (expected [0.5, 0.7, 0.8, 0.0]): {action_to_test_3}")
    assert abs(action_to_test_3[0] - 0.5) < 1e-6
    assert abs(action_to_test_3[1] - 0.7) < 1e-6
    assert abs(action_to_test_3[2] - 0.8) < 1e-6
    assert abs(action_to_test_3[3]) < 1e-6
    
    print("\nTest passed successfully!")
    wrapper.close()

if __name__ == "__main__":
    test_hrl_wrapper_params()
