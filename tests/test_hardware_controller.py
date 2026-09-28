import os
import sys
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pid_controller import PIDController
from flight_controller import FlightController
from drone_env import DroneEnv
from drone_hrl_wrapper import DroneHRLWrapper
from settings import (
    MAX_ALTITUDE, GROUND_THRESHOLD, MIN_ALTITUDE, HOVER_THROTTLE_PWM,
    MAX_THROTTLE, DRONE_WEIGHT, G, PHYSICS_FREQ
)


def test_pid_controller_derivative_on_measurement():
    """Verify that changing setpoint does not cause a derivative kick."""
    pid = PIDController(Kp=5.0, Ki=0.0, Kd=2.0, setpoint=0.0)
    
    # First measurement at 0.0 -> derivative should be 0
    out1 = pid.compute(0.0, dt=0.01)
    assert out1 == pytest.approx(0.0)
    assert pid.derivative == 0.0

    # Step the setpoint from 0.0 to 1.0 without changing measurement
    pid.setpoint = 1.0
    out2 = pid.compute(0.0, dt=0.01)
    # Derivative should still be 0 (no setpoint kick!)
    assert pid.derivative == 0.0
    assert out2 == pytest.approx(5.0 * 1.0)


def test_pid_controller_integral_clamping_and_enable():
    """Verify integral is bounded by [integral_min, integral_limit] and respects enable_integral."""
    pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0, setpoint=1.0, integral_limit=1.2, integral_min=-0.35)
    
    # Accumulate positive integral past the limit
    for _ in range(200):
        pid.compute(0.0, dt=0.05, enable_integral=True)
    assert pid.integral == pytest.approx(1.2)

    # Accumulate negative integral past the minimum
    pid.setpoint = -1.0
    for _ in range(200):
        pid.compute(0.0, dt=0.05, enable_integral=True)
    assert pid.integral == pytest.approx(-0.35)

    # When enable_integral is False, integral should freeze
    frozen_val = pid.integral
    pid.setpoint = 1.0
    pid.compute(0.0, dt=0.05, enable_integral=False)
    assert pid.integral == frozen_val


def test_flight_controller_hover_and_asymmetric_bounds():
    """Verify hover baseline 1550 and asymmetrical authority [-75, +130]."""
    fc = FlightController()
    assert fc.hover_throttle == 1550
    assert fc.min_throttle == 1341
    assert fc.max_throttle == 1800
    assert fc.max_descent_correction == -75.0
    assert fc.max_climb_correction == 130.0

    # Hover equilibrium test: current_alt == desired_alt
    # desired_alt = 0.5m (norm = 0.5/3.0). High-level action = 2*norm - 1
    norm_0_5 = 0.5 / MAX_ALTITUDE
    action_0_5 = (norm_0_5 * 2.0) - 1.0
    high_level_action = np.array([action_0_5, 0.0, 0.0, 0.0], dtype=np.float32)

    # Reset FC and initialize at 0.5m
    fc.reset()
    rc = fc.compute_rc_commands(high_level_action, current_alt_norm=norm_0_5, dt=0.01)
    # Throttle should be exactly hover baseline 1550
    assert rc[0] == pytest.approx(1550.0, abs=1.0)
    assert rc[1] == pytest.approx(1500.0) # roll
    assert rc[2] == pytest.approx(1500.0) # pitch
    assert rc[3] == pytest.approx(1500.0) # yaw

    # Extreme climb command -> check ceiling at hover + 130 = 1680
    high_climb_action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # max ceiling
    for _ in range(100):
        rc = fc.compute_rc_commands(high_climb_action, current_alt_norm=0.1, dt=0.05)
    assert rc[0] <= 1680.0
    assert rc[0] >= 1675.0

    # Extreme descent command -> check floor at hover - 75 = 1475
    fc.reset()
    low_descent_action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32) # floor
    for _ in range(100):
        rc = fc.compute_rc_commands(low_descent_action, current_alt_norm=0.8, dt=0.05)
    assert rc[0] >= 1475.0
    assert rc[0] <= 1480.0


def test_flight_controller_ground_unweight_antiwindup():
    """Verify integral accumulation is frozen when altitude is below ground threshold."""
    fc = FlightController()
    below_ground_alt = (GROUND_THRESHOLD - 0.005) / MAX_ALTITUDE
    above_ground_alt = (GROUND_THRESHOLD + 0.050) / MAX_ALTITUDE

    # Drone on the ground wanting to climb: integral should be frozen at 0
    fc.reset()
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) # target 1.5m
    for _ in range(50):
        fc.compute_rc_commands(action, current_alt_norm=below_ground_alt, dt=0.01)
    assert fc.throttle_pid.integral == 0.0

    # Once airborne past ground threshold, integral should accumulate
    for _ in range(50):
        fc.compute_rc_commands(action, current_alt_norm=above_ground_alt, dt=0.01)
    assert fc.throttle_pid.integral > 0.0


def test_hover_physics_pybullet():
    """Verify that applying 1550 PWM to 475g drone produces hover equilibrium in PyBullet."""
    env = DroneEnv(use_gui=False)
    # Start drone airborne at 0.5m
    obs, info = env.reset(options={"initial_pos": [0.0, 0.0, 0.5]})
    initial_z = env.client.getBasePositionAndOrientation(env.drone_id)[0][2]

    # Command 1550 PWM (exact hover baseline) with level attitude
    rc_hover = np.array([HOVER_THROTTLE_PWM, 1500, 1500, 1500], dtype=np.float32)

    # Step simulation for 1 second (240 steps)
    for _ in range(int(PHYSICS_FREQ)):
        env.step(rc_hover)

    final_z = env.client.getBasePositionAndOrientation(env.drone_id)[0][2]
    env.close()

    # Vertical displacement should be negligible (< 3cm over 1 second of open loop hover)
    drift = abs(final_z - initial_z)
    assert drift < 0.03, f"Hover drift {drift:.4f}m exceeds 0.03m"


def test_hrl_wrapper_7d_observation():
    """Verify DroneHRLWrapper observation vector is 7D matching hardware sensor array."""
    base_env = DroneEnv(use_gui=False)
    env = DroneHRLWrapper(base_env)
    obs, info = env.reset()

    assert obs.shape == (7,), f"Expected 7D observation vector, got {obs.shape}"
    # [alt, front_dist, shift_x, shift_y, vel_x, vel_y, goal_alt]
    assert 0.0 <= obs[0] <= 1.0 # altitude
    assert 0.0 <= obs[1] <= 1.0 # front_distance
    assert -1.0 <= obs[2] <= 1.0 # shift_x
    assert -1.0 <= obs[3] <= 1.0 # shift_y
    assert -1.0 <= obs[4] <= 1.0 # vel_x
    assert -1.0 <= obs[5] <= 1.0 # vel_y
    assert obs[6] == pytest.approx(env.goal_alt) # goal_alt

    env.close()


def test_optical_flow_ground_cutoff():
    """Verify optical flow shifts and velocities are 0.0 when below 0.04m (MIN_ALTITUDE)."""
    env = DroneEnv(use_gui=False)
    # Place on ground at 0.02m (below MIN_ALTITUDE = 0.04m)
    obs, info = env.reset(options={"initial_pos": [0.0, 0.0, 0.02]})

    assert obs["shift_x"][0] == 0.0
    assert obs["shift_y"][0] == 0.0
    assert obs["velocity_x"][0] == 0.0
    assert obs["velocity_y"][0] == 0.0

    env.close()
