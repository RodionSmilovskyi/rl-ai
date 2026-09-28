import os
import numpy as np

# Physical Constants
G = 9.81
DRONE_WEIGHT = 0.475  # 475 grams
HOVER_THROTTLE_PWM = 1550
HOVER_THROTTLE_NORM = (HOVER_THROTTLE_PWM - 1000) / 1000.0  # 0.550
MAX_THROTTLE = (DRONE_WEIGHT * G) / (4.0 * HOVER_THROTTLE_NORM)  # ~2.11807 N per motor
THRUST_TO_WEIGHT_RATIO = 1.0 / HOVER_THROTTLE_NORM  # ~1.818

# Environment Limits (aligned with DESIGN.md)
MAX_ALTITUDE = 3.0        # meters (downward ToF VL53L1X range)
MAX_FRONT_DISTANCE = 3.0  # meters (forward ToF VL53L1X range)
MAX_DISTANCE = MAX_FRONT_DISTANCE
MIN_ALTITUDE = 0.04       # meters (optical flow cutoff altitude)
START_ALTITUDE = 0.05     # meters
GROUND_THRESHOLD = 0.020  # meters (landing gear unweight threshold)
TILT_LIMIT = np.deg2rad(55)
MAX_YAW_RATE_RADS = np.deg2rad(360)
MAX_XY_SHIFT = 1.0        # meters
MAX_VELOCITY = 5.0        # m/s

# Sensor Filter Coefficients (aligned with DESIGN.md)
ALPHA_ALT = 0.6           # Low-latency altitude feedback filter
ALPHA_FRONT = 0.3         # Forward obstacle filter
ALPHA_FLOW = 0.2          # Optical flow filter
FLOW_DEADBAND = 0.05      # Optical flow jitter deadband

# Observation Constants
DRONE_IMG_WIDTH = 256
DRONE_IMG_HEIGHT = 256
NUMBER_OF_CHANNELS = 3

# Timing & Episode Length (2-minute battery lifetime)
PHYSICS_FREQ = 240.0
K_STEPS = 20              # 12 Hz high-level control loop
EPISODE_TIME_SEC = 120.0  # 2 minutes battery flight endurance
SUB_EPISODE_LIMIT = int(EPISODE_TIME_SEC * (PHYSICS_FREQ / K_STEPS))  # 1440 steps
MAX_PHYSICS_STEPS = int(EPISODE_TIME_SEC * PHYSICS_FREQ)  # 28800 steps
GAMMA = 0.99

# Internal Constants
MIN_VAL = 1e-4
FRAME_NUMBER = 500

# Paths
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIRECTORY = os.path.normpath(os.path.join(WORKING_DIRECTORY, "assets"))
