import os
import numpy as np

# Physical Constants
G = 9.81
DRONE_WEIGHT = 0.421
THRUST_TO_WEIGHT_RATIO = 4
MAX_THROTTLE = 9.82 / 4.0 # (DRONE_WEIGHT * G * THRUST_TO_WEIGHT_RATIO) / 4.0

# Environment Limits
MAX_DISTANCE = 4.0  # meters
MAX_ALTITUDE = 1.0  # meters
MIN_ALTITUDE = 0.04  # meters
START_ALTITUDE = 0.05
TILT_LIMIT = np.deg2rad(55)
MAX_YAW_RATE_RADS = np.deg2rad(360)
MAX_XY_SHIFT = 1.0  # Meters
MAX_VELOCITY = 5.0  # Meters per second

# Observation Constants
DRONE_IMG_WIDTH = 256
DRONE_IMG_HEIGHT = 256
NUMBER_OF_CHANNELS = 3

# HRL Constants
K_STEPS = 20
SUB_EPISODE_LIMIT = 24
PHYSICS_FREQ = 240.0
GAMMA = 0.99

# Internal Constants
MIN_VAL = 1e-4
FRAME_NUMBER = 500

# Paths
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIRECTORY = os.path.normpath(os.path.join(WORKING_DIRECTORY, "assets"))
