import time
import os
import math
from typing import Any, Tuple, Dict, Optional
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from urdf_parser_py import urdf
from pybullet_utils import bullet_client as bc

from settings import (
    G, MAX_THROTTLE, TILT_LIMIT, MAX_YAW_RATE_RADS, MAX_XY_SHIFT, MAX_VELOCITY,
    MAX_ALTITUDE, MIN_ALTITUDE, START_ALTITUDE, MAX_DISTANCE, MIN_VAL,
    DRONE_IMG_WIDTH, DRONE_IMG_HEIGHT, NUMBER_OF_CHANNELS, ASSETS_DIRECTORY
)

def convert_range(
    x: float, x_min: float, x_max: float, y_min: float, y_max: float
) -> float:
    """Converts value from one range system to another"""
    return ((x - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min


class DroneEnv(gym.Env):
    """Class responsible for drone avionics adapted for Gymnasium"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, use_gui=False) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == "human")

        self.plane_id = None
        self.drone_id = None
        self.target_id = None
        self.step_number = 0

        self.world_space = gym.spaces.Box(
            low=np.array([-20, -20, 0]),
            high=np.array([20, 20, MAX_ALTITUDE]),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "drone_img": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(DRONE_IMG_WIDTH, DRONE_IMG_HEIGHT, NUMBER_OF_CHANNELS),
                    dtype=np.uint8,
                ),
                "altitude": gym.spaces.Box(0, MAX_ALTITUDE, shape=(1,), dtype=np.float32),
                "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,), dtype=np.float32),
                "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,), dtype=np.float32),
                "yaw": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                "distance": gym.spaces.Box(0, MAX_DISTANCE, shape=(1,), dtype=np.float32),
                "shift_x": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                "shift_y": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                "velocity_x": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                "velocity_y": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.array([1000, 1000, 1000, 1000]),
            high=np.array([2000, 2000, 2000, 2000]),
            dtype=np.float32,
        )

        self.drone_img = np.zeros(self.observation_space["drone_img"].shape, dtype=np.uint8)

        self.num_motors = 0
        self.mass = 0
        self.max_rpm = 0
        self.min_rpm = 0
        self.target_pos = np.zeros(3)

        # PyBullet Client
        self.client = bc.BulletClient(
            connection_mode=p.GUI if self.use_gui else p.DIRECT
        )

        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -G)
        
        self.drone_img = np.zeros(self.observation_space["drone_img"].shape, dtype=np.uint8)
        self.step_number = 0

        self.plane_id = self.client.loadURDF("plane.urdf")

        # Drone initialization parameters
        initial_pos = options.get("initial_pos", [MIN_VAL, MIN_VAL, START_ALTITUDE])
        self.initial_pos_np = np.array(initial_pos)

        # Sample or get target position
        if options and "target_pos" in options:
            target_position = np.array(options["target_pos"])
        else:
            # Ensure target and drone do not overlap (at least 0.5m away)
            while True:
                target_position = self.world_space.sample()
                # Ensure it's not too close to the drone's initial position
                if np.linalg.norm(target_position - self.initial_pos_np) >= 0.5:
                    break

        # Target initialization
        collision_shape_id = self.client.createCollisionShape(
            shapeType=self.client.GEOM_MESH,
            fileName=os.path.join(ASSETS_DIRECTORY, "a_cube.obj"),
        )
        visual_shape_id = self.client.createVisualShape(
            shapeType=self.client.GEOM_MESH,
            fileName=os.path.join(ASSETS_DIRECTORY, "a_cube.obj"),
        )
        self.target_id = self.client.createMultiBody(
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=target_position.tolist(),
        )
        self.target_pos = target_position

        # Drone initialization
        self.drone_id = self.client.loadURDF(
            os.path.join(ASSETS_DIRECTORY, "drone.urdf"),
            initial_pos
        )
        self.client.changeDynamics(self.drone_id, -1, linearDamping=0.01, angularDamping=0.1)
        
        self.client.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=45, 
            cameraPitch=-30, 
            cameraTargetPosition=initial_pos
        )

        linear_velocity, _ = self.client.getBaseVelocity(self.drone_id)
        self.set_drone_params()
        shift = self._get_cumulative_shift()

        obs = {
            "drone_img": self.drone_img,
            "distance": np.array([1.0], dtype=np.float32),
            "roll": np.array([0.0], dtype=np.float32),
            "pitch": np.array([0.0], dtype=np.float32),
            "yaw": np.array([0.0], dtype=np.float32),
            "altitude": np.array([self._get_altitude()], dtype=np.float32),
            "shift_x": np.array([shift[0]], dtype=np.float32),
            "shift_y": np.array([shift[1]], dtype=np.float32),
            "velocity_x": np.array([np.clip(linear_velocity[0] / MAX_VELOCITY, -1.0, 1.0)], dtype=np.float32),
            "velocity_y": np.array([np.clip(linear_velocity[1] / MAX_VELOCITY, -1.0, 1.0)], dtype=np.float32),
        }
        
        return obs, {"vertical_velocity": linear_velocity[2]}

    def set_drone_params(self):
        drone_params = urdf.URDF.from_xml_file(os.path.join(ASSETS_DIRECTORY, "drone.urdf"))
        self.mass = sum([m.inertial.mass for m in drone_params.links])
        motor_links = [l for l in drone_params.links if l.name.startswith("rotor_")]
        self.num_motors = len(motor_links)

        kv = float(drone_params.gazebos[0].attrib["kv"])
        voltage = float(drone_params.gazebos[0].attrib["voltage"])
        rpm_unloaded = kv * voltage
        self.max_rpm = rpm_unloaded * 0.62
        self.min_rpm = self.max_rpm * 0.04

    def step(self, rc_command: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._apply_physics(rc_command)

        altitude = self._get_altitude()
        angles = self._get_angles()
        linear_velocity, angular_velocity = self.client.getBaseVelocity(self.drone_id)
        distance = self._get_distance()
        self.drone_img = self._get_drone_view()

        self.step_number += 1

        below_alt_min_limit = (self.step_number > 10 and altitude <= MIN_ALTITUDE / MAX_ALTITUDE)
        tilt_too_big = abs(angles[0]) > 1.0 or abs(angles[1]) > 1.0 # Angles are normalized by TILT_LIMIT
        above_alt_max_limit = altitude >= 1.0

        terminated = above_alt_max_limit or below_alt_min_limit or tilt_too_big
        truncated = False # Or some other condition if needed

        shift = self._get_cumulative_shift()
        yaw_rate_norm = np.clip(angular_velocity[2] / MAX_YAW_RATE_RADS, -1.0, 1.0)

        obs = {
            "drone_img": self.drone_img,
            "distance": np.array([distance], dtype=np.float32),
            "altitude": np.array([altitude], dtype=np.float32),
            "roll": np.array([angles[0]], dtype=np.float32),
            "pitch": np.array([angles[1]], dtype=np.float32),
            "yaw": np.array([yaw_rate_norm], dtype=np.float32),
            "shift_x": np.array([shift[0]], dtype=np.float32),
            "shift_y": np.array([shift[1]], dtype=np.float32),
            "velocity_x": np.array([np.clip(linear_velocity[0] / MAX_VELOCITY, -1.0, 1.0)], dtype=np.float32),
            "velocity_y": np.array([np.clip(linear_velocity[1] / MAX_VELOCITY, -1.0, 1.0)], dtype=np.float32),
        }

        reward = 1.0 # Basic reward

        return obs, reward, terminated, truncated, {
            "step_number": self.step_number,
            "vertical_velocity": linear_velocity[2],
        }

    def render(self):
        if self.render_mode == "rgb_array":
            pos, _ = self.client.getBasePositionAndOrientation(self.drone_id)
            view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=pos,
                distance=1.5,
                yaw=45.0,
                pitch=-30.0,
                roll=0.0,
                upAxisIndex=2
            )
            proj_matrix = self.client.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, rgb, _, _) = self.client.getCameraImage(
                width=DRONE_IMG_WIDTH,
                height=DRONE_IMG_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )

            rgb_array = np.array(rgb, dtype=np.uint8).reshape((DRONE_IMG_HEIGHT, DRONE_IMG_WIDTH, 4))
            return rgb_array[:, :, :3]
        return None

    def close(self):
        if self.client:
            self.client.disconnect()

    def _apply_physics(self, rc_command: Any):
        throttle_norm = convert_range(rc_command[0], 1000, 2000, 0, 1)
        roll_norm = convert_range(rc_command[1], 1000, 2000, -1, 1)
        pitch_norm = convert_range(rc_command[2], 1000, 2000, -1, 1)
        yaw_norm = convert_range(rc_command[3], 1000, 2000, -1, 1)

        motor_mix_thrust = np.array([
            throttle_norm + roll_norm - pitch_norm, # FL
            throttle_norm + roll_norm + pitch_norm, # RL
            throttle_norm - roll_norm - pitch_norm, # FR
            throttle_norm - roll_norm + pitch_norm  # RR
        ])
        motor_mix_thrust = np.clip(motor_mix_thrust, 0, 1)
        thrusts = motor_mix_thrust * MAX_THROTTLE
        z_torque = yaw_norm * MAX_THROTTLE
            
        for i in range(self.num_motors):
            self.client.applyExternalForce(
                self.drone_id, i, [0, 0, thrusts[i]], [0, 0, 0], p.LINK_FRAME
            )
            
        self.client.applyExternalTorque(self.drone_id, -1, [0, 0, z_torque], p.LINK_FRAME)
        self.client.stepSimulation()

        if self.use_gui:
            time.sleep(0.01)

    def _get_distance(self) -> float:
        pos, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        rot_mat = self.client.getMatrixFromQuaternion(orn)
        drone_direction = np.array([rot_mat[0], rot_mat[3], rot_mat[6]]) * MAX_DISTANCE
        ray_result = self.client.rayTest(pos, pos + drone_direction)
        results = [hit[2] for hit in ray_result if hit[0] != -1]
        if len(results):
            return min(results)
        return 1.0

    def _get_angles(self) -> tuple[float, float, float]:
        _, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        orn_euler = self.client.getEulerFromQuaternion(orn)
        return (
            round(orn_euler[0] / TILT_LIMIT, 4),
            round(orn_euler[1] / TILT_LIMIT, 4),
            round(orn_euler[2] / np.deg2rad(360), 4),
        )

    def _get_altitude(self) -> float:
        pos, _ = self.client.getBasePositionAndOrientation(self.drone_id)
        return round(pos[2] / MAX_ALTITUDE, 4)

    def _get_drone_view(self) -> np.ndarray:
        pos, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        rot_mat = self.client.getMatrixFromQuaternion(orn)
        # Look down from the drone
        camera_eye = np.array(pos)
        # Assuming Z is UP, the drone looks down (-Z in link frame)
        # We transform [0, 0, -1] from link frame to world frame
        look_dir = np.array([rot_mat[2], rot_mat[5], rot_mat[8]]) * -1.0
        camera_target = camera_eye + look_dir
        # Up vector is along drone's X-axis
        up_vector = np.array([rot_mat[0], rot_mat[3], rot_mat[6]])
        
        view_matrix = self.client.computeViewMatrix(
            cameraEyePosition=camera_eye.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=up_vector.tolist(),
        )
        proj_matrix = self.client.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.01, farVal=10.0
        )
        (_, _, rgb, _, _) = self.client.getCameraImage(
            width=DRONE_IMG_WIDTH,
            height=DRONE_IMG_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((DRONE_IMG_HEIGHT, DRONE_IMG_WIDTH, 4))
        return rgb_array[:, :, :3]

    def _get_cumulative_shift(self) -> list:
        drone_pos, _ = self.client.getBasePositionAndOrientation(self.drone_id)
        drone_pos = np.array(drone_pos)
        # TODO: change to drone_pos - initial pos
        diff_world = drone_pos - self.initial_pos_np 
        shift_x = np.clip(diff_world[0] / MAX_XY_SHIFT, -1.0, 1.0)
        shift_y = np.clip(diff_world[1] / MAX_XY_SHIFT, -1.0, 1.0)
        return [shift_x, shift_y]
