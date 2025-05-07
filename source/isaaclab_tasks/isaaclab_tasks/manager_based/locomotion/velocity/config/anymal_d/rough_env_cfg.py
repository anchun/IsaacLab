# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
import math
import numpy as np
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import numpy as np
import os

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

# 从 .npy 文件加载轨迹点
current_file_directory = os.path.dirname(os.path.abspath(__file__))
trajectory_points = np.load(os.path.join(current_file_directory, "trajectory_points.npy"))
# trajectory_points = np.array([
#     [3195.0, -1548.0],
#     [3391.3, -1576.6],
#     [3354.22, -1576.63]
# ])
# trajectory_points = np.array([
#     [-28.0, 10.0],
#     [-31.0, 13.5],
#     [-31.0, 22.3]
# ])
# trajectory_points = [
#     [-28.0, 10.0],
#     [-30.5, 13.5],
#     [-31.5, 22.3]
# ]    # 训练时用这个
all_trajectory_points = [
    [
    [-28.0, 10.0],
    [-30.5, 13.5],
    [-31.5, 22.3]
    ],
    [
    [-21.0, 10.0],
    [-10.0, 10.0],
    [3.0, 10.0],
    [7.0, 10.0],
    [7.0, 5.0],
    [7.0, 10.0]
    ],
    [
    [-21.0, 10.0],
    [-10.0, 10.0],
    [3.0, 10.0],
    [7.0, 10.0],
    [18.0, 1.0],
    [16.0, 3.0],
    [16.0, 10.0]
    ],
    [
    [-41.0, 13.0],
    [-40.0, 30.0],
    [-41.0, 13.0]
    ]]    # 训练时用这个

@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class AnymalDRoughWithNavigationCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityNavigationCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.1, 0.1),
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.0, 2.0), heading=(0, 0)),
        waypoints=all_trajectory_points,
    )


@configclass
class AnymalDRoughEnvCfg_PLAY(AnymalDRoughEnvCfg):
    commands: AnymalDRoughWithNavigationCommandsCfg = AnymalDRoughWithNavigationCommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 60.0
        self.scene.terrain.terrain_type = "usd"
        # self.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room_dog.usd"
        self.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital_dog_very_low.usd"
        # self.scene.terrain.usd_path = r"D:\fsy\scene\Data_01_RobotArm\L_Workshop_01_latest.usd"
        self.scene.sky_light = None

        self.curriculum.terrain_levels = None

        self.events.reset_base.params = {
            # "pose_range": {"x": (-3.5, -3.5), "y": (0.0, 0.0), "z": (0.04, 0.04), "yaw": (0, 0)},  # room
            "pose_range": [{"x": (-28.0, -28.0), "y": (10.0, 10.0), "z": (0.72, 0.72), "yaw": (0, 0)},
                           {"x": (-34.0, -34.0), "y": (12.0, 12.0), "z": (0.72, 0.72), "yaw": (0, 0)},
                           {"x": (-36.0, -36.0), "y": (12.0, 12.0), "z": (0.72, 0.72), "yaw": (0, 0)},
                           {"x": (-40.5, -40.5), "y": (5.0, 5.0), "z": (0.72, 0.72), "yaw": (0, 0)},
                           {"x": (-44.0, -44.0), "y": (8.5, 8.5), "z": (0.72, 0.72), "yaw": (0, 0)},
                           {"x": (-40.5, -40.5), "y": (18.0, 18.0), "z": (0.72, 0.72), "yaw": (0, 0)}],   # hospital
            # "pose_range": {"x": (3195, 3195), "y": (-1548, -1548), "z": (120, 120), "yaw": (0, 0)},   # workshop1
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)

        # New Commands
        # self.commands.base_velocity.ranges.lin_vel_z = (0.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_x = (-2.0, 2.0)
        # self.commands.base_velocity.ranges.ang_vel_y = (-2.0, 2.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
