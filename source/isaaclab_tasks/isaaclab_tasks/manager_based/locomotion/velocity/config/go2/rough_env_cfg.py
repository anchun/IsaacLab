# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import numpy as np
import os
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# 从 .npy 文件加载轨迹点
all_trajectory_points = []
# base_point = [2.52, 2.69]

# 获取当前目录下所有匹配 trajectory_points_*.npy 的文件
current_directory = os.path.dirname(os.path.abspath(__file__))
npy_files = [f for f in os.listdir(current_directory) 
             if f.startswith('trajectory_points_') and f.endswith('.npy')]

# 按数字排序文件（确保顺序正确）
npy_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

# 加载每个文件并在开头添加基准点
for file in npy_files:
    file_path = os.path.join(current_directory, file)
    trajectory_data = np.load(file_path)
    # all_trajectory_points.append([base_point] + list(trajectory_data))
    all_trajectory_points.append(list(trajectory_data))

# all_trajectory_points = [[
#     [-1.2, -0.5],
#     [1.8, 2.1]
# ]]
# all_trajectory_points_hospital = [
#     [
#     [-28.0, 10.0],
#     [-30.5, 13.5],
#     [-31.5, 22.3]
#     ],
#     [
#     [-21.0, 10.0],
#     [-10.0, 10.0],
#     [3.0, 10.0],
#     [7.0, 10.0],
#     [7.0, 5.0],
#     [7.0, 10.0]
#     ],
#     [
#     [-21.0, 10.0],
#     [-10.0, 10.0],
#     [3.0, 10.0],
#     [7.0, 10.0],
#     [18.0, 1.0],
#     [16.0, 3.0],
#     [16.0, 10.0]
#     ],
#     [
#     [-41.0, 13.0],
#     [-40.0, 30.0],
#     [-41.0, 13.0]
#     ]]    # 训练时用这个

pose_range_hospital = [{"x": (-28.0, -28.0), "y": (10.0, 10.0), "z": (0.72, 0.72), "yaw": (0, 0)},
            {"x": (-34.0, -34.0), "y": (12.0, 12.0), "z": (0.72, 0.72), "yaw": (0, 0)},
            {"x": (-36.0, -36.0), "y": (12.0, 12.0), "z": (0.72, 0.72), "yaw": (0, 0)},
            {"x": (-40.5, -40.5), "y": (5.0, 5.0), "z": (0.72, 0.72), "yaw": (0, 0)},
            {"x": (-44.0, -44.0), "y": (8.5, 8.5), "z": (0.72, 0.72), "yaw": (0, 0)},
            {"x": (-40.5, -40.5), "y": (18.0, 18.0), "z": (0.72, 0.72), "yaw": (0, 0)}]   # hospital
pose_range_showroom = [{"x": (4.8, 4.8), "y": (9.5, 9.5), "z": (0.3, 0.3), "yaw": (0, 0)}] #,
                    #    {"x": (2.0, 2.0), "y": (2.0, 2.0), "z": (0.3, 0.3), "yaw": (0, 0)},
                    #    {"x": (-0.5, -0.5), "y": (2.1, 2.1), "z": (0.3, 0.3), "yaw": (0, 0)}]


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": pose_range_showroom,
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

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
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    commands: AnymalDRoughWithNavigationCommandsCfg = AnymalDRoughWithNavigationCommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 60.0
        self.scene.terrain.terrain_type = "usd"
        # self.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room_dog.usd"
        self.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/AITower/L_ShowRoom_01.usd"#demo_showroom.usd"
        # self.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
        # self.scene.terrain.usd_path = r"D:\fsy\scene\Data_01_RobotArm\L_Workshop_01_latest.usd"
        self.scene.sky_light = None

        self.curriculum.terrain_levels = None


        # # make a smaller scene for play
        # self.scene.num_envs = 50
        # self.scene.env_spacing = 2.5
        # # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
