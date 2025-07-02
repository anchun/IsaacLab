# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots import UR10_CFG

@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.object_pose.debug_vis = False
        # room
        self.scene.plane.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Data_02_ShowRoom/L_ShowRoom_02.usd"
        self.scene.plane.init_state.pos = [0, 0, -0.7]
        self.scene.plane.init_state.rot = [0.707, 0, 0, -0.707]
        # change table to a rigid box
        self.scene.table = None
        self.scene.targetBox = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetBox",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.3, -0.6, 0.1], rot=[0.707, 0, 0, -0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd"),
        )
        # robots
        self.scene.robot.init_state.pos = [-2.8, -1.0, 0.0]
        self.scene.robot.init_state.rot = [1.0, 0, 0, 0.0]
        self.scene.robot_ur10 = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_UR10")
        self.scene.robot_ur10.init_state.pos = [-2.8, 0.6, 0.0]
        self.scene.robot_ur10.init_state.rot = [1.0, 0, 0, 0.0]
        self.scene.robot_ur10.spawn.rigid_props.disable_gravity = True
        self.scene.robot_ur10.actuators["arm"].stiffness = 0.0
        self.scene.robot_ur10.actuators["arm"].damping = 0.0
        
        # object
        self.scene.object.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd"
        self.scene.object.init_state.pos = [-2.2, -1.0, 0.1]
        self.scene.object.spawn.scale = (0.7, 0.7, 0.7)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0), "yaw": (-0.314, 0.314)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

