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

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


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
        resampling_time_range=(4.0, 4.0),
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.0, 2.0), heading=(0, 0)),
        waypoints=np.array([[-3.5, 0.0], [0, -3.5], [3.5, 0], [0, 3.5]]),
    )

@configclass
class AnymalDRoughEnvCfg_PLAY(AnymalDRoughEnvCfg):
    commands: AnymalDRoughWithNavigationCommandsCfg = AnymalDRoughWithNavigationCommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 60.0
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room_dog.usd"
        self.scene.sky_light = None

        self.curriculum.terrain_levels = None

        self.events.reset_base.params = {
            "pose_range": {"x": (-3.5, -3.5), "y": (0.0, 0.0), "z": (0.04, 0.04), "yaw": (0, 0)},
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

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
