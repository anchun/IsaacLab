# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import math

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg, UniformVelocityNavigationCommandCfg


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0

# class UniformVelocityNavigationCommand(UniformVelocityCommand):
#     """Command generator that generates a velocity command in SE(2) for navigation tasks.

#     The command comprises of a linear velocity in x and y direction and an angular velocity around
#     the z-axis. It is given in the robot's base frame.
#     """

#     cfg: UniformVelocityNavigationCommandCfg
#     """The command generator configuration."""

#     def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
#         """Initializes the command generator.

#         Args:
#             cfg: The command generator configuration.
#             env: The environment.
#         """
#         super().__init__(cfg, env)
#         self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

#     def _resample_command(self, env_ids):
#         super()._resample_command(env_ids)
#         r = torch.empty(len(env_ids), device=self.device)
#         if self.command_counter[env_ids] <= 1:
#             self.waypointIndex[env_ids] = 0 # reset waypoint index to 0 for the first time
#         firstIndex = self.waypointIndex[env_ids]
#         nextIndex = (self.waypointIndex[env_ids] + 1) % len(self.cfg.waypoints)
#         heading = self.cfg.waypoints[nextIndex] - self.cfg.waypoints[firstIndex]
#         heading_angle = math.atan2(heading[1], heading[0])
#         self.heading_target[env_ids] = torch.tensor(heading_angle).to(self.device)
#         # update waypoint index
#         self.waypointIndex[env_ids] = nextIndex

class UniformVelocityNavigationCommand_ori(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) for navigation tasks."""

    cfg: UniformVelocityNavigationCommandCfg

    def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        if self.command_counter[env_ids] <= 1:
            # print("===0000====")
            self.waypointIndex[env_ids] = 0 # reset waypoint index to 0 for the first time
        
        # 获取当前的轨迹点
        firstIndex = self.waypointIndex[env_ids]
        current_waypoint = np.array([self.cfg.waypoints[firstIndex][0], self.cfg.waypoints[firstIndex][1]])
        # print(current_waypoint)
        # 计算到目标点的距离
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # 转换为 NumPy 数组
        base_pos_w_pr = base_pos_w[env_ids, :3].cpu().numpy()  # 转换为 NumPy 数组
        # 计算距离
        # print("====after====", current_waypoint)
        distance_to_target = np.linalg.norm(base_pos_w_np - current_waypoint)
        # print("====after_2====", current_waypoint)
        # distance_to_target = torch.norm(base_pos_w[env_ids, :2].cpu().numpy() - current_waypoint, dim=1)
        # 检查是否到达目标点
        reached_target = distance_to_target < 0.2  # 设定阈值，例如 40 cm

        print("====position=====", base_pos_w_pr, current_waypoint, distance_to_target, reached_target)

        # 只有在到达目标点时才更新目标点索引
        if reached_target:
            nextIndex = (self.waypointIndex[env_ids] + 1) % len(self.cfg.waypoints)
            heading = (self.cfg.waypoints[nextIndex] - base_pos_w_np)[0]
            # print(heading)
            heading_angle = math.atan2(heading[1], heading[0])
            # update waypoint index
            self.waypointIndex[env_ids] = nextIndex
            self.heading_target[env_ids] = torch.tensor(heading_angle).to(self.device)
        else:
            # lastIndex = (self.waypointIndex[env_ids] - 1) % len(self.cfg.waypoints)
            heading = (self.cfg.waypoints[firstIndex] - base_pos_w_np)[0]
            heading_angle = math.atan2(heading[1], heading[0])
            # update waypoint index
            self.heading_target[env_ids] = torch.tensor(heading_angle).to(self.device)

        print("=====tar=====", self.heading_target[env_ids])

class UniformVelocityNavigationCommand_one_trajectory(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) for navigation tasks."""

    cfg: UniformVelocityNavigationCommandCfg

    def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        # 重置初始索引
        reset_indices = (self.command_counter[env_ids] <= 1)
        if reset_indices.any():
            self.waypointIndex[env_ids[reset_indices]] = 0

        # 获取当前位置信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)
        base_pos_w_pr = base_pos_w[env_ids, :3].cpu().numpy()  # (N, 3)

        # 获取每个环境的当前轨迹点
        current_waypoints = np.array([
            [self.cfg.waypoints[idx.item()][0], self.cfg.waypoints[idx.item()][1]]
            for idx in self.waypointIndex[env_ids]
        ])  # (N, 2)

        # 计算到目标点的距离 (向量化操作)
        distances = np.linalg.norm(base_pos_w_np - current_waypoints, axis=1)  # (N,)
        reached_targets = distances < 0.2  # (N,)

        # 打印调试信息
        for i, env_id in enumerate(env_ids):
            print(f"Dog {env_id}: pos={base_pos_w_pr[i]}, target={current_waypoints[i]}, "
                  f"distance={distances[i]}, reached={reached_targets[i]}")

        # 更新到达目标点的机器狗的目标点索引
        for i, (env_id, reached) in enumerate(zip(env_ids, reached_targets)):
            if reached:
                # 更新waypoint索引
                next_idx = (self.waypointIndex[env_id] + 1) % len(self.cfg.waypoints)
                self.waypointIndex[env_id] = next_idx
                # 计算新的目标方向
                heading = self.cfg.waypoints[next_idx] - base_pos_w_np[i]
                heading_angle = math.atan2(heading[1], heading[0])
                self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)
            else:
                # 继续朝向当前目标点
                current_idx = self.waypointIndex[env_id]
                heading = self.cfg.waypoints[current_idx] - base_pos_w_np[i]
                heading_angle = math.atan2(heading[1], heading[0])
                self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)

        # 打印目标方向
        print("Target headings:", self.heading_target[env_ids])

current_waypoints = {}
class UniformVelocityNavigationCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) for navigation tasks."""

    cfg: UniformVelocityNavigationCommandCfg

    def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        # 重置初始索引
        reset_indices = (self.command_counter[env_ids] <= 1)
        if reset_indices.any():
            self.waypointIndex[env_ids[reset_indices]] = 0

        # 获取当前位置信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)

        # 随机选择轨迹
        print(env_ids, type(env_ids))
        num_dogs = len(env_ids)
        selected_trajectories = [i % len(self.cfg.waypoints) for i in range(num_dogs)]
        # selected_trajectories = np.random.choice(len(self.cfg.waypoints), num_dogs)
        global current_waypoints

        base_point = np.array([2.52, 2.69])

        for i in range(num_dogs):
            trajectory = self.cfg.waypoints[selected_trajectories[i]]
            if base_pos_w_np[i][0] > 2.5:
                trajectory = base_point + trajectory
            # 在轨迹点上添加微扰
            print("trajectory", trajectory)
            perturbed_points = [
                [point[0] + np.random.uniform(-0.2, 0.2), point[1] + np.random.uniform(-0.2, 0.2)]
                for point in trajectory
            ]
            # 只在current_waypoints中不存在env_id时才赋值
            if env_ids[i].item() not in current_waypoints:
                current_waypoints[env_ids[i].item()] = perturbed_points

        # 计算到目标点的距离
        distances = []
        for i in range(num_dogs):
            current_idx = self.waypointIndex[env_ids[i]].item()
            if current_idx < len(current_waypoints[env_ids[i].item()]):
                target_point = current_waypoints[env_ids[i].item()][current_idx]
                distance = np.linalg.norm(base_pos_w_np[i] - target_point)  # 计算距离
                distances.append(distance)
            else:
                distances.append(float('inf'))  # 如果索引超出范围，设置为无穷大

        distances = np.array(distances)  # (N,)
        reached_targets = distances < 0.6  # (N,)

        # 打印调试信息
        for i, env_id in enumerate(env_ids):
            print(f"Dog {env_id}: pos={base_pos_w_np[i]}, target={current_waypoints[env_ids[i].item()]}, "
                f"distance={distances[i]}, reached={reached_targets[i]}")

        # 更新到达目标点的机器狗的目标点索引
        for i, (env_id, reached) in enumerate(zip(env_ids, reached_targets)):
            if reached:
                # 更新waypoint索引
                next_idx = (self.waypointIndex[env_id] + 1) % len(current_waypoints[env_ids[i].item()])  # 确保索引在范围内
                self.waypointIndex[env_id] = next_idx
                # 计算新的目标方向
                if next_idx < len(current_waypoints[env_ids[i].item()]):
                    heading = current_waypoints[env_ids[i].item()][next_idx] - base_pos_w_np[i]
                    heading_angle = math.atan2(heading[1], heading[0])
                    self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)
            else:
                # 继续朝向当前目标点
                current_idx = self.waypointIndex[env_id]
                if current_idx < len(current_waypoints[env_ids[i].item()]):
                    heading = current_waypoints[env_ids[i].item()][current_idx] - base_pos_w_np[i]
                    heading_angle = math.atan2(heading[1], heading[0])
                    self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)

        # 打印目标方向
        print("Target headings:", self.heading_target[env_ids])
