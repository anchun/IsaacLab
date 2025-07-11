# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
        # self.vel_command_b[env_ids, 0] = r.uniform_(*[0.0,0.0])
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # self.vel_command_b[env_ids, 2] = r.uniform_(*[-0.2,0.2])
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _resample_command_demo(self, env_ids: Sequence[int]):
        # 首先获取当前位置信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)

        # 生成基础速度命令
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # 计算避障速度调整
        num_dogs = len(env_ids)
        avoidance_velocities = torch.zeros((num_dogs, 2), device=self.device)
        
        # 避障参数设置
        min_distance = 0.8  # 最小安全距离
        max_distance = 1.5  # 最大影响距离
        max_avoidance_force = 0.8  # 最大避障力
        
        for i in range(num_dogs):
            for j in range(num_dogs):
                if i != j:  # 不与自己比较
                    # 计算两只狗之间的距离向量
                    pos_diff = base_pos_w_np[i] - base_pos_w_np[j]
                    distance = np.linalg.norm(pos_diff)
                    
                    if distance < max_distance:
                        # 计算避障力的强度（距离越近，力越大）
                        if distance < min_distance:
                            force_magnitude = max_avoidance_force
                        else:
                            # 在min_distance和max_distance之间平滑过渡
                            force_magnitude = max_avoidance_force * (1 - (distance - min_distance)/(max_distance - min_distance))
                        
                        # 计算避障方向（归一化）
                        if distance > 0:
                            avoid_dir = pos_diff / distance
                        else:
                            avoid_dir = np.array([1.0, 0.0])  # 如果距离为0，给一个默认方向
                        
                        # 将避障力转换为速度调整
                        avoidance_velocities[i] += torch.tensor(avoid_dir * force_magnitude, device=self.device)
        
        # 应用避障速度调整
        # 保持原始速度的方向，但根据避障力调整大小
        for i in range(num_dogs):
            original_vel = self.vel_command_b[env_ids[i], :2]
            avoidance_vel = avoidance_velocities[i]
            
            # 计算合成速度
            combined_vel = original_vel + avoidance_vel
            
            # 限制最大速度
            max_speed = torch.norm(original_vel)  # 使用原始速度的大小作为最大速度
            current_speed = torch.norm(combined_vel)
            
            if current_speed > max_speed:
                combined_vel = combined_vel * (max_speed / current_speed)
            
            # 更新速度命令
            self.vel_command_b[env_ids[i], :2] = combined_vel
            
            # 打印调试信息
            if torch.any(avoidance_velocities[i] != 0):
                print(f"Dog {env_ids[i]}: Original vel: {original_vel}, Avoidance vel: {avoidance_vel}, "
                    f"Final vel: {combined_vel}")

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _resample_command_demo1(self, env_ids: Sequence[int]):
        # 首先获取当前位置信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)
        base_vel_w = self.robot.data.root_lin_vel_b[env_ids, :2].cpu().numpy()  # 获取当前速度

        # 生成基础速度命令
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # 计算避障速度调整
        num_dogs = len(env_ids)
        avoidance_velocities = torch.zeros((num_dogs, 2), device=self.device)
        
        # 避障参数设置
        min_distance = 0.5  # 最小安全距离
        max_distance = 0.8  # 最大影响距离
        max_avoidance_force = 0.5  # 最大避障力
        
        # 只对最后一只狗进行避障
        if num_dogs >= 1:
            last_dog_idx = num_dogs - 1  # 最后一只狗的索引
            last_dog_pos = base_pos_w_np[last_dog_idx]
            last_dog_vel = base_vel_w[last_dog_idx]
            
            for j in range(num_dogs - 1):  # 只检查与其他狗的距离
                # 计算与目标狗之间的距离向量
                pos_diff = last_dog_pos - base_pos_w_np[j]
                distance = np.linalg.norm(pos_diff)
                
                if distance < max_distance:
                    # 计算避障力的强度（距离越近，力越大）
                    if distance < min_distance:
                        force_magnitude = max_avoidance_force
                    else:
                        # 在min_distance和max_distance之间平滑过渡
                        force_magnitude = max_avoidance_force * (1 - (distance - min_distance)/(max_distance - min_distance))
                    
                    # 计算避障方向
                    if distance > 0:
                        # 获取当前位置到目标狗的方向向量
                        avoid_dir = pos_diff / distance
                        
                        # 计算当前速度方向
                        if np.linalg.norm(last_dog_vel) > 0.1:  # 如果速度足够大
                            vel_dir = last_dog_vel / np.linalg.norm(last_dog_vel)
                            # 计算速度方向与避障方向的点积
                            dot_product = np.dot(vel_dir, avoid_dir)
                            
                            # 如果速度方向与避障方向夹角小于90度（点积大于0）
                            # 说明正在接近目标狗，需要调整避障方向
                            if dot_product > 0:
                                # 计算垂直于避障方向的方向
                                perp_dir = np.array([-avoid_dir[1], avoid_dir[0]])
                                # 根据速度方向选择垂直方向
                                if np.dot(vel_dir, perp_dir) < 0:
                                    perp_dir = -perp_dir
                                # 将避障方向调整为垂直方向
                                avoid_dir = perp_dir
                        else:
                            # 如果速度很小，使用垂直于避障方向的方向
                            avoid_dir = np.array([-pos_diff[1], pos_diff[0]]) / distance
                            # 随机选择垂直方向
                            if np.random.random() < 0.5:
                                avoid_dir = -avoid_dir
                    else:
                        # 如果距离为0，随机选择一个方向
                        angle = np.random.uniform(0, 2*np.pi)
                        avoid_dir = np.array([np.cos(angle), np.sin(angle)])
                    
                    # 将避障力转换为速度调整
                    avoidance_velocities[last_dog_idx] += torch.tensor(avoid_dir * force_magnitude, device=self.device)
        
        # 应用避障速度调整（只对最后一只狗）
        if num_dogs >= 1:
            last_dog_idx = num_dogs - 1
            original_vel = self.vel_command_b[env_ids[last_dog_idx], :2]
            avoidance_vel = avoidance_velocities[last_dog_idx]
            
            # 计算合成速度
            combined_vel = original_vel + avoidance_vel
            
            # 限制最大速度
            max_speed = torch.norm(original_vel)  # 使用原始速度的大小作为最大速度
            current_speed = torch.norm(combined_vel)
            
            if current_speed > max_speed:
                combined_vel = combined_vel * (max_speed / current_speed)
            
            # 更新速度命令
            self.vel_command_b[env_ids[last_dog_idx], :2] = combined_vel
            
            # 打印调试信息
            if torch.any(avoidance_velocities[last_dog_idx] != 0):
                print(f"Last dog {env_ids[last_dog_idx]}: Original vel: {original_vel}, Avoidance vel: {avoidance_vel}, "
                    f"Final vel: {combined_vel}")

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _resample_command_demo2(self, env_ids: Sequence[int]):
        # 首先获取当前位置信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)

        # 生成基础速度命令
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # 计算避障速度调整
        num_dogs = len(env_ids)
        avoidance_velocities = torch.zeros((num_dogs, 2), device=self.device)
        
        # 避障参数设置
        min_distance = 0.8  # 最小安全距离
        max_distance = 1.5  # 最大影响距离
        max_avoidance_force = 0.8  # 最大避障力
        
        # 只对最后两只狗进行避障
        if num_dogs >= 2:
            last_two_dogs = [num_dogs-2, num_dogs-1]  # 最后两只狗的索引
            
            for i in last_two_dogs:
                for j in range(num_dogs):
                    if i != j:  # 不与自己比较
                        # 计算两只狗之间的距离向量
                        pos_diff = base_pos_w_np[i] - base_pos_w_np[j]
                        distance = np.linalg.norm(pos_diff)
                        
                        if distance < max_distance:
                            # 计算避障力的强度（距离越近，力越大）
                            if distance < min_distance:
                                force_magnitude = max_avoidance_force
                            else:
                                # 在min_distance和max_distance之间平滑过渡
                                force_magnitude = max_avoidance_force * (1 - (distance - min_distance)/(max_distance - min_distance))
                            
                            # 计算避障方向（归一化）
                            if distance > 0:
                                avoid_dir = pos_diff / distance
                            else:
                                avoid_dir = np.array([1.0, 0.0])  # 如果距离为0，给一个默认方向
                            
                            # 将避障力转换为速度调整
                            avoidance_velocities[i] += torch.tensor(avoid_dir * force_magnitude, device=self.device)
        
        # 应用避障速度调整（只对最后两只狗）
        if num_dogs >= 2:
            for i in [num_dogs-2, num_dogs-1]:
                original_vel = self.vel_command_b[env_ids[i], :2]
                avoidance_vel = avoidance_velocities[i]
                
                # 计算合成速度
                combined_vel = original_vel + avoidance_vel
                
                # 限制最大速度
                max_speed = torch.norm(original_vel)  # 使用原始速度的大小作为最大速度
                current_speed = torch.norm(combined_vel)
                
                if current_speed > max_speed:
                    combined_vel = combined_vel * (max_speed / current_speed)
                
                # 更新速度命令
                self.vel_command_b[env_ids[i], :2] = combined_vel
                
                # 打印调试信息
                if torch.any(avoidance_velocities[i] != 0):
                    print(f"Dog {env_ids[i]}: Original vel: {original_vel}, Avoidance vel: {avoidance_vel}, "
                        f"Final vel: {combined_vel}")

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

        # print("====position=====", base_pos_w_pr, current_waypoint, distance_to_target, reached_target)

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
class UniformVelocityNavigationCommand_ori(UniformVelocityCommand):
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
        reached_targets = distances < 0.5  # (N,)

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


class UniformVelocityNavigationCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) for navigation tasks."""

    cfg: UniformVelocityNavigationCommandCfg

    def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 添加避障相关参数
        self.forward_check_distance = 1.0  # 前进方向检查距离
        self.stop_dogs = {}  # 记录需要停止的狗

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        # 重置初始索引
        reset_indices = (self.command_counter[env_ids] <= 1)
        if reset_indices.any():
            self.waypointIndex[env_ids[reset_indices]] = 0
            # 重置停止状态
            for idx in env_ids[reset_indices]:
                if idx.item() in self.stop_dogs:
                    del self.stop_dogs[idx.item()]

        # 获取当前位置和朝向信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)
        base_heading = self.robot.data.heading_w[env_ids].cpu().numpy()  # 获取当前朝向

        # 随机选择轨迹
        num_dogs = len(env_ids)
        selected_trajectories = [i % len(self.cfg.waypoints) for i in range(num_dogs)]
        global current_waypoints

        base_point = np.array([2.52, 2.69])

        # 检查每只狗的前进方向是否有其他狗
        for i in range(num_dogs):
            env_id = env_ids[i].item()
            current_pos = base_pos_w_np[i]
            current_heading = base_heading[i]
            
            # 计算前进方向向量
            forward_dir = np.array([np.cos(current_heading), np.sin(current_heading)])
            
            # 检查其他狗是否在前进方向上
            should_stop = False
            for j in range(num_dogs):
                if i != j:  # 不与自己比较
                    other_pos = base_pos_w_np[j]
                    # 计算到其他狗的方向向量
                    to_other = other_pos - current_pos
                    distance = np.linalg.norm(to_other)
                    
                    if distance < self.forward_check_distance:
                        # 计算到其他狗的方向与前进方向的夹角
                        to_other_dir = to_other / distance
                        dot_product = np.dot(forward_dir, to_other_dir)
                        
                        # 如果夹角小于45度（cos(45°) ≈ 0.707），说明其他狗在前进方向上
                        if dot_product > 0.707:
                            should_stop = True
                            break
            
            # 更新停止状态
            if should_stop:
                if env_id not in self.stop_dogs:
                    print(f"Dog {env_id} stopping due to dog in forward path")
                self.stop_dogs[env_id] = True
            else:
                if env_id in self.stop_dogs:
                    print(f"Dog {env_id} resuming movement")
                self.stop_dogs[env_id] = False

        # 处理轨迹选择和避障
        for i in range(num_dogs):
            env_id = env_ids[i].item()
            trajectory = self.cfg.waypoints[selected_trajectories[i]]
            if base_pos_w_np[i][0] > 2.5:
                trajectory = base_point + trajectory

            # 如果狗需要停止，设置速度为0
            if self.stop_dogs.get(env_id, False):
                self.vel_command_b[env_id, :2] = torch.zeros(2, device=self.device)
                continue

            # 在轨迹点上添加微扰
            if env_id not in current_waypoints:
                perturbed_points = [
                    [point[0] + np.random.uniform(-0.2, 0.2), point[1] + np.random.uniform(-0.2, 0.2)]
                    for point in trajectory
                ]
                current_waypoints[env_id] = perturbed_points

            # 获取当前目标点
            current_idx = self.waypointIndex[env_id].item()
            if current_idx < len(current_waypoints[env_id]):
                target_point = current_waypoints[env_id][current_idx]
                # 计算目标方向
                heading = target_point - base_pos_w_np[i]
                heading_angle = math.atan2(heading[1], heading[0])
                self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)

        # 更新到达目标点的机器狗的目标点索引
        for i, env_id in enumerate(env_ids):
            if self.stop_dogs.get(env_id.item(), False):
                continue  # 如果狗停止，不更新轨迹点索引
            
            current_idx = self.waypointIndex[env_id].item()
            if current_idx < len(current_waypoints[env_id.item()]):
                target_point = current_waypoints[env_id.item()][current_idx]
                distance = np.linalg.norm(base_pos_w_np[i] - target_point)
                
                if distance < 0.6:  # 到达目标点
                    next_idx = (current_idx + 1) % len(current_waypoints[env_id.item()])
                    self.waypointIndex[env_id] = next_idx

        # 打印调试信息
        for i, env_id in enumerate(env_ids):
            status = "stopped" if self.stop_dogs.get(env_id.item(), False) else "moving"
            print(f"Dog {env_id}: pos={base_pos_w_np[i]}, status={status}, "
                  f"target={current_waypoints[env_id.item()][self.waypointIndex[env_id].item()]}")


class UniformVelocityNavigationCommand_demo(UniformVelocityCommand):

    cfg: UniformVelocityNavigationCommandCfg

    def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 添加避障相关参数
        self.min_distance = 0.8  # 最小安全距离
        self.max_distance = 1.5  # 最大影响距离
        self.max_avoidance_force = 3.0  # 最大避障力
        self.acceleration_factor = 1.5  # 加速因子

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
        base_vel_w = self.robot.data.root_lin_vel_b[env_ids, :2].cpu().numpy()  # 获取当前速度

        # 随机选择轨迹
        num_dogs = len(env_ids)
        selected_trajectories = [i % len(self.cfg.waypoints) for i in range(num_dogs)]
        global current_waypoints

        base_point = np.array([2.52, 2.69])

        # 检查狗之间的距离，计算避障力
        avoidance_forces = np.zeros((num_dogs, 2))
        for i in range(num_dogs):
            for j in range(num_dogs):
                if i != j:  # 不与自己比较
                    # 计算距离和相对速度
                    pos_diff = base_pos_w_np[i] - base_pos_w_np[j]
                    vel_diff = base_vel_w[i] - base_vel_w[j]
                    distance = np.linalg.norm(pos_diff)
                    
                    if distance < self.max_distance:
                        # 计算避障力的强度
                        if distance < self.min_distance:
                            force_magnitude = self.max_avoidance_force
                        else:
                            # 平滑过渡
                            force_magnitude = self.max_avoidance_force * (1 - (distance - self.min_distance)/(self.max_distance - self.min_distance))
                        
                        # 计算避障方向
                        if distance > 0:
                            avoid_dir = pos_diff / distance
                        else:
                            avoid_dir = np.array([1.0, 0.0])
                        
                        # 考虑相对速度的影响
                        # vel_factor = np.dot(vel_diff, avoid_dir) / (np.linalg.norm(vel_diff) + 1e-6)
                        # force_magnitude *= (1 + vel_factor)  # 相对速度越大，避障力越大
                        
                        avoidance_forces[i] += avoid_dir * force_magnitude

        # 处理轨迹选择和避障
        for i in range(num_dogs):
            trajectory = self.cfg.waypoints[selected_trajectories[i]]
            if base_pos_w_np[i][0] > 2.5:
                trajectory = base_point + trajectory
            
            # 在轨迹点上添加微扰，考虑避障力
            perturbed_points = []
            for point in trajectory:
                # 基础扰动
                perturb_x = np.random.uniform(-0.2, 0.2)
                perturb_y = np.random.uniform(-0.2, 0.2)
                
                # 添加避障扰动
                if np.any(avoidance_forces[i] != 0):
                    perturb_x += avoidance_forces[i][0] * 0.3
                    perturb_y += avoidance_forces[i][1] * 0.3
                
                perturbed_points.append([point[0] + perturb_x, point[1] + perturb_y])
            
            if env_ids[i].item() not in current_waypoints:
                current_waypoints[env_ids[i].item()] = perturbed_points

        # 计算到目标点的距离和更新索引
        distances = []
        for i in range(num_dogs):
            current_idx = self.waypointIndex[env_ids[i]].item()
            if current_idx < len(current_waypoints[env_ids[i].item()]):
                target_point = current_waypoints[env_ids[i].item()][current_idx]
                distance = np.linalg.norm(base_pos_w_np[i] - target_point)
                distances.append(distance)
            else:
                distances.append(float('inf'))

        distances = np.array(distances)
        reached_targets = distances < 0.6

        # 更新目标方向和速度
        for i, (env_id, reached) in enumerate(zip(env_ids, reached_targets)):
            if reached:
                next_idx = (self.waypointIndex[env_id] + 1) % len(current_waypoints[env_ids[i].item()])
                self.waypointIndex[env_id] = next_idx
                if next_idx < len(current_waypoints[env_ids[i].item()]):
                    target_point = current_waypoints[env_ids[i].item()][next_idx]
            else:
                current_idx = self.waypointIndex[env_id]
                if current_idx < len(current_waypoints[env_ids[i].item()]):
                    target_point = current_waypoints[env_ids[i].item()][current_idx]
                else:
                    continue

            # 计算目标方向
            heading = target_point - base_pos_w_np[i]
            heading_angle = math.atan2(heading[1], heading[0])
            
            # 考虑避障力调整目标方向
            if np.any(avoidance_forces[i] != 0):
                # 计算避障方向的角度
                avoid_angle = math.atan2(avoidance_forces[i][1], avoidance_forces[i][0])
                # 将避障角度的影响加入到目标方向中
                heading_angle = heading_angle + 0.3 * (avoid_angle - heading_angle)
            
            self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)
            
            # 调整速度大小
            if np.any(avoidance_forces[i] != 0):
                # 当需要避障时，增加速度
                self.vel_command_b[env_id, :2] *= self.acceleration_factor
                print(f"Dog {env_id} accelerating for avoidance")

        # 打印调试信息
        for i, env_id in enumerate(env_ids):
            if np.any(avoidance_forces[i] != 0):
                print(f"Dog {env_id}: pos={base_pos_w_np[i]}, avoidance_force={avoidance_forces[i]}, "
                      f"heading={self.heading_target[env_id]}, vel={self.vel_command_b[env_id, :2]}")

class UniformVelocityNavigationCommand_demo2(UniformVelocityCommand):

    cfg: UniformVelocityNavigationCommandCfg

    def __init__(self, cfg: UniformVelocityNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.waypointIndex = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 添加避障相关参数
        self.min_distance = 0.8  # 最小安全距离
        self.max_distance = 3.0  # 最大影响距离
        self.max_avoidance_force = 10.0  # 最大避障力
        self.acceleration_factor = 1.5  # 加速因子
        # 添加避障目标点存储
        self.avoidance_targets = {}  # 存储每只狗的避障目标点
        self.original_targets = {}  # 存储每只狗的原始目标点

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        # 重置初始索引
        reset_indices = (self.command_counter[env_ids] <= 1)
        if reset_indices.any():
            self.waypointIndex[env_ids[reset_indices]] = 0
            # 重置避障目标点
            for idx in env_ids[reset_indices]:
                if idx.item() in self.avoidance_targets:
                    del self.avoidance_targets[idx.item()]
                if idx.item() in self.original_targets:
                    del self.original_targets[idx.item()]

        # 获取当前位置信息
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w_np = base_pos_w[env_ids, :2].cpu().numpy()  # (N, 2)
        base_vel_w = self.robot.data.root_lin_vel_b[env_ids, :2].cpu().numpy()  # 获取当前速度

        # 随机选择轨迹
        num_dogs = len(env_ids)
        selected_trajectories = [i % len(self.cfg.waypoints) for i in range(num_dogs)]
        global current_waypoints

        base_point = np.array([2.52, 2.69])

        # 计算避障力
        avoidance_forces = np.zeros((num_dogs, 2))
        for i in range(num_dogs):
            for j in range(num_dogs):
                if i != j:  # 不与自己比较
                    pos_diff = base_pos_w_np[i] - base_pos_w_np[j]
                    vel_diff = base_vel_w[i] - base_vel_w[j]
                    distance = np.linalg.norm(pos_diff)
                    
                    if distance < self.max_distance:
                        if distance < self.min_distance:
                            force_magnitude = self.max_avoidance_force
                        else:
                            force_magnitude = self.max_avoidance_force * (1 - (distance - self.min_distance)/(self.max_distance - self.min_distance))
                        
                        if distance > 0:
                            avoid_dir = pos_diff / distance
                        else:
                            avoid_dir = np.array([1.0, 0.0])
                        
                        # 考虑相对速度的影响
                        vel_factor = np.dot(vel_diff, avoid_dir) / (np.linalg.norm(vel_diff) + 1e-6)
                        force_magnitude *= (1 + vel_factor)
                        
                        avoidance_forces[i] += avoid_dir * force_magnitude

        # 处理轨迹选择和避障
        for i in range(num_dogs):
            env_id = env_ids[i].item()
            trajectory = self.cfg.waypoints[selected_trajectories[i]]
            if base_pos_w_np[i][0] > 2.5:
                trajectory = base_point + trajectory

            # 获取当前目标点
            current_idx = self.waypointIndex[env_id].item()
            if current_idx < len(trajectory):
                current_target = trajectory[current_idx]
            else:
                continue

            # 检查是否需要避障
            if np.any(avoidance_forces[i] != 0):
                # 如果还没有避障目标点，创建一个新的
                if env_id not in self.avoidance_targets:
                    # 计算避障目标点（当前位置 + 避障方向 * 安全距离）
                    avoid_dir = avoidance_forces[i] / (np.linalg.norm(avoidance_forces[i]) + 1e-6)
                    self.avoidance_targets[env_id] = base_pos_w_np[i] + avoid_dir * self.min_distance * 2
                    self.original_targets[env_id] = current_target
                    print(f"Dog {env_id} creating avoidance target: {self.avoidance_targets[env_id]}")
                
                # 使用避障目标点
                target_point = self.avoidance_targets[env_id]
                # 增加速度
                self.vel_command_b[env_id, :2] *= self.acceleration_factor
            else:
                # 如果不需要避障，检查是否需要恢复原始目标点
                if env_id in self.avoidance_targets:
                    # 检查是否到达避障目标点
                    dist_to_avoidance = np.linalg.norm(base_pos_w_np[i] - self.avoidance_targets[env_id])
                    if dist_to_avoidance < 0.3:  # 到达避障目标点
                        print(f"Dog {env_id} reached avoidance target, returning to original path")
                        del self.avoidance_targets[env_id]
                        del self.original_targets[env_id]
                        target_point = current_target
                    else:
                        target_point = self.avoidance_targets[env_id]
                else:
                    target_point = current_target

            # 计算目标方向
            heading = target_point - base_pos_w_np[i]
            heading_angle = math.atan2(heading[1], heading[0])
            self.heading_target[env_id] = torch.tensor(heading_angle).to(self.device)

            # 在轨迹点上添加微扰
            if env_id not in current_waypoints:
                perturbed_points = [
                    [point[0] + np.random.uniform(-0.2, 0.2), point[1] + np.random.uniform(-0.2, 0.2)]
                    for point in trajectory
                ]
                current_waypoints[env_id] = perturbed_points

        # 更新到达目标点的机器狗的目标点索引
        for i, env_id in enumerate(env_ids):
            if env_id.item() in self.avoidance_targets:
                continue  # 如果正在避障，不更新轨迹点索引
            
            current_idx = self.waypointIndex[env_id].item()
            if current_idx < len(current_waypoints[env_id.item()]):
                target_point = current_waypoints[env_id.item()][current_idx]
                distance = np.linalg.norm(base_pos_w_np[i] - target_point)
                
                if distance < 0.6:  # 到达目标点
                    next_idx = (current_idx + 1) % len(current_waypoints[env_id.item()])
                    self.waypointIndex[env_id] = next_idx

        # 打印调试信息
        for i, env_id in enumerate(env_ids):
            status = "avoiding" if env_id.item() in self.avoidance_targets else "normal"
            print(f"Dog {env_id}: pos={base_pos_w_np[i]}, status={status}, "
                  f"target={self.avoidance_targets.get(env_id.item(), current_waypoints[env_id.item()][self.waypointIndex[env_id].item()])}")
