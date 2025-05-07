# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    print("===========reward.feet_air_time=============", reward)
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    print("===========reward.feet_air_time_positive_biped=============", reward)
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    print("===========reward.feet_slide=============", reward)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), height_threshold: float = 1.0) -> torch.Tensor:
    """Reward for achieving a height above a certain threshold."""
    asset = env.scene[asset_cfg.name]
    # 获取机器狗的 z 方向位置
    z_position = asset.data.root_pos_w[:, 2]
    # 计算奖励
    reward = (z_position - height_threshold).clamp(min=0)  # 只有当 z_position 大于 height_threshold 时才有奖励
    print("===========reward.height_reward=============", reward)
    return reward

def stability_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    """Reward for maintaining stability while climbing."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触状态
    contacts = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 计算稳定性奖励
    stability = (contacts > threshold).float()  # 如果接触时间大于阈值，则认为稳定
    return stability.sum(dim=1)  # 返回每个环境的稳定性奖励

def foot_height_difference_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, height_threshold: float = 0.10, height_top_threshold: float = 0.20, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for the highest two feet being higher than the lowest two feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    # print("========asset.get_state========", asset.data.body_state_w[:, :])
    body_state = asset.data.body_state_w
    # 获取所有脚的 z 方向位置
    foot_positions = body_state[..., 2]  #.cpu().numpy()  # 获取所有body的 z 位置
    
    # 获取脚
    foot = -((-foot_positions).topk(4, dim=-1).values)
    highest_one = foot.topk(1, dim=-1).values  # 获取最高的脚
    lowest_one = -((-foot).topk(1, dim=-1).values)  # 获取最低的脚
    
    # 计算高度差
    height_difference = highest_one - lowest_one
    
    # 计算奖励
    offset = height_top_threshold - height_threshold
    reward = (height_difference - height_threshold).clamp(min=0).reshape(-1)  # 只有当高度差大于阈值时才有奖励
    reward = torch.where(reward > offset, torch.tensor(0.0, device=reward.device), reward).reshape(-1)

    print("===========reward.foot_height_difference_reward=============", reward)
    
    return reward  # 返回每个 agent 的总奖励

def distance_to_target_reward(env: ManagerBasedRLEnv, target_position: tuple = (-31.5, 22.3), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for getting closer to the target position."""
    asset = env.scene[asset_cfg.name]
    
    # 获取机器狗的当前位置
    current_position = asset.data.root_pos_w[:, :2]  # 获取 x 和 y 坐标
    
    # 计算与目标点之间的距离
    distance = torch.norm(current_position - torch.tensor(target_position, device=current_position.device), dim=1)
    
    # 计算奖励（距离越小，奖励越高）
    reward = -distance  # 奖励为负距离，鼓励靠近目标点
    print("===========reward.distance_to_target_reward=============", reward)
    return reward  # 返回每个环境的总奖励