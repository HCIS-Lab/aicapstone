from __future__ import annotations

import carb
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from leisaac.devices.device_base import Device


class FrankaKeyboard(Device):
    """Keyboard teleop for Franka with device-side differential IK.

    Reads SE(3) deltas from key presses and integrates them into a latched
    target pose held in the robot base frame. The target is an absolute
    setpoint, so an axis the user is not driving stays put instead of
    following whatever pose the arm happens to have drifted to. Differential
    IK solves against that target and emits an 8-D action matching the env's
    joint-position action space: ``[panda_joint_1..7 target, gripper_cmd]``
    where ``gripper_cmd`` is a latched scalar (>= 0 open, < 0 close) consumed
    by ``BinaryJointPositionActionCfg``.
    """

    _GRIPPER_OPEN = 1.0
    _GRIPPER_CLOSE = -1.0

    def __init__(
        self,
        env,
        sensitivity: float = 1.0,
        frame: str = "ee",
        max_pos_lead: float = 0.02,
        max_rot_lead: float = 0.10,
        smoothing: float = 0.15,
    ):
        """
        Args:
            env: The environment holding the robot to control.
            sensitivity: Scales the per-step position and rotation deltas.
            frame: Frame the key deltas are expressed in. ``"ee"`` moves along
                the ``panda_hand`` axes, ``"base"`` along the robot base axes.
            max_pos_lead: Max distance [m] the target may lead the measured ee
                position. Keeps the target from running away when IK cannot
                track (joint limits, contact), and keeps the tracking error
                small enough that the joint PD does not saturate.
            max_rot_lead: Max angle [rad] the target may lead the measured ee
                orientation.
            smoothing: First-order ramp coefficient applied to the key deltas,
                in (0, 1]. Key press/release is a step, and feeding a step
                straight into the target makes the arm jerk; the ramp gives a
                time constant of ``env_dt / smoothing``. Use 1.0 to disable.
        """
        super().__init__(env, "keyboard")

        # Per control step, at decimation 1 and sim dt 1/60 this is 0.24 m/s and
        # ~1.2 rad/s. Faster than this outruns what the arm's PD can track.
        self.pos_sensitivity = 0.004 * sensitivity
        self.rot_sensitivity = 0.02 * sensitivity
        self._smoothing = float(smoothing)

        if frame not in ("ee", "base"):
            raise ValueError(f"frame must be 'ee' or 'base', got {frame!r}")
        self.frame = frame
        self._max_pos_lead = max_pos_lead
        self._max_rot_lead = max_rot_lead

        self._create_key_bindings()

        # (dx, dy, dz, drx, dry, drz, gripper_latch); rotation is a rotation
        # vector (axis-angle), matching what the IK controller consumes.
        self._delta_action = np.zeros(7)
        self._delta_action[6] = self._GRIPPER_OPEN
        # ramped copy of _delta_action[:6], what actually drives the target
        self._delta_smoothed = np.zeros(6)

        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "panda_hand"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self._body_idx = body_idxs[0]
        self.target_frame_idx = self._body_idx

        arm_joint_ids, _ = self.robot_asset.find_joints(["panda_joint.*"])
        self._arm_joint_ids = arm_joint_ids
        self._num_arm_joints = len(arm_joint_ids)

        if self.robot_asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = arm_joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in arm_joint_ids]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls", use_relative_mode=False
        )
        self._ik = DifferentialIKController(
            ik_cfg, num_envs=self.env.num_envs, device=self.env.device
        )

        # latched absolute target pose in base frame; seeded on first use
        self._target_pos_b: torch.Tensor | None = None
        self._target_quat_b: torch.Tensor | None = None

    def _add_device_control_description(self):
        rows = [
            ("W", "+x"), ("S", "-x"),
            ("A", "+y"), ("D", "-y"),
            ("J", "+z"), ("K", "-z"),
            ("H", "roll-"), ("L", "roll+"),
            ("U", "pitch-"), ("I", "pitch+"),
            ("Q", "yaw-"), ("E", "yaw+"),
            ("C", "gripper open"), ("M", "gripper close"),
        ]
        for key, desc in rows:
            self._display_controls_table.add_row([key, desc])

    def _ee_pose_b(self):
        """Current ee pose expressed in the robot base frame."""
        ee_pos_w = self.robot_asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self.robot_asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self.robot_asset.data.root_pos_w
        root_quat_w = self.robot_asset.data.root_quat_w
        return math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

    def _jacobian_b(self):
        """Arm jacobian rotated from world into the robot base frame."""
        jac_w = self.robot_asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        root_quat_w = self.robot_asset.data.root_quat_w
        base_R = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_R, jac_b[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_R, jac_b[:, 3:, :])
        return jac_b

    def _integrate_target(self):
        """Advance the latched target pose by one step of key deltas."""
        num_envs = self.env.num_envs
        self._delta_smoothed += self._smoothing * (
            self._delta_action[:6] - self._delta_smoothed
        )
        delta = torch.tensor(
            self._delta_smoothed, device=self.env.device, dtype=torch.float32
        ).unsqueeze(0).repeat(num_envs, 1)
        d_pos = delta[:, :3]
        d_rotvec = delta[:, 3:6]

        if self.frame == "ee":
            # Deltas are given in the panda_hand frame. Rotate them into the
            # base frame using the *target* orientation, not the measured one,
            # so tracking error cannot feed back into the commanded direction.
            q = self._target_quat_b
            d_pos = math_utils.quat_apply(q, d_pos)
            d_rotvec = math_utils.quat_apply(q, d_rotvec)

        self._target_pos_b = self._target_pos_b + d_pos

        angle = torch.linalg.vector_norm(d_rotvec, dim=-1)
        if bool((angle > 1e-6).any()):
            axis = d_rotvec / angle.clamp(min=1e-9).unsqueeze(-1)
            identity = torch.zeros_like(self._target_quat_b)
            identity[:, 0] = 1.0
            dq = torch.where(
                (angle > 1e-6).unsqueeze(-1),
                math_utils.quat_from_angle_axis(angle, axis),
                identity,
            )
            # pre-multiply in base frame == body-frame rotation, since the
            # rotation vector was already mapped through the target orientation
            self._target_quat_b = math_utils.normalize(
                math_utils.quat_mul(dq, self._target_quat_b)
            )

    def _leash_target(self, ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor):
        """Clamp how far the target may lead the measured pose."""
        pos_err = self._target_pos_b - ee_pos_b
        dist = torch.linalg.vector_norm(pos_err, dim=-1, keepdim=True)
        pos_scale = (self._max_pos_lead / dist.clamp(min=1e-9)).clamp(max=1.0)
        self._target_pos_b = ee_pos_b + pos_err * pos_scale

        q_err = math_utils.quat_mul(self._target_quat_b, math_utils.quat_inv(ee_quat_b))
        rotvec = math_utils.axis_angle_from_quat(math_utils.quat_unique(q_err))
        ang = torch.linalg.vector_norm(rotvec, dim=-1)
        over = ang > self._max_rot_lead
        if bool(over.any()):
            axis = rotvec / ang.clamp(min=1e-9).unsqueeze(-1)
            clamped = math_utils.quat_from_angle_axis(
                ang.clamp(max=self._max_rot_lead), axis
            )
            q_err = torch.where(over.unsqueeze(-1), clamped, q_err)
            self._target_quat_b = math_utils.normalize(
                math_utils.quat_mul(q_err, ee_quat_b)
            )

    def get_device_state(self):
        gripper_cmd = float(self._delta_action[6])

        ee_pos_b, ee_quat_b = self._ee_pose_b()
        if self._target_pos_b is None:
            self._target_pos_b = ee_pos_b.clone()
            self._target_quat_b = ee_quat_b.clone()

        self._integrate_target()
        self._leash_target(ee_pos_b, ee_quat_b)

        command = torch.cat([self._target_pos_b, self._target_quat_b], dim=-1)
        self._ik.set_command(command)

        joint_pos = self.robot_asset.data.joint_pos[:, self._arm_joint_ids]
        joint_pos_des = self._ik.compute(
            ee_pos_b, ee_quat_b, self._jacobian_b(), joint_pos
        )

        action = torch.zeros(self._num_arm_joints + 1, device=self.env.device)
        action[: self._num_arm_joints] = joint_pos_des[0]
        action[self._num_arm_joints] = gripper_cmd
        return action.cpu().numpy()

    def reset(self):
        self._delta_action[:6] = 0.0
        self._delta_smoothed[:] = 0.0
        self._target_pos_b = None
        self._target_quat_b = None
        self._ik.reset()

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key in self._POSE_KEY_DELTAS:
                self._delta_action[:6] += self._POSE_KEY_DELTAS[key]
            elif key == "C":
                self._delta_action[6] = self._GRIPPER_OPEN
            elif key == "M":
                self._delta_action[6] = self._GRIPPER_CLOSE
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            if key in self._POSE_KEY_DELTAS:
                self._delta_action[:6] -= self._POSE_KEY_DELTAS[key]

    def _create_key_bindings(self):
        p = self.pos_sensitivity
        r = self.rot_sensitivity
        self._POSE_KEY_DELTAS: dict[str, np.ndarray] = {
            "W": np.array([+p, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "S": np.array([-p, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "A": np.array([0.0, -p, 0.0, 0.0, 0.0, 0.0]),
            "D": np.array([0.0, +p, 0.0, 0.0, 0.0, 0.0]),
            "J": np.array([0.0, 0.0, +p, 0.0, 0.0, 0.0]),
            "K": np.array([0.0, 0.0, -p, 0.0, 0.0, 0.0]),
            "H": np.array([0.0, 0.0, 0.0, -r, 0.0, 0.0]),
            "L": np.array([0.0, 0.0, 0.0, +r, 0.0, 0.0]),
            "U": np.array([0.0, 0.0, 0.0, 0.0, -r, 0.0]),
            "I": np.array([0.0, 0.0, 0.0, 0.0, +r, 0.0]),
            "Q": np.array([0.0, 0.0, 0.0, 0.0, 0.0, -r]),
            "E": np.array([0.0, 0.0, 0.0, 0.0, 0.0, +r]),
        }
