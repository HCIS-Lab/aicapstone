#!/usr/bin/env python3
# Copyright (c) 2025. HW1 — Camera Pose Estimation in IsaacLab.
# SPDX-License-Identifier: BSD-3-Clause
"""HW1 — Interactive camera-pose-estimation pipeline on top of IsaacLab.

This is an intentionally *flat*, single-file MVP. The goal is to validate the
end-to-end idea, not to build a reusable library, so abstractions are kept to a
minimum and the whole pipeline lives here.

Pipeline
--------
1. Drive a Turtlebot3 (burger) around an IsaacLab *Warehouse* scene with the
   keyboard (W/S/A/D). While driving we accumulate, per captured keyframe:
       - RGB image                      (H, W, 3) uint8
       - depth image                    (H, W)    float32   [metric, z-buffer]
       - camera intrinsics              (3, 3)    float32
       - camera world pose              pos [x, y, z] + quat [qw, qx, qy, qz]
   The world pose is the *ground-truth* pose reported by the simulator; we keep
   it both as a label and as an init/eval reference for the estimator.

2. On request (press F) we save the raw keyframe buffer to <out>/keyframes.npz
   for offline reconstruction by `scripts/reconstruct.py`.

Everything in the simulation half (scene / robot / sensor / control) uses only
the IsaacLab API. The geometric registration half (depth unprojection /
FPFH/RANSAC/ICP) lives in the standalone `scripts/reconstruct.py`, which only
needs Open3D + numpy and runs outside the simulator.

Run
---
    # inside the IsaacLab python env / container
    python scripts/hw1.py                       # GUI, drive with W/S/A/D
    python scripts/hw1.py --warehouse_usd <p> --robot_usd <p>

Keys
----
    W / S      drive forward / backward
    A / D      turn left / right
    R          capture the current frame as a keyframe
    C          clear the keyframe buffer
    F          save the raw keyframe buffer to <out>/keyframes.npz
    L          reset robot to spawn pose
    Q / ESC    quit the simulation
"""

from __future__ import annotations

import argparse

# -----------------------------------------------------------------------------
# 0. Launch the Omniverse / Isaac Sim app FIRST.
#    Almost every `isaaclab.*` import below pulls in `omni.*` / `carb`, which
#    only exist once the app is up, so the AppLauncher has to come before them.
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HW1 — IsaacLab camera pose estimation.")
parser.add_argument(
    "--assets_root",
    type=str,
    default=None,
    help="Root that the relative --warehouse_usd / --robot_usd paths are joined to. "
    "Defaults to the Isaac nucleus assets dir.",
)
parser.add_argument(
    "--warehouse_usd",
    type=str,
    default="Environments/Simple_Warehouse/warehouse.usd",
    help="Warehouse USD path (absolute, or relative to --assets_root).",
)
parser.add_argument(
    "--robot_usd",
    type=str,
    default="Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    help="Turtlebot3 USD path (absolute, or relative to --assets_root).",
)
parser.add_argument("--out", type=str, default="outputs/hw1", help="Output directory.")
parser.add_argument("--width", type=int, default=640, help="Camera image width.")
parser.add_argument("--height", type=int, default=480, help="Camera image height.")
parser.add_argument(
    "--capture_every",
    type=int,
    default=0,
    help="If > 0, auto-capture a keyframe every N control steps (in addition to 'R').",
)
# Inject the standard AppLauncher CLI args (--headless, --device, etc.).
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Teleoperation is inherently single-environment.
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 1. Now it is safe to import the rest.
# -----------------------------------------------------------------------------
import os
import weakref
from dataclasses import MISSING

import gymnasium as gym
import numpy as np
import torch

import carb
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp


# =============================================================================
# Asset path resolution
# =============================================================================
def _resolve(path: str, root: str) -> str:
    """Return `path` if it already looks absolute (local or omniverse), else join with root."""
    if path.startswith(("/", "omniverse://", "http://", "https://")) or os.path.isabs(path):
        return path
    return f"{root}/{path}"


ASSETS_ROOT = args_cli.assets_root if args_cli.assets_root is not None else ISAAC_NUCLEUS_DIR
WAREHOUSE_USD = _resolve(args_cli.warehouse_usd, ASSETS_ROOT)
ROBOT_USD=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"

# Jetbot kinematic constants (used only for diff-drive command mapping; approximate).
WHEEL_RADIUS = 0.0325  # [m]  (65 mm wheel diameter)
WHEEL_SEPARATION = 0.118  # [m]
# Body link the camera is rigidly attached to. The Jetbot's root body link.
CAMERA_PARENT_LINK = "chassis"
# Wheel joint names of the Jetbot model.
WHEEL_JOINTS = ["left_wheel_joint", "right_wheel_joint"]


# =============================================================================
# 2. Scene: warehouse + light + turtlebot3 + RGB-D camera
# =============================================================================
@configclass
class HW1SceneCfg(InteractiveSceneCfg):
    """Warehouse scene with a Turtlebot3 carrying a forward-facing RGB-D camera."""

    # -- Static environment (the warehouse USD). AssetBaseCfg = no articulation/rigid wrapper.
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
    )

    # -- Lighting. A dome light gives flat, even ambient illumination for clean RGB-D.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)),
    )

    # -- Robot: Jetbot, velocity-driven wheels.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
            ),
        ),
        actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

    # -- RGB-D camera, rigidly mounted on the robot base, looking forward.
    #    `distance_to_image_plane` is the metric z-buffer depth used for unprojection.
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + CAMERA_PARENT_LINK + "/front_camera",
        update_period=0.0,  # update every render
        width=args_cli.width,
        height=args_cli.height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        # Mount ~12cm above the base, tilted slightly down. ROS convention => +Z forward.
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.12),
            rot=(0.5, -0.5, 0.5, -0.5),  # optical frame: +Z forward, -Y up (ROS), facing +X world
            convention="ros",
        ),
    )


# =============================================================================
# 3. MDP managers — actions / observations / terminations
# =============================================================================
@configclass
class ActionsCfg:
    """Direct joint-velocity control of the two drive wheels.

    The action is a 2-vector of wheel *angular* velocity targets [left, right]
    in rad/s. The keyboard (v, omega) command is converted to wheel speeds in
    the main loop via differential-drive kinematics.
    """

    drive = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,  # action[:, 0] = left, action[:, 1] = right
    )


@configclass
class ObservationsCfg:
    """Minimal observation group (teleop does not consume observations)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Only time-out; teleop episodes are effectively open-ended."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset the robot to its spawn root pose on episode reset."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """No rewards — this is a data-collection task, not RL."""

    pass


# =============================================================================
# 4. The environment configuration
# =============================================================================
@configclass
class HW1EnvCfg(ManagerBasedRLEnvCfg):
    scene: HW1SceneCfg = HW1SceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 1.0e9  # effectively never time out during teleop
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # Put the default viewer behind/above the robot.
        self.viewer.eye = (-3.0, -3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# -- Register as a Gymnasium environment (IsaacLab manager-based pattern).
GYM_ID = "Isaac-HW1-Turtlebot-Warehouse-v0"
gym.register(
    id=GYM_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": HW1EnvCfg},
)


# =============================================================================
# 5. Keyboard controller (W/S/A/D + action keys) via carb input.
#    Built directly on the omni/carb input stack that IsaacLab itself uses,
#    so we get continuous held-key velocity control and arbitrary key bindings.
# =============================================================================
class KeyboardController:
    """Tracks held W/S/A/D for (v, omega) and fires one-shot callbacks for action keys."""

    def __init__(self, lin_speed: float = 0.5, ang_speed: float = 1.5):
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self._pressed: set[str] = set()
        self._callbacks: dict[str, callable] = {}

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *a, obj=weakref.proxy(self): obj._on_event(event, *a),
        )

    def add_callback(self, key: str, func) -> None:
        self._callbacks[key] = func

    def _on_event(self, event, *args) -> bool:
        # `event.input` is a `carb.input.KeyboardInput` enum on most builds (use `.name`),
        # but some Isaac Sim builds hand it through already as a plain key-name string.
        ev_input = event.input
        key = ev_input.name if hasattr(ev_input, "name") else str(ev_input)  # e.g. "W", "A", "R"
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(key)
            if key in self._callbacks:
                self._callbacks[key]()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
        return True

    def command(self) -> tuple[float, float]:
        """Return (v [m/s], omega [rad/s]) from currently held W/S/A/D keys."""
        v = 0.0
        w = 0.0
        if "W" in self._pressed:
            v += self.lin_speed
        if "S" in self._pressed:
            v -= self.lin_speed
        if "A" in self._pressed:
            w += self.ang_speed
        if "D" in self._pressed:
            w -= self.ang_speed
        return v, w


def diff_drive(v: float, w: float) -> torch.Tensor:
    """(v, omega) -> [left, right] wheel angular velocities [rad/s] for a (1, 2) action."""
    v_l = (v - w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    v_r = (v + w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    return torch.tensor([[v_l, v_r]], dtype=torch.float32)


# =============================================================================
# 7. Main loop
# =============================================================================
def main() -> None:
    os.makedirs(args_cli.out, exist_ok=True)

    env_cfg = HW1EnvCfg()
    # Create through the Gymnasium registry; `.unwrapped` exposes the ManagerBasedRLEnv.
    env = gym.make(GYM_ID, cfg=env_cfg).unwrapped
    env.reset()

    camera = env.scene["camera"]
    device = env.device

    keyframes: list[dict] = []

    def grab_frame() -> dict | None:
        """Pull the current RGB-D + intrinsics + GT world pose off the camera sensor."""
        out = camera.data.output
        if out is None or "distance_to_image_plane" not in out:
            print("[HW1] Camera has no data yet.")
            return None
        rgb = out["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = out["distance_to_image_plane"][0].squeeze(-1).cpu().numpy().astype(np.float32)
        intr = camera.data.intrinsic_matrices[0].cpu().numpy().astype(np.float32)
        pos = camera.data.pos_w[0].cpu().numpy().astype(np.float32)          # [x, y, z]
        quat = camera.data.quat_w_ros[0].cpu().numpy().astype(np.float32)    # [qw, qx, qy, qz]
        return {"rgb": rgb, "depth": depth, "intrinsics": intr, "pos": pos, "quat": quat}

    def capture() -> None:
        f = grab_frame()
        if f is not None:
            keyframes.append(f)
            print(f"[HW1] captured keyframe #{len(keyframes)}  "
                  f"pos=({f['pos'][0]:+.2f},{f['pos'][1]:+.2f},{f['pos'][2]:+.2f})")

    def clear() -> None:
        keyframes.clear()
        print("[HW1] cleared keyframe buffer.")

    def save_raw() -> None:
        if not keyframes:
            print("[HW1] nothing to save.")
            return
        path = os.path.join(args_cli.out, "keyframes.npz")
        np.savez_compressed(
            path,
            rgb=np.stack([f["rgb"] for f in keyframes]),
            depth=np.stack([f["depth"] for f in keyframes]),
            intrinsics=np.stack([f["intrinsics"] for f in keyframes]),
            pos=np.stack([f["pos"] for f in keyframes]),
            quat=np.stack([f["quat"] for f in keyframes]),
        )
        print(f"[HW1] saved {len(keyframes)} keyframes -> {path}")

    def request_quit() -> None:
        print("[HW1] quit requested — exiting simulation loop.")
        simulation_app.close()

    kb = KeyboardController(lin_speed=0.5, ang_speed=1.5)
    kb.add_callback("R", capture)
    kb.add_callback("C", clear)
    kb.add_callback("F", save_raw)
    kb.add_callback("L", lambda: env.reset())
    kb.add_callback("Q", request_quit)
    kb.add_callback("ESCAPE", request_quit)

    print(
        "\n[HW1] Ready.\n"
        "      W/S = forward/back, A/D = turn,\n"
        "      R = capture keyframe, C = clear, F = save, L = reset, Q/ESC = quit.\n"
    )

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            v, w = kb.command()
            action = diff_drive(v, w).to(device)
            env.step(action)

            if args_cli.capture_every > 0 and (v != 0.0 or w != 0.0):
                if step % args_cli.capture_every == 0:
                    capture()
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
