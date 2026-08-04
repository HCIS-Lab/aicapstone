from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData

from leisaac.devices.action_process import preprocess_device_action as preprocess_device_action_common
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg

from simulator.assets.robots.franka import FRANKA_PANDA_CFG
from simulator import FRANKA_JOINT_NAMES
from simulator.utils.object_poses_loader import ObjectPoseConfig

from . import mdp


def euler_deg_to_quat(x, y, z):
    import math
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    q = quat_from_euler_xyz(
        torch.tensor([math.radians(x)], dtype=torch.float32),
        torch.tensor([math.radians(y)], dtype=torch.float32),
        torch.tensor([math.radians(z)], dtype=torch.float32),
    )[0]
    return tuple(float(v) for v in q)


@configclass
class SingleArmFrankaTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the single arm task."""

    scene: AssetBaseCfg = MISSING
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.04, 0.0, 0.0), rot=(0.707, 0, 0, 0.707), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # NOTE: must live under {ENV_REGEX_NS} so InteractiveScene clones one
    # camera per environment. A global "/World/front_camera" yields a single
    # sensor instance (view count == 1) while the scene reports num_envs > 1,
    # so resetting env_ids beyond 0 indexes the size-1 sensor buffers out of
    # bounds (CUDA device-side assert in SensorBase.reset). The offset below
    # is now interpreted relative to each environment's origin.
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.35, 1.1, 0.6), rot=(0.0, -0.0, -0.60182, -0.79864), convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=55,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class SingleArmFrankaEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class SingleArmFrankaObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False})
        front = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False})
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SingleArmFrankaActionsCfg:
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class SingleArmFrankaRewardsCfg:
    pass


@configclass
class SingleArmFrankaTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SingleArmFrankaTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: SingleArmFrankaTaskSceneCfg = MISSING
    observations: SingleArmFrankaObservationsCfg = MISSING
    actions: SingleArmFrankaActionsCfg = SingleArmFrankaActionsCfg()
    events: SingleArmFrankaEventCfg = SingleArmFrankaEventCfg()
    rewards: SingleArmFrankaRewardsCfg = SingleArmFrankaRewardsCfg()
    terminations: SingleArmFrankaTerminationsCfg = MISSING
    recorders: RecordTerm = RecordTerm()
    dynamic_reset_gripper_effort_limit: bool = True
    object_pose_cfg: ObjectPoseConfig | None = None
    robot_name: str = "franka_panda"
    default_feature_joint_names: list[str] = MISSING
    task_description: str = MISSING
    teleop_target_frame: str = "panda_hand"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 15
        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (2.0, -0.5, 1.0)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True
        self.default_feature_joint_names = [f"{joint_name}.pos" for joint_name in FRANKA_JOINT_NAMES]

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        if teleop_device not in ["keyboard", "gamepad"]:
            raise ValueError(f"Franka teleoperation only supports keyboard/gamepad, got '{teleop_device}'.")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )
        self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        if action.get("keyboard") is not None or action.get("gamepad") is not None:
            processed_action = torch.zeros(teleop_device.env.num_envs, 8, device=teleop_device.env.device)
            processed_action[:, :] = action["joint_state"]
            return processed_action
        return preprocess_device_action_common(action, teleop_device)

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = episode_data._data["actions"][-1]
        frame = {
            "action": action.cpu().numpy(),
            "observation.state": obs_data["joint_pos"][-1].cpu().numpy(),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()
        return frame
