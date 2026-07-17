import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.utils.object_poses_loader import ObjectPoseConfig
from simulator.assets.scenes.dining_room import DINING_ROOM_CFG, DINING_ROOM_USD_PATH

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

DINING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "dining_room" / "objects"

# kujiale dining_table_0000 (light wood, convexDecomposition colliders, self-contained).
# Origin is at the table's vertical CENTER (spans z +-0.355), so place at z=0.355 to
# sit the base on the floor (top ends up at ~0.71).
DININGTABLE_USD = str(
    ASSETS_ROOT / "scenes" / "dining_table_0000" / "dining_table.usd"
)
DININGTABLE_WORLD_POS: tuple[float, float, float] = (7.0, 3.5, 0.354)
DININGTABLE_WORLD_ROT: tuple[float, float, float, float] = ( 0.70711, 0.0, 0.0, -0.70711)

TAG_TO_OBJECT: dict[int, str] = {2: "knife", 3: "fork"}
ANCHOR_TAG_ID: int = 0
# Anchor for fork/knife spawns; placed away from the fixed plate so the cutlery
# starts well clear of the plate area.
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.40, 0.10, 0.0)
OBJECT_Z: float = 1.00
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0
# Per-USD yaw correction (rad) so the spawned object matches its visual heading
# under the gripper's coordinate convention. Tune once per USD by viewing the
# spawned object and the printed yaw side-by-side.
PER_OBJECT_YAW_OFFSET: dict[str, float] = {
    "knife": math.pi,
    "fork": 2.0 * math.pi,
}
# Plate is spawned at a fixed position (see RigidObjectCfg below) and not
# loaded from object_poses.json; the JSON entry is silently skipped.
IGNORED_OBJECT_NAMES: tuple[str, ...] = ("plate",)
# Fixed plate world position. Robot is at (0.35, -0.74); plate sits in front of
# it with ≥ 10 cm of free space on both ±y sides for fork (left) and knife
# (right) drop targets (state machine uses `_PLACE_Y_OFFSET = 0.10`).
PLATE_WORLD_POS: tuple[float, float, float] = (7.0, 2.9, 0.74416) # (0.50, -0.40, 0.05)


@configclass
class CutleryArrangementSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the cutlery arrangement task."""

    scene: AssetBaseCfg = DINING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # Standalone white table (matte, static colliders baked in). Static furniture
    # -> AssetBaseCfg (no rigid body). Tune DININGTABLE_WORLD_POS for the scene.
    diningtable: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/diningtable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=DININGTABLE_WORLD_POS,
            rot=DININGTABLE_WORLD_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=DININGTABLE_USD,
        ),
    )

    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/plate",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PLATE_WORLD_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Plate" / "plate.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    knife: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/knife",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Knife" / "knife.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    fork: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/fork",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Fork" / "fork.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )


def cutlery_arranged(
    env,
    plate_cfg: SceneEntityCfg,
    fork_cfg: SceneEntityCfg,
    knife_cfg: SceneEntityCfg,
    max_dist_xy: float,
) -> torch.Tensor:
    """Termination: fork on +y side of plate, knife on -y side, both within max_dist_xy."""
    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]

    plate_pos = plate.data.root_pos_w - env.scene.env_origins
    fork_pos = fork.data.root_pos_w - env.scene.env_origins
    knife_pos = knife.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    fork_dist_xy = torch.norm(fork_pos[:, :2] - plate_pos[:, :2], dim=1)
    knife_dist_xy = torch.norm(knife_pos[:, :2] - plate_pos[:, :2], dim=1)

    done = torch.logical_and(done, fork_dist_xy <= max_dist_xy)
    done = torch.logical_and(done, knife_dist_xy <= max_dist_xy)

    fork_on_left = fork_pos[:, 0] > plate_pos[:, 0]
    knife_on_right = knife_pos[:, 0] < plate_pos[:, 0]

    done = torch.logical_and(done, fork_on_left)
    done = torch.logical_and(done, knife_on_right)

    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration for the cutlery arrangement task."""

    success = DoneTerm(
        func=cutlery_arranged,
        params={
            "plate_cfg": SceneEntityCfg("plate"),
            "fork_cfg": SceneEntityCfg("fork"),
            "knife_cfg": SceneEntityCfg("knife"),
            "max_dist_xy": 0.15,
        },
    )


@configclass
class CutleryArrangementEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the cutlery arrangement task environment."""

    scene: CutleryArrangementSceneCfg = CutleryArrangementSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "place the fork on the left and knife on the right of the plate."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (7.0, 2.4, 0.6)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)


        # Per-task front camera: pos (7.05,4.5,1.6), aimed at table workspace
        # (7,3.5,~1.05). rot computed as opengl look-at quaternion (w,x,y,z).
        self.scene.front.offset.pos = (7.05, 4.77904, 1.16108)
        self.scene.front.offset.rot = (0.00747, 0.00888, 0.64418, 0.76479)
        self.scene.front.offset.convention = "opengl"
        self.scene.front.spawn.focal_length = 55


        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }

        parse_usd_and_create_subassets(DINING_ROOM_USD_PATH, self)

        self.object_pose_cfg = ObjectPoseConfig(
            tag_to_object=TAG_TO_OBJECT,
            anchor_tag_id=ANCHOR_TAG_ID,
            anchor_world_pose=ANCHOR_WORLD_POSE,
            object_z=OBJECT_Z,
            object_roll=OBJECT_ROLL,
            object_pitch=OBJECT_PITCH,
            per_object_yaw_offset=PER_OBJECT_YAW_OFFSET,
            use_fixed_yaw=True,
            ignored_object_names=IGNORED_OBJECT_NAMES,
        )
