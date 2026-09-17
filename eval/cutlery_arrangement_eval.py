import math

import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.utils.seed import configure_seed

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from simulator import ASSETS_ROOT
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
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
# it with ≥ 10 cm of free space on both ±x sides for fork (-x) and knife
# (+x) drop targets (state machine uses `_PLACE_OFFSET = 0.10`).
PLATE_WORLD_POS: tuple[float, float, float] = (7.0, 2.9, 0.74416)


configure_seed(42)

@configclass
class CutleryArrangementSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = DINING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.75, 3.15, 0.75), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    fork: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/fork",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Fork" / "fork.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.05, 3.05, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )


# Success when fork on -x side of plate, knife on +x side, both within max_dist_xy.
def cutlery_arranged(
    env,
    plate_cfg: SceneEntityCfg,
    fork_cfg: SceneEntityCfg,
    knife_cfg: SceneEntityCfg,
    max_dist_xy: float,
) -> torch.Tensor:
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

    fork_on_minus_x = fork_pos[:, 0] < plate_pos[:, 0]
    knife_on_plus_x = knife_pos[:, 0] > plate_pos[:, 0]

    done = torch.logical_and(done, fork_on_minus_x)
    done = torch.logical_and(done, knife_on_plus_x)

    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
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

        self.scene.front.offset.pos = (7.05, 4.77904, 1.16108)
        self.scene.front.offset.rot = (0.00747, 0.00888, 0.64418, 0.76479)
        self.scene.front.offset.convention = "opengl"
        self.scene.front.spawn.focal_length = 55

        parse_usd_and_create_subassets(DINING_ROOM_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "knife",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_object_uniform(
                    "fork",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
            ],
        )


TASK_ID = "Private-CutleruArrangement-Eval-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:CutleryArrangementEnvCfg"},
)
