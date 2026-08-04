import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.assets.scenes.living_room import LIVING_ROOM_CFG, LIVING_ROOM_USD_PATH
from simulator.utils.domain_randomization import randomize_light_conditions

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

LIVING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "living_room" / "objects"

TAG_TO_OBJECT: dict[int, str] = {1: "green_block", 2: "blue_block", 3: "red_block"}
ANCHOR_TAG_ID: int = 0
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.35, 0.0, 0.0)
OBJECT_Z: float = 0.05
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0
# Per-USD yaw correction (rad) so the spawned object matches its visual heading
# under the gripper's coordinate convention. Tag-to-mesh alignment differs per
# asset; tune until ``data.root_quat_w`` matches what you see in the viewport.
PER_OBJECT_YAW_OFFSET: dict[str, float] = {
    "green_block": math.pi / 2.0,
    "blue_block": math.pi / 2.0,
    "red_block": math.pi / 2.0,
}


@configclass
class ToyBlocksCollectionSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the toy blocks collection task."""

    scene: AssetBaseCfg = LIVING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    green_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/green_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Bridge" / "Bridge.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.32, -0.35, OBJECT_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    blue_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Cylinder" / "Cylinder.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.35, OBJECT_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    red_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Triangle" / "Triangle.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.58, -0.35, OBJECT_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    storage_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/storage_box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.65, -0.55, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Storage_Box" / "storage_box.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )


def toys_in_box(
    env,
    green_block_cfg: SceneEntityCfg,
    blue_block_cfg: SceneEntityCfg,
    red_block_cfg: SceneEntityCfg,
    storage_box_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Termination: all blocks within (x,y,z)_range of storage_box."""
    green_block: RigidObject = env.scene[green_block_cfg.name]
    blue_block: RigidObject = env.scene[blue_block_cfg.name]
    red_block: RigidObject = env.scene[red_block_cfg.name]
    storage_box: RigidObject = env.scene[storage_box_cfg.name]

    green_block_pos = green_block.data.root_pos_w - env.scene.env_origins
    blue_block_pos = blue_block.data.root_pos_w - env.scene.env_origins
    red_block_pos = red_block.data.root_pos_w - env.scene.env_origins
    storage_box_pos = storage_box.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for toy_pos in (green_block_pos, blue_block_pos, red_block_pos):
        done = torch.logical_and(done, toy_pos[:, 0] < storage_box_pos[:, 0] + x_range[1])
        done = torch.logical_and(done, toy_pos[:, 0] > storage_box_pos[:, 0] + x_range[0])
        done = torch.logical_and(done, toy_pos[:, 1] < storage_box_pos[:, 1] + y_range[1])
        done = torch.logical_and(done, toy_pos[:, 1] > storage_box_pos[:, 1] + y_range[0])
        done = torch.logical_and(done, toy_pos[:, 2] < storage_box_pos[:, 2] + z_range[1])
        done = torch.logical_and(done, toy_pos[:, 2] > storage_box_pos[:, 2] + z_range[0])

    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration for the toy blocks collection task."""

    success = DoneTerm(
        func=toys_in_box,
        params={
            "green_block_cfg": SceneEntityCfg("green_block"),
            "blue_block_cfg": SceneEntityCfg("blue_block"),
            "red_block_cfg": SceneEntityCfg("red_block"),
            "storage_box_cfg": SceneEntityCfg("storage_box"),
            "x_range": (-0.12, 0.12),
            "y_range": (-0.12, 0.12),
            "z_range": (-0.08, 0.08),
        },
    )


@configclass
class ToyBlocksCollectionEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the toy blocks collection task environment."""

    scene: ToyBlocksCollectionSceneCfg = ToyBlocksCollectionSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the toys and place them into the storage box."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
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

        parse_usd_and_create_subassets(LIVING_ROOM_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "green_block",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_object_uniform(
                    "blue_block",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_object_uniform(
                    "red_block",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_light_conditions("light", textures=[], intensity_range=(1100, 1300)),
            ],
        )
