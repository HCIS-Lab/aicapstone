import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from simulator import ASSETS_ROOT
from simulator.utils.object_poses_loader import ObjectPoseConfig
from simulator.assets.scenes.kitchen import KITCHEN_CFG, KITCHEN_USD_PATH
from simulator.utils.domain_randomization import randomize_light_conditions

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

KITCHEN_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "kitchen" / "objects"

TAG_TO_OBJECT: dict[int, str] = {1: "blue_cup", 2: "pink_cup"}
ANCHOR_TAG_ID: int = 0
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.5, -0.2, 0.0)
OBJECT_Z: float = 1.0
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0


@configclass
class CupStackingSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the custom task."""

    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    blue_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "BlueCup" / "BlueCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, -0.4, 0.12), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    pink_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/pink_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "PinkCup" / "PinkCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.4, 0.12), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def blue_cup_on_top_pink_cup(
    env,
    blue_cup_cfg: SceneEntityCfg,
    pink_cup_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
) -> torch.Tensor:
    """Termination condition for the cup stacking task."""
    blue_cup: RigidObject = env.scene[blue_cup_cfg.name]
    pink_cup: RigidObject = env.scene[pink_cup_cfg.name]

    blue_cup_pos = blue_cup.data.root_pos_w - env.scene.env_origins
    pink_cup_pos = pink_cup.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    done = torch.logical_and(done, blue_cup_pos[:, 0] < pink_cup_pos[:, 0] + x_range[1])
    done = torch.logical_and(done, blue_cup_pos[:, 0] > pink_cup_pos[:, 0] + x_range[0])
    done = torch.logical_and(done, blue_cup_pos[:, 1] < pink_cup_pos[:, 1] + y_range[1])
    done = torch.logical_and(done, blue_cup_pos[:, 1] > pink_cup_pos[:, 1] + y_range[0])
    done = torch.logical_and(done, blue_cup_pos[:, 2] > pink_cup_pos[:, 2] + height_threshold)
    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration for the custom task."""

    success = DoneTerm(
        func=blue_cup_on_top_pink_cup,
        params={
            "blue_cup_cfg": SceneEntityCfg("blue_cup"),
            "pink_cup_cfg": SceneEntityCfg("pink_cup"),
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "height_threshold": 0.10,
        },
    )


@configclass
class CupStackingEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the custom task environment."""

    scene: CupStackingSceneCfg = CupStackingSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the blue cup and place it on the pink cup."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (9.0, 6.0, 0.9)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        # Per-task front camera: pos (7.2,6.2,1.45), aimed at (9,6.45,1.05)
        # over the cup spawn. rot is that look-at as an opengl quaternion
        # (w,x,y,z).
        # The other two tasks put the camera on the far side of the objects
        # looking back at the arm; the kitchen has no room for that -- the arm
        # faces +y into the backsplash (wall at y~6.8) and wall cabinets close
        # in above z~1.3 from x~8.0 on -- so this is a raised three-quarter
        # view from the arm's left, clearing the cabinets and framing the whole
        # +-0.15 m cup jitter box with the arm entering from frame right.
        self.scene.front.offset.pos = (7.80362, 6.78599, 1.1197),
        self.scene.front.offset.rot = (0.37025, 0.37025, -0.60242, -0.60242)
        self.scene.front.offset.convention = "opengl"
        self.scene.front.spawn.focal_length = 18.14756

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

        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "blue_cup",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_object_uniform(
                    "pink_cup",
                    pose_range={
                        "x": (0.02, 0.05),
                        "y": (0.02, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_light_conditions("light", textures=[], intensity_range=(1100, 1300)),
            ],
        )
