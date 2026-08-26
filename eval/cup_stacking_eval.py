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
from simulator.assets.scenes.kitchen import KITCHEN_CFG, KITCHEN_USD_PATH

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

KITCHEN_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "kitchen" / "objects"

TAG_TO_OBJECT: dict[int, str] = {1: "blue_cup", 2: "pink_cup"}
ANCHOR_TAG_ID: int = 0
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.0, 0.0, 0.0)
OBJECT_Z: float = 1.0
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0


configure_seed(42)

@configclass
class CupStackingEvalSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    blue_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "BlueCup" / "BlueCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.0, 6.45, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    pink_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/pink_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "PinkCup" / "PinkCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.2, 6.5, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


# Success when blue cup centered over pink cup within xy tolerance and above by height_threshold.
def blue_cup_on_top_pink_cup(
    env,
    blue_cup_cfg: SceneEntityCfg,
    pink_cup_cfg: SceneEntityCfg,
    x_tolerance: float,
    y_tolerance: float,
    height_threshold: float,
) -> torch.Tensor:
    blue_cup: RigidObject = env.scene[blue_cup_cfg.name]
    pink_cup: RigidObject = env.scene[pink_cup_cfg.name]

    blue_cup_pos = blue_cup.data.root_pos_w - env.scene.env_origins
    pink_cup_pos = pink_cup.data.root_pos_w - env.scene.env_origins

    done = torch.abs(blue_cup_pos[:, 0] - pink_cup_pos[:, 0]) < x_tolerance
    done = torch.logical_and(done, torch.abs(blue_cup_pos[:, 1] - pink_cup_pos[:, 1]) < y_tolerance)
    done = torch.logical_and(done, blue_cup_pos[:, 2] > pink_cup_pos[:, 2] + height_threshold)
    return done


@configclass
class EvalTerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=blue_cup_on_top_pink_cup,
        params={
            "blue_cup_cfg": SceneEntityCfg("blue_cup"),
            "pink_cup_cfg": SceneEntityCfg("pink_cup"),
            "x_tolerance": 0.05,
            "y_tolerance": 0.05,
            "height_threshold": 0.10,
        },
    )


@configclass
class CupStackingEvalEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: CupStackingEvalSceneCfg = CupStackingEvalSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: EvalTerminationsCfg = EvalTerminationsCfg()
    task_description: str = "pick up the blue cup and place it on the pink cup."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (9.0, 6.0, 0.9)
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
                        "x": (-0.05, 0.05), 
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
            ],
        )

TASK_ID = "Private-CupStacking-Eval-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:CupStackingEvalEnvCfg"},
)
