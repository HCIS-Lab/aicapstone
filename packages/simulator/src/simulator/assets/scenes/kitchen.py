from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from simulator import ASSETS_ROOT

"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

KITCHEN_USD_PATH = str(SCENES_ROOT / "kitchen" / "nycu_light_collider_wooden.usd")

KITCHEN_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_USD_PATH,
    )
)
