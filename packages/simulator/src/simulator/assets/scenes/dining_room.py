from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from simulator import ASSETS_ROOT

"""Configuration for the Dining Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

DINING_ROOM_USD_PATH = str(SCENES_ROOT / "dining_room" / "nycu_light_collider_wooden.usd")

DINING_ROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DINING_ROOM_USD_PATH,
    )
)
