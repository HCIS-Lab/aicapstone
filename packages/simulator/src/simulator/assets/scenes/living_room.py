from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from simulator import ASSETS_ROOT

"""Configuration for the Living Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

LIVING_ROOM_USD_PATH = str(SCENES_ROOT / "living_room" / "nycu_light_collider_wooden.usd")

LIVING_ROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIVING_ROOM_USD_PATH,
    )
)
