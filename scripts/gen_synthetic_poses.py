"""Generate synthetic object_poses.json entries for cutlery arrangement datagen.

Samples random fork/knife positions within the robot's reachable workspace
and outputs them in the UMI object_poses.json format so generate.py can
consume them directly without any changes.

Anchor world pose: (0.40, 0.10, 0.0)
  → tvec = (x_world - 0.40, y_world - 0.10, 0)

Robot base: (0.35, -0.74), plate fixed at (0.50, -0.40).
"""

import json
import math
import random
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
SEED = 42
N_SYNTHETIC = 60          # how many synthetic episodes to generate
ANCHOR_X, ANCHOR_Y = 0.40, 0.10
PLATE_POS = (0.50, -0.40)
ROBOT_BASE = (0.35, -0.74)
R_MAX, R_MIN = 0.845, 0.20   # Franka horizontal reach at grasp height

MIN_FORK_KNIFE_DIST = 0.08   # keep them ≥8 cm apart
MIN_PLATE_DIST = 0.12        # keep each object ≥12 cm from plate
# ────────────────────────────────────────────────────────────────────────────

def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _in_workspace(x, y):
    d = math.sqrt((x - ROBOT_BASE[0])**2 + (y - ROBOT_BASE[1])**2)
    return R_MIN <= d <= R_MAX


def generate_synthetic(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    episodes = []
    attempts = 0
    while len(episodes) < n:
        attempts += 1
        if attempts > 100_000:
            print(f"[WARN] Only generated {len(episodes)}/{n} episodes after {attempts} attempts")
            break

        fx = rng.uniform(ROBOT_BASE[0] - R_MAX, ROBOT_BASE[0] + R_MAX)
        fy = rng.uniform(ROBOT_BASE[1] - R_MAX, ROBOT_BASE[1] + R_MAX)
        kx = rng.uniform(ROBOT_BASE[0] - R_MAX, ROBOT_BASE[0] + R_MAX)
        ky = rng.uniform(ROBOT_BASE[1] - R_MAX, ROBOT_BASE[1] + R_MAX)

        if not _in_workspace(fx, fy) or not _in_workspace(kx, ky):
            continue
        if _dist((fx, fy), (kx, ky)) < MIN_FORK_KNIFE_DIST:
            continue
        if _dist((fx, fy), PLATE_POS) < MIN_PLATE_DIST:
            continue
        if _dist((kx, ky), PLATE_POS) < MIN_PLATE_DIST:
            continue

        episodes.append({
            "video_name": f"synthetic_{len(episodes):04d}",
            "episode_range": [0, 100],
            "objects": [
                {
                    "object_name": "fork",
                    "rvec": [0.0, 0.0, 0.0],
                    "tvec": [fx - ANCHOR_X, fy - ANCHOR_Y, 0.0],
                },
                {
                    "object_name": "knife",
                    "rvec": [0.0, 0.0, 0.0],
                    "tvec": [kx - ANCHOR_X, ky - ANCHOR_Y, 0.0],
                },
                {
                    "object_name": "plate",
                    "rvec": [0.0, 0.0, 0.0],
                    "tvec": [PLATE_POS[0] - ANCHOR_X, PLATE_POS[1] - ANCHOR_Y, 0.0],
                },
            ],
            "status": "full",
        })

    print(f"Generated {len(episodes)} synthetic episodes ({attempts} attempts)")
    return episodes


def main():
    data_dir = Path(__file__).parent.parent / "data" / "AI-final-49"
    real_path = data_dir / "object_poses_filtered.json"
    out_path = data_dir / "object_poses_combined.json"

    with open(real_path) as f:
        real_episodes = json.load(f)
    print(f"Loaded {len(real_episodes)} real episodes from {real_path}")

    synthetic = generate_synthetic(N_SYNTHETIC, SEED)

    combined = real_episodes + synthetic
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved {len(combined)} total episodes → {out_path}")
    print(f"  real: {len(real_episodes)}  synthetic: {len(synthetic)}")
    print(f"  × factor 10 = {len(combined) * 10} datagen episodes")


if __name__ == "__main__":
    main()
