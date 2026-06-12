import math

import pytest

from simulator.utils.object_pose_augmentation import (
    PoseAugmentationError,
    augment_episode_world_poses,
)


def _episode(blue_xy, pink_xy):
    return {
        "blue_cup": ((blue_xy[0], blue_xy[1], 0.12), (1.0, 0.0, 0.0, 0.0)),
        "pink_cup": ((pink_xy[0], pink_xy[1], 0.12), (1.0, 0.0, 0.0, 0.0)),
    }


def _yaw_from_quat_wxyz(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def test_factor_one_returns_copy_with_same_content():
    episodes = [_episode((0.4, -0.2), (0.55, -0.2))]
    out = augment_episode_world_poses(episodes, factor=1, seed=7)

    assert out == episodes
    assert out is not episodes
    assert out[0] is not episodes[0]


def test_factor_multiplies_episode_count():
    episodes = [
        _episode((0.40, -0.20), (0.56, -0.20)),
        _episode((0.42, -0.24), (0.58, -0.22)),
    ]

    out = augment_episode_world_poses(
        episodes,
        factor=3,
        seed=11,
        global_xy_jitter=0.01,
        local_xy_jitter=0.005,
    )

    assert len(out) == 6
    assert out[:2] == episodes


def test_yaw_jitter_rotates_quaternion():
    episodes = [_episode((0.4, -0.2), (0.55, -0.2))]

    out = augment_episode_world_poses(
        episodes,
        factor=2,
        seed=5,
        yaw_jitter_deg=15.0,
    )

    assert len(out) == 2
    augmented_yaw = _yaw_from_quat_wxyz(out[1]["blue_cup"][1])
    assert abs(augmented_yaw) > 1e-6


def test_mix_objects_recombines_pose_bank():
    episodes = [
        _episode((0.40, -0.20), (0.70, -0.20)),
        _episode((0.48, -0.32), (0.86, -0.32)),
    ]

    out = augment_episode_world_poses(
        episodes,
        factor=2,
        seed=3,
        mix_objects=True,
        global_xy_jitter=0.0,
        local_xy_jitter=0.0,
        min_object_distance=0.05,
    )

    augmented = out[2]
    blue_x = augmented["blue_cup"][0][0]
    pink_x = augmented["pink_cup"][0][0]
    assert blue_x in {0.40, 0.48}
    assert pink_x in {0.70, 0.86}


def test_min_object_distance_rejects_collision_and_falls_back_to_base():
    episodes = [_episode((0.40, -0.20), (0.44, -0.20))]

    out = augment_episode_world_poses(
        episodes,
        factor=2,
        seed=19,
        mix_objects=True,
        min_object_distance=0.10,
        max_attempts=2,
    )

    assert out[1] == episodes[0]


def test_inconsistent_object_sets_raise():
    episodes = [
        _episode((0.40, -0.20), (0.56, -0.20)),
        {"blue_cup": ((0.42, -0.24, 0.12), (1.0, 0.0, 0.0, 0.0))},
    ]

    with pytest.raises(PoseAugmentationError, match="object set"):
        augment_episode_world_poses(episodes, factor=2, seed=1)
