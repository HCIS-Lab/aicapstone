"""Pose augmentation utilities for synthetic data generation.

These helpers operate on the world-frame episode poses returned by
``load_episode_poses``. The goal is to expand a small number of UMI-derived
scene setups into more simulator rollouts without depending on the raw UMI
trajectory, which is not used by the current training pipeline.
"""

from __future__ import annotations

import math
import random

from simulator.utils.object_poses_loader import WorldPose

EpisodeWorldPoses = dict[str, WorldPose]

_CUTLERY_EVAL_BASE_XY: dict[str, tuple[float, float]] = {
    "knife": (0.50, -0.10),
    "fork": (0.55, -0.10),
}


class PoseAugmentationError(ValueError):
    """Raised when pose augmentation inputs are malformed."""


def augment_episode_world_poses(
    episodes: list[EpisodeWorldPoses],
    *,
    factor: int,
    seed: int,
    global_xy_jitter: float = 0.0,
    local_xy_jitter: float = 0.0,
    yaw_jitter_deg: float = 0.0,
    min_object_distance: float = 0.0,
    mix_objects: bool = False,
    max_attempts: int = 64,
) -> list[EpisodeWorldPoses]:
    """Expand replay episodes with conservative world-frame perturbations.

    Args:
        episodes: Base replay episodes from ``load_episode_poses``.
        factor: Dataset multiplier. ``1`` returns a copy of the original list.
        seed: RNG seed for deterministic augmentation.
        global_xy_jitter: Uniform translation range applied to the full scene.
        local_xy_jitter: Additional per-object uniform translation range.
        yaw_jitter_deg: Uniform world-yaw jitter applied to every object.
        min_object_distance: Minimum allowed XY distance between any two objects.
        mix_objects: If True, build new episodes by sampling each object from the
            full per-object pose bank instead of preserving original pairings.
        max_attempts: Rejection-sampling attempts per synthetic episode.
    """
    if factor < 1:
        raise PoseAugmentationError(f"factor must be >= 1, got {factor}")
    if max_attempts < 1:
        raise PoseAugmentationError(f"max_attempts must be >= 1, got {max_attempts}")
    if any(value < 0.0 for value in (global_xy_jitter, local_xy_jitter, yaw_jitter_deg, min_object_distance)):
        raise PoseAugmentationError("jitter and min_object_distance values must be non-negative")
    if not episodes:
        return []

    object_names = tuple(sorted(episodes[0].keys()))
    if not object_names:
        raise PoseAugmentationError("episodes must contain at least one object")
    for ep_idx, episode in enumerate(episodes[1:], start=1):
        names = tuple(sorted(episode.keys()))
        if names != object_names:
            raise PoseAugmentationError(
                f"episode {ep_idx} object set {names} does not match episode 0 object set {object_names}"
            )

    rng = random.Random(seed)
    pose_bank = {
        name: [_clone_world_pose(episode[name]) for episode in episodes]
        for name in object_names
    }
    out = [_clone_episode(episode) for episode in episodes]

    if factor == 1:
        return out

    yaw_jitter_rad = math.radians(yaw_jitter_deg)
    for episode in episodes:
        for _ in range(factor - 1):
            out.append(
                _sample_augmented_episode(
                    base_episode=episode,
                    pose_bank=pose_bank,
                    object_names=object_names,
                    rng=rng,
                    global_xy_jitter=global_xy_jitter,
                    local_xy_jitter=local_xy_jitter,
                    yaw_jitter_rad=yaw_jitter_rad,
                    min_object_distance=min_object_distance,
                    mix_objects=mix_objects,
                    max_attempts=max_attempts,
                )
            )
    return out


def inject_cutlery_eval_pose_distribution(
    episodes: list[EpisodeWorldPoses],
    *,
    seed: int,
    eval_like_fraction: float,
    eval_xy_jitter: float,
    replaceable_start_index: int = 0,
    min_object_distance: float = 0.0,
    max_attempts: int = 64,
) -> list[EpisodeWorldPoses]:
    """Blend eval-like cutlery starts into an existing replay set.

    A subset of the replaceable episodes is rewritten so fork/knife spawn near
    the public eval distribution, while the remaining episodes keep the broader
    UMI-derived or augmented scene coverage. This helps datagen see some scenes
    close to eval without collapsing the whole dataset onto that narrow range.
    """
    if not episodes:
        return []
    if not 0.0 <= eval_like_fraction <= 1.0:
        raise PoseAugmentationError(
            f"eval_like_fraction must be in [0, 1], got {eval_like_fraction}"
        )
    if eval_xy_jitter < 0.0:
        raise PoseAugmentationError(f"eval_xy_jitter must be non-negative, got {eval_xy_jitter}")
    if max_attempts < 1:
        raise PoseAugmentationError(f"max_attempts must be >= 1, got {max_attempts}")
    if not 0 <= replaceable_start_index <= len(episodes):
        raise PoseAugmentationError(
            "replaceable_start_index must be within the episode list bounds"
        )

    object_names = set(episodes[0].keys())
    if not set(_CUTLERY_EVAL_BASE_XY).issubset(object_names):
        return [_clone_episode(episode) for episode in episodes]

    out = [_clone_episode(episode) for episode in episodes]
    replaceable_indices = list(range(replaceable_start_index, len(out)))
    if not replaceable_indices or eval_like_fraction == 0.0:
        return out

    rng = random.Random(seed)
    target_count = max(1, round(len(replaceable_indices) * eval_like_fraction))
    for episode_index in rng.sample(replaceable_indices, k=min(target_count, len(replaceable_indices))):
        source_episode = rng.choice(out[:replaceable_start_index] or out)
        out[episode_index] = _sample_cutlery_eval_episode(
            base_episode=source_episode,
            rng=rng,
            eval_xy_jitter=eval_xy_jitter,
            min_object_distance=min_object_distance,
            max_attempts=max_attempts,
        )
    return out


def _sample_augmented_episode(
    *,
    base_episode: EpisodeWorldPoses,
    pose_bank: dict[str, list[WorldPose]],
    object_names: tuple[str, ...],
    rng: random.Random,
    global_xy_jitter: float,
    local_xy_jitter: float,
    yaw_jitter_rad: float,
    min_object_distance: float,
    mix_objects: bool,
    max_attempts: int,
) -> EpisodeWorldPoses:
    for _ in range(max_attempts):
        dx_global = rng.uniform(-global_xy_jitter, global_xy_jitter)
        dy_global = rng.uniform(-global_xy_jitter, global_xy_jitter)
        yaw_delta = rng.uniform(-yaw_jitter_rad, yaw_jitter_rad) if yaw_jitter_rad > 0.0 else 0.0
        episode: EpisodeWorldPoses = {}

        for name in object_names:
            source_pose = rng.choice(pose_bank[name]) if mix_objects else base_episode[name]
            pos, quat = source_pose
            dx_local = rng.uniform(-local_xy_jitter, local_xy_jitter)
            dy_local = rng.uniform(-local_xy_jitter, local_xy_jitter)
            episode[name] = (
                (pos[0] + dx_global + dx_local, pos[1] + dy_global + dy_local, pos[2]),
                _rotate_world_yaw(quat, yaw_delta),
            )

        if _has_valid_separation(episode, min_object_distance):
            return episode

    return _clone_episode(base_episode)


def _sample_cutlery_eval_episode(
    *,
    base_episode: EpisodeWorldPoses,
    rng: random.Random,
    eval_xy_jitter: float,
    min_object_distance: float,
    max_attempts: int,
) -> EpisodeWorldPoses:
    for _ in range(max_attempts):
        episode = _clone_episode(base_episode)
        for name, (base_x, base_y) in _CUTLERY_EVAL_BASE_XY.items():
            pos, quat = episode[name]
            episode[name] = (
                (
                    base_x + rng.uniform(-eval_xy_jitter, eval_xy_jitter),
                    base_y + rng.uniform(-eval_xy_jitter, eval_xy_jitter),
                    pos[2],
                ),
                quat,
            )
        if _has_valid_separation(episode, min_object_distance):
            return episode
    return _clone_episode(base_episode)


def _has_valid_separation(episode: EpisodeWorldPoses, min_object_distance: float) -> bool:
    if min_object_distance <= 0.0 or len(episode) < 2:
        return True
    poses = list(episode.values())
    min_dist_sq = min_object_distance * min_object_distance
    for idx, (pos_a, _) in enumerate(poses):
        for pos_b, _ in poses[idx + 1 :]:
            dx = pos_a[0] - pos_b[0]
            dy = pos_a[1] - pos_b[1]
            if dx * dx + dy * dy < min_dist_sq:
                return False
    return True


def _rotate_world_yaw(quat_wxyz: tuple[float, float, float, float], yaw_delta: float) -> tuple[float, float, float, float]:
    if abs(yaw_delta) < 1e-12:
        return tuple(float(v) for v in quat_wxyz)
    half = yaw_delta * 0.5
    delta = (math.cos(half), 0.0, 0.0, math.sin(half))
    return _quat_mul_wxyz(delta, quat_wxyz)


def _quat_mul_wxyz(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _clone_episode(episode: EpisodeWorldPoses) -> EpisodeWorldPoses:
    return {name: _clone_world_pose(pose) for name, pose in episode.items()}


def _clone_world_pose(world_pose: WorldPose) -> WorldPose:
    pos, quat = world_pose
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )
