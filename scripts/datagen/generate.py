"""
References: https://github.com/LightwheelAI/leisaac
Unified data generation script using state machines.

Selects the appropriate state machine based on --task and runs the recording loop.
Episode count is driven by --object_poses: each ``status == "full"`` entry in the
file yields one replayed episode. Object placements are written via
``RigidObject.write_root_pose_to_sim`` after each ``env.reset()``.

Usage:
    python scripts/datagen/generate.py \
        --task HCIS-CupStacking-SingleArm-v0 \
        --num_envs 1 --device cuda --enable_cameras \
        --record --dataset_file ./datasets/cup_stacking.hdf5 \
        --object_poses datasets/0210_kitchen/demos/mapping/object_poses.json
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument(
    "--object_poses",
    type=str,
    required=True,
    help="Path to the per-episode object_poses.json (UMI schema). Episode count = number of status=='full' entries.",
)
parser.add_argument(
    "--augment_pose_factor",
    type=int,
    default=1,
    help="Multiply the replay set by this factor using pose jitter. Example: 16 source episodes with factor 10 -> 160 replay episodes.",
)
parser.add_argument(
    "--augment_global_xy_jitter",
    type=float,
    default=0.01,
    help="Scene-level XY translation jitter in meters applied during pose augmentation.",
)
parser.add_argument(
    "--augment_local_xy_jitter",
    type=float,
    default=0.01,
    help="Per-object XY translation jitter in meters applied during pose augmentation.",
)
parser.add_argument(
    "--augment_yaw_jitter_deg",
    type=float,
    default=0.0,
    help="World-yaw jitter in degrees applied during pose augmentation.",
)
parser.add_argument(
    "--augment_min_object_distance",
    type=float,
    default=0.05,
    help="Minimum XY spacing in meters enforced between augmented objects.",
)
parser.add_argument(
    "--augment_mix_objects",
    action="store_true",
    help="Mix object poses across different source episodes before jittering. Increases diversity but is slightly riskier than jitter-only augmentation.",
)
parser.add_argument(
    "--cutlery_eval_pose_fraction",
    type=float,
    default=0.35,
    help="For the cutlery task only: rewrite this fraction of augmented episodes so fork/knife start near the eval initial-pose distribution.",
)
parser.add_argument(
    "--cutlery_eval_pose_jitter",
    type=float,
    default=0.05,
    help="For the cutlery task only: XY jitter in meters around the eval base poses when injecting eval-like initial states.",
)
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import simulator.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.datagen.state_machine import PickOrangeStateMachine
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

from simulator.datagen.state_machine.cup_stacking import CupStackingStateMachine
from simulator.datagen.state_machine.cutlery_arrangement import CutleryArrangementStateMachine
from simulator.datagen.state_machine.toy_blocks_collection import ToyBlocksCollectionStateMachine
from simulator.utils.object_pose_augmentation import (
    augment_episode_world_poses,
    inject_cutlery_eval_pose_distribution,
)
from simulator.utils.object_poses_loader import load_episode_poses

# Maps gym task id → (StateMachineClass, device_type)
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
    "HCIS-CupStacking-SingleArm-v0": (CupStackingStateMachine, "keyboard"),
    "HCIS-ToyBlocksCollection-SingleArm-v0": (ToyBlocksCollectionStateMachine, "keyboard"),
    "HCIS-CutleryArrangement-SingleArm-v0": (CutleryArrangementStateMachine, "keyboard"),
}


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name):
    """Configure termination and recorder settings on env_cfg."""
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


def _replace_recorder_manager(env, env_cfg, args_cli):
    """Replace the default recorder manager with streaming or lerobot recorder."""
    del env.recorder_manager
    if args_cli.use_lerobot_recorder:
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from leisaac.enhance.managers.lerobot_recorder_manager import (
            LeRobotRecorderManager,
        )

        dataset_cfg = LeRobotDatasetCfg(
            repo_id=args_cli.lerobot_dataset_repo_id,
            fps=args_cli.lerobot_dataset_fps,
        )
        env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
    else:
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


def _apply_episode_poses(env, poses):
    """Write per-object root poses for the current episode into the sim."""
    import math as _math

    device = env.device
    for name, (pos, quat) in poses.items():
        obj = env.scene[name]
        pose_tensor = torch.tensor(
            [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
            device=device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        obj.write_root_pose_to_sim(pose_tensor)
        w, x, y, z = quat
        yaw_deg = _math.degrees(_math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        print(
            f"  [pose] {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
            f"yaw={yaw_deg:+6.1f}°"
        )


# z below which a task object is considered to have fallen off the table.
# Objects sit at object_z ≈ 0.05; anything under the table surface trips this.
_FALL_THRESHOLD_Z: float = 0.0


def _any_object_fell(env, object_names, z_threshold: float) -> bool:
    """Return True if any named scene object has root_pos_w.z below z_threshold."""
    for name in object_names:
        try:
            obj = env.scene[name]
        except KeyError:
            continue
        if torch.any(obj.data.root_pos_w[:, 2] < z_threshold).item():
            return True
    return False


def _on_episode_done(
    env,
    sm,
    args_cli,
    episodes,
    next_episode_idx,
    resume_recorded_demo_count,
    current_recorded_demo_count,
    start_record_state,
):
    """Handle end-of-episode logic.

    Returns (next_episode_idx, current_recorded_demo_count, start_record_state, should_break).
    """
    total_episodes = len(episodes)

    try:
        success = sm.check_success(env)
    except Exception as e:
        print("Success check failed:", e)
        success = False

    print("Episode success!" if success else "Episode failed!")

    if start_record_state:
        if args_cli.record:
            print("Stop Recording!!!")
        start_record_state = False

    if args_cli.record and success:
        auto_terminate(env, True)
        current_recorded_demo_count += 1
    else:
        auto_terminate(env, False)

    if (
        args_cli.record
        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        > current_recorded_demo_count
    ):
        current_recorded_demo_count = (
            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        )
        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

    if next_episode_idx >= total_episodes:
        print(f"Replayed all {total_episodes} episodes. Exiting the app.")
        return next_episode_idx, current_recorded_demo_count, start_record_state, True, success

    env.reset()
    sm.reset()
    auto_terminate(env, False)
    _apply_episode_poses(env, episodes[next_episode_idx])
    next_episode_idx += 1

    return next_episode_idx, current_recorded_demo_count, start_record_state, False, success


def main():
    """Run a state machine in a LeIsaac manipulation environment."""
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\nAvailable tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device = TASK_REGISTRY[task_name]
    run_seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = run_seed

    if getattr(env_cfg, "object_pose_cfg", None) is None:
        raise ValueError(
            f"Task '{task_name}' env_cfg has no 'object_pose_cfg' attribute; "
            "cannot resolve anchor frame for --object_poses."
        )
    episodes = load_episode_poses(args_cli.object_poses, env_cfg.object_pose_cfg)
    if not episodes:
        raise ValueError(
            f"No 'status==full' episodes in {args_cli.object_poses}; nothing to replay."
        )
    base_episode_count = len(episodes)
    print(f"Loaded {base_episode_count} replay episodes from {args_cli.object_poses}")
    if args_cli.augment_pose_factor > 1:
        episodes = augment_episode_world_poses(
            episodes,
            factor=args_cli.augment_pose_factor,
            seed=run_seed,
            global_xy_jitter=args_cli.augment_global_xy_jitter,
            local_xy_jitter=args_cli.augment_local_xy_jitter,
            yaw_jitter_deg=args_cli.augment_yaw_jitter_deg,
            min_object_distance=args_cli.augment_min_object_distance,
            mix_objects=args_cli.augment_mix_objects,
        )
        print(
            "Augmented replay episodes: "
            f"{base_episode_count} -> {len(episodes)} "
            f"(factor={args_cli.augment_pose_factor}, mix_objects={args_cli.augment_mix_objects}, seed={run_seed})"
        )
    if task_name == "HCIS-CutleryArrangement-SingleArm-v0":
        episodes = inject_cutlery_eval_pose_distribution(
            episodes,
            seed=run_seed,
            eval_like_fraction=args_cli.cutlery_eval_pose_fraction,
            eval_xy_jitter=args_cli.cutlery_eval_pose_jitter,
            replaceable_start_index=base_episode_count,
            min_object_distance=args_cli.augment_min_object_distance,
        )
        print(
            "Blended cutlery eval-like starts into replay set: "
            f"fraction={args_cli.cutlery_eval_pose_fraction}, jitter={args_cli.cutlery_eval_pose_jitter}"
        )

    is_direct_env = "Direct" in task_name
    _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # disable gravity for every robot link prim
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if args_cli.record:
        _replace_recorder_manager(env, env_cfg, args_cli)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    # one-time state machine setup (e.g. FK calibration)
    sm = SMClass()
    sm.setup(env)
    env.reset()
    sm.reset()

    fall_check_object_names = tuple(getattr(sm, "task_object_names", ()))

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    next_episode_idx = min(resume_recorded_demo_count, len(episodes))
    if next_episode_idx >= len(episodes):
        print(f"Resume count {next_episode_idx} ≥ total episodes {len(episodes)}; nothing to do.")
        env.close()
        simulation_app.close()
        return
    _apply_episode_poses(env, episodes[next_episode_idx])
    next_episode_idx += 1

    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    cnt = 1
    success_ID = []
    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, device)

                if sm.is_episode_done:
                    (
                        next_episode_idx,
                        current_recorded_demo_count,
                        start_record_state,
                        should_break,
                        success,
                    ) = _on_episode_done(
                        env,
                        sm,
                        args_cli,
                        episodes,
                        next_episode_idx,
                        resume_recorded_demo_count,
                        current_recorded_demo_count,
                        start_record_state,
                    )
                    if success:
                        print(f"\033[92m[Data Usage]{cnt}/{len(episodes)} success.\033[0m")
                        success_ID.append(cnt)
                        cnt += 1
                    else:
                        print(f"\033[91m[Data Usage]{cnt}/{len(episodes)} fail.\033[0m")
                    if should_break:
                        break
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    env.step(actions)
                    sm.advance()

                    if fall_check_object_names and _any_object_fell(
                        env, fall_check_object_names, _FALL_THRESHOLD_Z
                    ):
                        print(
                            "[INFO] Task object fell off the table; aborting this "
                            "episode and skipping to next."
                        )
                        sm._episode_done = True

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()
    
    print(success_ID)


if __name__ == "__main__":
    main()
