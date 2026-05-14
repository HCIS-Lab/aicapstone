# Synchronous LeRobot rollout script for LeIsaac.
# Derived partially from upstream LeIsaac
# `scripts/evaluation/policy_inference.py`
# (https://github.com/LightwheelAI/leisaac/blob/main/scripts/evaluation/policy_inference.py
# @ SHA 6b933e80786a69eb27d47503d11725c9c846566e), trimmed to local LeRobot
# inference and extended with a dual-viewport setup, a debug shape printer,
# and an in-process LeRobotSyncPolicy. Entry point lives at the top of
# `scripts/` (NOT under `scripts/evaluation/`) per AUT-81.

"""Run local LeRobot policy inference in the same process as Isaac Sim."""

"""Launch Isaac Sim Simulator first."""
import json as _json
import multiprocessing
import os
from pathlib import Path as _Path


# Fields that newer LeRobot adds at training time but the inference-side
# LeRobot installed in the worker image doesn't accept. They're all
# training-only (LoRA, torch.compile, image-preproc) and safe to strip
# from the checkpoint's config.json before from_pretrained() reads it.
# Extend whenever draccus.utils.DecodingError surfaces a new field.
_LEROBOT_INCOMPAT_CONFIG_FIELDS: tuple[str, ...] = (
    "use_peft",
    "resize_shape",
    "crop_ratio",
    "compile_model",
    "compile_mode",
)


def _patch_lerobot_config(checkpoint_dir: str) -> None:
    """Strip known-incompatible fields from <checkpoint>/config.json.

    Idempotent — running twice is fine. Errors are swallowed; if config.json
    is missing or unreadable the original from_pretrained call will still
    surface a helpful message.
    """
    cfg_path = _Path(checkpoint_dir) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open("r") as f:
            cfg = _json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[rollout] config.json read skipped: {exc}", flush=True)
        return
    stripped = [k for k in _LEROBOT_INCOMPAT_CONFIG_FIELDS if k in cfg]
    if not stripped:
        return
    for k in stripped:
        cfg.pop(k, None)
    try:
        with cfg_path.open("w") as f:
            _json.dump(cfg, f, indent=2)
        print(
            f"[rollout] stripped LeRobot-incompatible config fields: {stripped}",
            flush=True,
        )
    except OSError as exc:
        print(f"[rollout] config.json patch skipped: {exc}", flush=True)




if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Synchronous LeRobot inference for LeIsaac simulation."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument(
    "--episode_length_s", type=float, default=60.0, help="Episode length in seconds."
)
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="lerobot-smolvla",
    help="Local LeRobot policy type. Use lerobot-, for example lerobot-smolvla.",
)
parser.add_argument(
    "--policy_action_horizon",
    type=int,
    default=16,
    help="Number of actions to execute per policy call.",
)
parser.add_argument(
    "--policy_language_instruction",
    type=str,
    default=None,
    help="Language instruction of the policy.",
)
parser.add_argument(
    "--policy_checkpoint_path",
    type=str,
    required=True,
    help="Path to the local LeRobot checkpoint.",
)
parser.add_argument(
    "--debug_policy_shapes",
    action="store_true",
    help="Print observation and action tensor shapes around each local LeRobot inference call.",
)
parser.add_argument(
    "--video_out",
    type=str,
    default="",
    help=(
        "Optional path to write a side-by-side wrist+front MP4 replay. "
        "Frames are taken from the same observation tensors the policy "
        "sees (zero extra render cost) and piped to ffmpeg. Empty string "
        "disables recording. Backend leaderboard sets this to "
        "{LOCAL_STAGE_DIR}/{submission_id}/eval_video.mp4 and moves the "
        "result to NAS after the eval completes."
    ),
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=10,
    help="Capture rate (frames/sec) for --video_out. Sim runs at 60Hz; "
         "default 10 fps grabs every 6th sim step.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import time
from typing import Any

import omni.ui as ui
import omni.kit.app
import omni.kit.viewport.utility as vp_util

import carb
import gymnasium as gym
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

import leisaac  # noqa: F401
import simulator.tasks  # noqa: F401
from simulator.tasks.external import resolve_task
from simulator import FRANKA_JOINT_NAMES


def setup_dual_viewports():
    """Setup dual viewports: main perspective view and GoPro camera view."""
    perspective_path = "/World/envs/env_0/Robot/panda_hand/wrist"

    # Get main viewport window
    v1_window = ui.Workspace.get_window("Viewport")
    if not v1_window:
        print("Error: Main viewport window not found")
        return

    v1_api = vp_util.get_viewport_from_window_name("Viewport")
    if v1_api:
        v1_api.camera_path = perspective_path

    # Get or create secondary viewport window
    v2_window = ui.Workspace.get_window("Viewport 2")
    if not v2_window:
        v2_window = vp_util.create_viewport_window("Viewport 2")
        # Important: Wait for UI to register the new window
        omni.kit.app.get_app().update()  # Synchronous frame update

    v2_api = vp_util.get_viewport_from_window_name("Viewport 2")
    if v2_api:
        v2_api.camera_path = f"/World/front_camera"

    # Ensure both windows exist before docking
    if v1_window and v2_window:
        # Wait for UI to stabilize before docking
        omni.kit.app.get_app().update()

        # Attempt docking with error handling
        try:
            v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
            print("Viewports docked: [Viewport (Persp)] | [Viewport 2 (Camera)]")
        except Exception as e:
            print(f"Docking failed: {str(e)}")
            # Alternative docking approach if direct docking fails
            try:
                # Try docking after another frame
                omni.kit.app.get_app().update()
                v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
                print("Viewports docked on second attempt")
            except Exception as e2:
                print(f"Second docking attempt failed: {str(e2)}")
    else:
        print("Error: Could not find one or both viewport windows for docking.")


class _EvalVideoRecorder:
    """Pipe raw RGB frames from the observation cameras into an ffmpeg
    subprocess that encodes to H.264 MP4.

    Why use the obs cameras instead of grabbing the viewport:
      * No extra render cost — the policy already triggered the camera
        renders on this frame, we're just reusing the tensors.
      * The result IS what the policy saw, which is the most useful
        thing to show students afterwards.
      * Deterministic resolution — viewport sizes vary with the
        compositor's window state in headless Isaac Sim.

    The recorder is forgiving: any error closes the pipe and disables
    future frames, but never raises into the eval loop. A broken
    recording must never fail a submission.
    """

    def __init__(self, out_path: str, fps: int):
        self.out_path = out_path
        self.fps = max(1, int(fps))
        self.proc = None
        self.width = None
        self.height = None
        self.frames_written = 0
        self._broken = False

    def _spawn(self, width: int, height: int) -> None:
        import subprocess as _sp
        self.width, self.height = width, height
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self.out_path,
        ]
        try:
            self.proc = _sp.Popen(cmd, stdin=_sp.PIPE)
            print(
                f"[rollout] eval recording → {self.out_path} "
                f"({width}x{height} @ {self.fps} fps)",
                flush=True,
            )
        except (OSError, FileNotFoundError) as exc:
            print(f"[rollout] could not start ffmpeg ({exc}); video disabled", flush=True)
            self._broken = True
            self.proc = None

    def write(self, obs_dict: dict) -> None:
        """Stitch wrist + front and push one frame.

        Both cameras are (H, W, 3) uint8 tensors with a leading batch
        dim, sized 480x640 by the env config. Side-by-side gives a
        1280x480 frame.
        """
        if self._broken:
            return
        wrist = obs_dict.get("wrist")
        front = obs_dict.get("front")
        if wrist is None or front is None:
            return
        try:
            w_arr = wrist.cpu().numpy().astype(np.uint8)[0]
            f_arr = front.cpu().numpy().astype(np.uint8)[0]
            frame = np.concatenate([w_arr, f_arr], axis=1)
        except Exception as exc:  # noqa: BLE001
            print(f"[rollout] frame conversion failed: {exc}", flush=True)
            self._broken = True
            return
        if self.proc is None:
            h, w = frame.shape[:2]
            self._spawn(w, h)
            if self._broken or self.proc is None:
                return

        # Diagnostic: hash a few frames' raw bytes so we can tell, without
        # decoding the resulting MP4, whether the obs cameras are actually
        # producing different content per capture. If the hash is the same
        # for many consecutive frames the recorder is OK and the camera
        # render didn't advance (sim / sensor cadence bug). EVAL_DEBUG_ACTIONS
        # gates the print so production runs stay quiet.
        if os.environ.get("EVAL_DEBUG_ACTIONS", "").strip().lower() == "true" \
                and self.frames_written < 20:
            import hashlib as _hl
            w_hash = _hl.md5(w_arr.tobytes()).hexdigest()[:10]
            f_hash = _hl.md5(f_arr.tobytes()).hexdigest()[:10]
            print(
                f"[recorder frame={self.frames_written}] "
                f"wrist md5={w_hash} mean={w_arr.mean():.2f} "
                f"front md5={f_hash} mean={f_arr.mean():.2f}",
                flush=True,
            )

        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames_written += 1
        except (BrokenPipeError, OSError) as exc:
            print(f"[rollout] ffmpeg pipe broke after {self.frames_written} frames: {exc}", flush=True)
            self._broken = True
            self.proc = None

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
        print(
            f"[rollout] eval recording finalized: {self.frames_written} frames "
            f"→ {self.out_path}",
            flush=True,
        )
        self.proc = None


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        rendered = False
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
            rendered = True

        # CRITICAL: at least one render() per env.step is required for
        # cameras to refresh in headless Isaac Sim. When the policy
        # inference (e.g. diffusion DDPM at 100 steps) is slower than
        # sleep_duration, the while-loop body never runs, env.sim.render()
        # is never called, and BOTH the recorder AND the policy obs keep
        # serving the same stale camera frame for the whole episode —
        # which then looks exactly like a "collapsed policy" because the
        # diffusion model sees identical visual input every step and
        # outputs the same near-init action every step. Force one
        # render here so the next env.step's obs reflects an updated
        # scene. ~10-30ms overhead is negligible against 1s/step
        # diffusion inference.
        if not rendered:
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        if (
            hasattr(self, "_input")
            and hasattr(self, "_keyboard")
            and hasattr(self, "_keyboard_sub")
        ):
            self._input.unsubscribe_from_keyboard_events(
                self._keyboard, self._keyboard_sub
            )
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def _shape_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    return type(value).__name__


def _print_mapping_shapes(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key in sorted(values):
        print(f"  {key}: {_shape_summary(values[key])}")


class LeRobotSyncPolicy:
    """Local LeRobot inference path matching the async server pipeline."""

    def __init__(
        self,
        policy_type: str,
        pretrained_name_or_path: str,
        task_type: str,
        camera_infos: dict[str, tuple[int, int]],
        actions_per_chunk: int,
        device: str,
        debug_policy_shapes: bool = False,
    ):
        if actions_per_chunk <= 0:
            raise ValueError(
                f"policy_action_horizon must be positive, got {actions_per_chunk}."
            )

        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.debug_policy_shapes = debug_policy_shapes

        if task_type == "so101leader":
            self.state_joint_names = SINGLE_ARM_JOINT_NAMES
            self.action_dim = len(SINGLE_ARM_JOINT_NAMES)
        elif task_type == "franka_panda":
            self.state_joint_names = FRANKA_JOINT_NAMES
            self.action_dim = 8
        else:
            raise ValueError(
                f"Task type {task_type} not supported for synchronous LeRobot inference yet."
            )

        self.lerobot_features = self._build_lerobot_features(camera_infos)
        self.camera_keys = list(camera_infos.keys())

        print(
            f"Loading local LeRobot policy '{policy_type}' from {pretrained_name_or_path}...",
            flush=True,
        )
        # Strip training-only fields that newer LeRobot adds but the
        # inference-side LeRobot doesn't accept. Safe because these flags
        # never affect inference. See _patch_lerobot_config above.
        _patch_lerobot_config(pretrained_name_or_path)
        policy_class = get_policy_class(policy_type)
        self.policy = policy_class.from_pretrained(pretrained_name_or_path, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()

        device_override = {"device": device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        print("Local LeRobot policy is ready.", flush=True)

    def reset(self):
        policy_reset = getattr(self.policy, "reset", None)
        if callable(policy_reset):
            policy_reset()

    def _build_lerobot_features(
        self, camera_infos: dict[str, tuple[int, int]]
    ) -> dict[str, dict]:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_joint_names),),
                "names": [f"{joint_name}.pos" for joint_name in self.state_joint_names],
            }
        }
        for camera_key, camera_image_shape in camera_infos.items():
            features[f"observation.images.{camera_key}"] = {
                "dtype": "image",
                "shape": (camera_image_shape[0], camera_image_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _build_raw_observation(self, observation_dict: dict) -> dict[str, Any]:
        raw_observation = {
            key: observation_dict[key].cpu().numpy().astype(np.uint8)[0]
            for key in self.camera_keys
        }
        raw_observation["task"] = observation_dict["task_description"]

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        elif self.task_type == "franka_panda":
            joint_pos = observation_dict["joint_pos"].cpu().numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for synchronous LeRobot inference yet."
            )

        for joint_index, joint_name in enumerate(self.state_joint_names):
            raw_observation[f"{joint_name}.pos"] = joint_pos[0, joint_index].item()

        return raw_observation

    def _config_horizon_summary(self) -> str:
        names = ["chunk_size", "n_action_steps", "action_chunk_size", "action_horizon"]
        values = []
        for name in names:
            if hasattr(self.policy.config, name):
                values.append(f"{name}={getattr(self.policy.config, name)}")
        return ", ".join(values) if values else "no known horizon fields found"

    def _prepare_observation(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        observation = raw_observation_to_observation(
            raw_observation,
            self.lerobot_features,
            self.policy.config.image_features,
        )
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Prepared observation:", observation)

        observation = self.preprocessor(observation)
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Preprocessed observation:", observation)
        return observation

    def _predict_lerobot_actions(self, observation: dict[str, Any]) -> torch.Tensor:
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        return self.postprocessor(action)

    def _convert_actions_to_leisaac(self, action_tensor: torch.Tensor) -> np.ndarray:
        if self.task_type == "so101leader":
            actions = convert_lerobot_action_to_leisaac(action_tensor)
        elif self.task_type == "franka_panda":
            actions = action_tensor.to("cpu").numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for synchronous LeRobot inference yet."
            )

        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action values for task type {self.task_type}, got {actions.shape[-1]}."
            )
        return actions

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        raw_observation = self._build_raw_observation(observation_dict)
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Raw observation:", raw_observation)

        observation = self._prepare_observation(raw_observation)
        action_tensor = self._predict_lerobot_actions(observation)
        actions = self._convert_actions_to_leisaac(action_tensor)
        return torch.from_numpy(actions[:, None, :])


# Policy types that can't be loaded in this Python 3.11 interpreter (they
# require lerobot >=0.5.1, which pins Python >=3.12, while isaacsim 5.1.0
# pins Python ==3.11). For these, inference runs in a sidecar venv and
# we exchange observations / actions over stdin/stdout. Extend this set
# whenever a new such policy type is added upstream.
SIDECAR_POLICY_TYPES: set[str] = {"multi_task_dit"}


class LeRobotSidecarPolicy:
    """Drives a separate Python 3.12 interpreter that hosts a lerobot >=0.5.1
    policy. The interface mirrors LeRobotSyncPolicy so the main loop is
    agnostic to which path runs.

    The sidecar interpreter is fixed at construction time; if it crashes
    mid-eval we propagate as an exception (no automatic restart — easier
    to surface real bugs in the worker logs)."""

    def __init__(
        self,
        policy_type: str,
        pretrained_name_or_path: str,
        task_type: str,
        camera_infos: dict[str, tuple[int, int]],
        actions_per_chunk: int,
        device: str,
        debug_policy_shapes: bool = False,
    ):
        import subprocess
        import pickle as _pickle
        import struct as _struct

        if actions_per_chunk <= 0:
            raise ValueError(
                f"policy_action_horizon must be positive, got {actions_per_chunk}."
            )

        self._pickle = _pickle
        self._struct = _struct
        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.debug_policy_shapes = debug_policy_shapes

        if task_type == "so101leader":
            self.state_joint_names = SINGLE_ARM_JOINT_NAMES
            self.action_dim = len(SINGLE_ARM_JOINT_NAMES)
        elif task_type == "franka_panda":
            self.state_joint_names = FRANKA_JOINT_NAMES
            self.action_dim = 8
        else:
            raise ValueError(
                f"Task type {task_type} not supported for sidecar LeRobot inference yet."
            )

        self.camera_keys = list(camera_infos.keys())
        lerobot_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_joint_names),),
                "names": [f"{j}.pos" for j in self.state_joint_names],
            }
        }
        for k, shape in camera_infos.items():
            lerobot_features[f"observation.images.{k}"] = {
                "dtype": "image",
                "shape": (shape[0], shape[1], 3),
                "names": ["height", "width", "channels"],
            }

        sidecar_py = os.environ.get(
            "LEROBOT_SIDECAR_PYTHON", "/opt/lerobot-py312/bin/python"
        )
        sidecar_script = str(_Path(__file__).resolve().parent / "policy_sidecar.py")
        print(
            f"[rollout] launching sidecar: {sidecar_py} {sidecar_script}",
            flush=True,
        )
        self._proc = subprocess.Popen(
            [sidecar_py, "-u", sidecar_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent's stderr so child logs land in eval log
            bufsize=0,
        )

        self._rpc({
            "op": "init",
            "policy_type": policy_type,
            "checkpoint_dir": pretrained_name_or_path,
            "device": device,
            "lerobot_features": lerobot_features,
        })
        print(f"Sidecar LeRobot policy '{policy_type}' is ready.", flush=True)

    # ── IPC helpers ────────────────────────────────────────────────
    # The sidecar runs under numpy 2.x and the parent under numpy 1.x, so we
    # can't let pickle embed numpy-version-specific module paths in the wire
    # format. Serialise ndarrays as plain ``(dtype, shape, bytes)`` blobs in
    # both directions. Strings, floats, ints, plain dicts/lists/tuples
    # pickle fine across numpy versions.
    _ND_TAG = "__ndarray__"

    def _encode_obj(self, obj):
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            return {
                self._ND_TAG: True,
                "dtype": str(arr.dtype),
                "shape": tuple(arr.shape),
                "data": arr.tobytes(),
            }
        if isinstance(obj, dict):
            return {k: self._encode_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._encode_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._encode_obj(v) for v in obj)
        return obj

    def _decode_obj(self, obj):
        if isinstance(obj, dict):
            if obj.get(self._ND_TAG):
                return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
            return {k: self._decode_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._decode_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._decode_obj(v) for v in obj)
        return obj

    def _write_frame(self, obj) -> None:
        body = self._pickle.dumps(self._encode_obj(obj), protocol=self._pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(self._struct.pack("<Q", len(body)))
        self._proc.stdin.write(body)
        self._proc.stdin.flush()

    def _read_frame(self):
        header = self._proc.stdout.read(8)
        if not header or len(header) < 8:
            rc = self._proc.poll()
            raise RuntimeError(
                f"Sidecar exited unexpectedly (rc={rc}). See stderr above."
            )
        (n,) = self._struct.unpack("<Q", header)
        body = bytearray()
        while len(body) < n:
            chunk = self._proc.stdout.read(n - len(body))
            if not chunk:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Sidecar truncated response (rc={rc}). See stderr above."
                )
            body += chunk
        return self._decode_obj(self._pickle.loads(bytes(body)))

    def _rpc(self, payload: dict) -> dict:
        self._write_frame(payload)
        reply = self._read_frame()
        if not reply.get("ok"):
            raise RuntimeError(
                f"Sidecar error on op={payload.get('op')!r}: "
                f"{reply.get('error')!s}\n{reply.get('traceback', '')}"
            )
        return reply

    # ── public API mirroring LeRobotSyncPolicy ─────────────────────
    def reset(self) -> None:
        self._rpc({"op": "reset"})

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        raw_observation = {
            k: observation_dict[k].cpu().numpy().astype(np.uint8)[0]
            for k in self.camera_keys
        }
        raw_observation["task"] = observation_dict["task_description"]

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        elif self.task_type == "franka_panda":
            joint_pos = observation_dict["joint_pos"].cpu().numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for sidecar inference."
            )
        for i, name in enumerate(self.state_joint_names):
            raw_observation[f"{name}.pos"] = joint_pos[0, i].item()

        reply = self._rpc({"op": "predict", "raw_observation": raw_observation})
        action_np = reply["action"]
        action_tensor = torch.from_numpy(action_np).to(self.device)

        if self.task_type == "so101leader":
            actions = convert_lerobot_action_to_leisaac(action_tensor)
        elif self.task_type == "franka_panda":
            actions = action_tensor.to("cpu").numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for sidecar inference."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action values for task {self.task_type}, "
                f"got {actions.shape[-1]}."
            )
        return torch.from_numpy(actions[:, None, :])

    def close(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._write_frame({"op": "shutdown"})
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def preprocess_obs_dict(obs_dict: dict, language_instruction: str):
    obs_dict["task_description"] = language_instruction
    return obs_dict


def get_policy_type(policy_type_arg: str) -> str:
    if policy_type_arg.startswith("lerobot-"):
        return policy_type_arg.split("lerobot-", 1)[1]
    return policy_type_arg


def get_camera_infos(
    env: ManagerBasedRLEnv, policy_obs_dict: dict
) -> dict[str, tuple[int, int]]:
    camera_infos = {}
    for key, sensor in env.scene.sensors.items():
        if isinstance(sensor, Camera) and key in policy_obs_dict:
            camera_infos[key] = sensor.image_shape
    return camera_infos


def main():
    task_id = resolve_task(args_cli.task)
    args_cli.task = task_id
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1)
    task_type = get_task_type(task_id)
    robot_name = getattr(env_cfg, "robot_name", None)
    policy_task_type = "franka_panda" if robot_name == "franka_panda" else task_type
    teleop_device = "keyboard" if policy_task_type == "franka_panda" else task_type
    env_cfg.use_teleop_device(teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(task_id, cfg=env_cfg).unwrapped

    # Warm up the renderer before the first reset. Headless Isaac Sim with
    # camera observations otherwise hangs the first env.reset() while the
    # Vulkan / DLSS / shader pipeline compiles — the worker sees no output
    # for several minutes and the eval looks dead. A handful of app updates
    # forces shader compilation and material warm-up to happen here, where
    # we can attribute it.
    print("[rollout] warming up renderer (20 app updates)...", flush=True)
    for _ in range(20):
        simulation_app.update()
    print("[rollout] resetting environment...", flush=True)
    obs_dict, _ = env.reset()
    print("[rollout] env.reset() returned", flush=True)

    language_instruction = args_cli.policy_language_instruction
    if language_instruction is None:
        language_instruction = getattr(env_cfg, "task_description", None)

    policy_obs_dict = preprocess_obs_dict(obs_dict["policy"], language_instruction)
    camera_infos = get_camera_infos(env, policy_obs_dict)
    print(
        f"[rollout] camera_infos = {camera_infos}; loading policy...",
        flush=True,
    )

    resolved_policy_type = get_policy_type(args_cli.policy_type)
    policy_cls = (
        LeRobotSidecarPolicy
        if resolved_policy_type in SIDECAR_POLICY_TYPES
        else LeRobotSyncPolicy
    )
    print(
        f"[rollout] policy_type={resolved_policy_type} → using {policy_cls.__name__}",
        flush=True,
    )
    policy = policy_cls(
        policy_type=resolved_policy_type,
        pretrained_name_or_path=args_cli.policy_checkpoint_path,
        task_type=policy_task_type,
        camera_infos=camera_infos,
        actions_per_chunk=args_cli.policy_action_horizon,
        device=args_cli.device,
        debug_policy_shapes=args_cli.debug_policy_shapes,
    )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()
    controller.reset()

    setup_dual_viewports()

    # Eval video recorder. Empty --video_out (or recorder spawn failure
    # later) is a no-op — never blocks the eval. capture_stride throttles
    # us to args_cli.video_fps from the env's args_cli.step_hz so we
    # don't try to encode every sim step.
    recorder = _EvalVideoRecorder(args_cli.video_out, args_cli.video_fps) \
        if args_cli.video_out else None
    capture_stride = max(1, args_cli.step_hz // max(1, args_cli.video_fps))
    sim_step_counter = 0

    # Diagnostic: when EVAL_DEBUG_ACTIONS=true, print the actual action
    # values + joint_pos for the first N sim steps of episode 1 so the
    # admin can tell at a glance whether the policy is outputting near-
    # zero (model collapsed), constant (bad checkpoint), or genuinely
    # varying values (policy is fine — investigate env / action interp).
    debug_actions = os.environ.get("EVAL_DEBUG_ACTIONS", "").strip().lower() == "true"
    # Coalesce empty / whitespace to default — docker-compose's
    # `${VAR:-}` substitution sets unset vars to "" and int("") raises.
    debug_steps_cap = int((os.environ.get("EVAL_DEBUG_ACTION_STEPS", "") or "12").strip() or "12")
    if debug_actions:
        print(
            f"[debug-action] enabled — will dump action+joint_pos for "
            f"first {debug_steps_cap} sim steps of episode 1",
            flush=True,
        )

    success_count, episode_count = 0, 1
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    policy.reset()
                    episode_count += 1
                    break

                policy_obs_dict = preprocess_obs_dict(
                    obs_dict["policy"], language_instruction
                )
                actions = policy.get_action(policy_obs_dict).to(env.device)
                if (debug_actions and episode_count == 1
                        and sim_step_counter < debug_steps_cap):
                    # Dump the predicted chunk + current physical state +
                    # PD target. Comparing the three columns settles
                    # any "is the action manager forwarding the policy
                    # output?" question:
                    #
                    #   action          ← what the policy emitted
                    #   joint_pos_target← what the PD controller was
                    #                    asked to track (i.e. what
                    #                    env's action manager translated
                    #                    the action into)
                    #   joint_pos       ← where the joint actually is
                    #   joint_vel       ← how fast joints are moving
                    #
                    # If action ≈ joint_pos_target the env is faithful;
                    # then any motion mismatch is purely policy + PD.
                    try:
                        a_np = actions.detach().cpu().numpy()
                        first_action = a_np[0, 0].tolist() if a_np.size else []
                        action_mean = float(a_np.mean()) if a_np.size else 0.0
                        action_std = float(a_np.std()) if a_np.size else 0.0

                        def _flat_round(t, k=4, n=9):
                            if t is None:
                                return []
                            return [round(v, k) for v in
                                    t.detach().cpu().flatten().tolist()[:n]]

                        jp = obs_dict["policy"].get("joint_pos")
                        jpt = obs_dict["policy"].get("joint_pos_target")
                        jv = obs_dict["policy"].get("joint_vel")

                        print(
                            f"[debug-action step={sim_step_counter}] "
                            f"action[0]={[round(v, 4) for v in first_action]} "
                            f"chunk_mean={action_mean:.4f} chunk_std={action_std:.4f}",
                            flush=True,
                        )
                        print(
                            f"[debug-action step={sim_step_counter}]   "
                            f"joint_pos       ={_flat_round(jp)}",
                            flush=True,
                        )
                        print(
                            f"[debug-action step={sim_step_counter}]   "
                            f"joint_pos_target={_flat_round(jpt)}",
                            flush=True,
                        )
                        print(
                            f"[debug-action step={sim_step_counter}]   "
                            f"joint_vel       ={_flat_round(jv)}",
                            flush=True,
                        )
                    except Exception as _exc:
                        print(f"[debug-action] print failed: {_exc}", flush=True)
                for action_index in range(
                    min(args_cli.policy_action_horizon, actions.shape[0])
                ):
                    action = actions[action_index, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, teleop_device)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    sim_step_counter += 1
                    if recorder and sim_step_counter % capture_stride == 0:
                        recorder.write(obs_dict["policy"])
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                policy.reset()
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                episode_count += 1
                policy.reset()
                break
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )

    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    if recorder:
        recorder.close()

    # Sidecar policies own a subprocess; ask them to shut down cleanly so the
    # next eval doesn't inherit a dangling child. No-op for in-process policies.
    close_fn = getattr(policy, "close", None)
    if callable(close_fn):
        close_fn()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
