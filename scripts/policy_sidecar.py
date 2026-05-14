"""Out-of-process LeRobot inference for policies that require lerobot >=0.5.1.

This runs in a Python 3.12 venv (`/opt/lerobot-py312`) with lerobot[multi_task_dit]
installed. The parent process (rollout.py, Python 3.11 with Isaac Sim) drives
the sim loop and talks to this child over a binary protocol. We do this because
isaacsim 5.1 pins to Python 3.11 while lerobot 0.5.1 (the first release shipping
multi_task_dit) requires Python >=3.12 — the two can't coexist in one venv.

Wire protocol — length-prefixed pickle frames:
    [8-byte little-endian uint64 N][N bytes of pickle]

The protocol uses a duplicated stdout FD held in `_protocol_out`. Python's
`sys.stdout` and the OS-level fd 1 are redirected to stderr at startup so
random prints from lerobot internals (CLIP "LOAD REPORT" tables,
huggingface_hub progress, etc.) don't corrupt our protocol stream.

Parent → child (over stdin):
    {"op": "init", "policy_type", "checkpoint_dir", "device", "lerobot_features"}
    {"op": "predict", "raw_observation"}
    {"op": "reset"}
    {"op": "shutdown"}              (or stdin EOF)

Child → parent (over the duplicated stdout fd):
    init / reset:    {"ok": True}   or  {"ok": False, "error", "traceback"}
    predict:         {"ok": True, "action": np.ndarray}
                  or {"ok": False, "error", "traceback"}
    shutdown:        no reply; exits 0.

stderr is inherited by the parent — log lines from this file land in the eval log.
"""
# Bind aside the real stdout (binary protocol channel) and redirect every
# subsequent stdout write — Python-level AND fd-level — to stderr, so noisy
# imports below (lerobot, transformers, hf_hub) can't poison the wire.
# Must happen BEFORE any other module-level imports.
import os as _os
import sys as _sys
_protocol_out_fd = _os.dup(1)
_os.dup2(2, 1)
_sys.stdout = _sys.stderr  # any `print(...)` after this point goes to stderr

import json as _json
import pickle
import struct
import sys
import traceback
from pathlib import Path

import numpy as np  # noqa: F401  (kept so pickle of numpy actions resolves)
import torch

# Opened unbuffered so write_frame's 8-byte header + payload land as one
# contiguous wire image, with no risk of partial flushes mid-frame.
_protocol_out = _os.fdopen(_protocol_out_fd, "wb", buffering=0)


# Mirror of rollout.py's _patch_lerobot_config — strips training-only fields
# that newer checkpoints carry but the inference-side from_pretrained rejects.
_LEROBOT_INCOMPAT_CONFIG_FIELDS = (
    "use_peft", "resize_shape", "crop_ratio", "compile_model", "compile_mode",
)


def _patch_lerobot_config(checkpoint_dir: str) -> None:
    cfg_path = Path(checkpoint_dir) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open("r") as f:
            cfg = _json.load(f)
    except (OSError, ValueError):
        return
    stripped = [k for k in _LEROBOT_INCOMPAT_CONFIG_FIELDS if k in cfg]
    if not stripped:
        return
    for k in stripped:
        cfg.pop(k, None)
    try:
        with cfg_path.open("w") as f:
            _json.dump(cfg, f, indent=2)
        print(f"[sidecar] patched config.json: stripped {stripped}", file=sys.stderr, flush=True)
    except OSError as exc:
        print(f"[sidecar] config.json write skipped: {exc}", file=sys.stderr, flush=True)


_ND_TAG = "__ndarray__"


def _encode_obj(obj):
    """Recursively replace numpy ndarrays with portable blobs.

    Pickling raw ndarrays embeds numpy-version-specific module paths (e.g.
    numpy 2.x → `numpy._core.multiarray._reconstruct`) that the other end
    can't import when its numpy is 1.x. We strip that dependency by using
    only ``dtype string + shape tuple + raw bytes``.
    """
    import numpy as _np  # local import: keep top-level fast
    if isinstance(obj, _np.ndarray):
        arr = _np.ascontiguousarray(obj)
        return {
            _ND_TAG: True,
            "dtype": str(arr.dtype),
            "shape": tuple(arr.shape),
            "data": arr.tobytes(),
        }
    if isinstance(obj, dict):
        return {k: _encode_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_encode_obj(v) for v in obj)
    return obj


def _decode_obj(obj):
    import numpy as _np
    if isinstance(obj, dict):
        if obj.get(_ND_TAG):
            return _np.frombuffer(obj["data"], dtype=_np.dtype(obj["dtype"])).reshape(obj["shape"])
        return {k: _decode_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_decode_obj(v) for v in obj)
    return obj


def _write_frame(stream, obj) -> None:
    body = pickle.dumps(_encode_obj(obj), protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(body)))
    stream.write(body)
    stream.flush()


def _read_frame(stream):
    header = stream.read(8)
    if not header or len(header) < 8:
        return None
    (n,) = struct.unpack("<Q", header)
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            return None
        body += chunk
    return _decode_obj(pickle.loads(bytes(body)))


class _Engine:
    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.lerobot_features = None

    def init(self, payload: dict) -> None:
        # Imports are deferred until init so an import-error message gets
        # delivered through the protocol instead of crashing at startup.
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        policy_type = payload["policy_type"]
        checkpoint_dir = payload["checkpoint_dir"]
        device = payload["device"]
        self.lerobot_features = payload["lerobot_features"]

        _patch_lerobot_config(checkpoint_dir)
        print(
            f"[sidecar] loading policy_type={policy_type} from {checkpoint_dir} on {device}",
            file=sys.stderr, flush=True,
        )
        cls = get_policy_class(policy_type)
        self.policy = cls.from_pretrained(checkpoint_dir, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()

        device_override = {"device": device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        print("[sidecar] policy ready", file=sys.stderr, flush=True)

    def reset(self) -> None:
        fn = getattr(self.policy, "reset", None)
        if callable(fn):
            fn()

    def predict(self, raw_observation: dict) -> np.ndarray:
        from lerobot.async_inference.helpers import raw_observation_to_observation

        observation = raw_observation_to_observation(
            raw_observation,
            self.lerobot_features,
            self.policy.config.image_features,
        )
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        return action.detach().to("cpu").numpy()


def _serve() -> int:
    engine = _Engine()
    stdin = sys.stdin.buffer
    stdout = _protocol_out  # the saved binary channel; sys.stdout points at stderr now

    while True:
        msg = _read_frame(stdin)
        if msg is None:
            return 0  # parent closed stdin
        op = msg.get("op")
        try:
            if op == "init":
                engine.init(msg)
                _write_frame(stdout, {"ok": True})
            elif op == "reset":
                engine.reset()
                _write_frame(stdout, {"ok": True})
            elif op == "predict":
                action = engine.predict(msg["raw_observation"])
                _write_frame(stdout, {"ok": True, "action": action})
            elif op == "shutdown":
                return 0
            else:
                _write_frame(stdout, {"ok": False, "error": f"unknown op: {op!r}"})
        except Exception as exc:
            _write_frame(stdout, {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    try:
        sys.exit(_serve())
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
