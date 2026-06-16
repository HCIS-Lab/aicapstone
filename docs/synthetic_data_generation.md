# Synthetic Data Generation Pipeline

End-to-end walkthrough of producing a LeRobot-format dataset from an Isaac Lab task and pushing it to the Hugging Face Hub. The running example is the in-tree **cup-stacking** task (`LeIsaac-HCIS-CupStacking-SingleArm-v0`) — every path below points at a real file on `main`.

> Companion docs:
> - [`isaaclab_leisaac_tutorial.md`](./isaaclab_leisaac_tutorial.md) — the env-config tree and templates.
> - [`standalone_env_config_export.md`](./standalone_env_config_export.md) — why a task config must live as its own file.
> - [`lerobot_model_format.md`](./lerobot_model_format.md) — the on-disk layout of a trained policy (downstream of this pipeline).

The pipeline has five stages:

1. Lay out the task package.
2. Implement a scripted state machine.
3. Wire the state machine into the datagen entry point.
4. Run data generation.
5. Upload the resulting dataset to Hugging Face.

---

## 1. Task package layout

A task lives under `packages/simulator/src/simulator/tasks/<task_slug>/`. For cup-stacking:

```
packages/simulator/src/simulator/tasks/cup_stacking/
├── __init__.py                 # gym.register(...)
├── cup_stacking_env_cfg.py     # CupStackingEnvCfg
└── mdp/
    └── terminations.py         # task-specific termination terms
```

### 1.1 Env config — `cup_stacking_env_cfg.py`

Defines `CupStackingEnvCfg` by subclassing `SingleArmFrankaTaskEnvCfg` (the in-tree single-arm Franka template). The config wires:

- a `CupStackingSceneCfg` with the kitchen USD, a `blue_cup` and a `pink_cup` `RigidObjectCfg`,
- a `TerminationsCfg` whose `success` term is `blue_cup_on_top_pink_cup` (x/y window + height threshold over the pink cup),
- `task_description = "pick up the blue cup and place it on the pink cup."` (baked into every LeRobot frame's `"task"` field),
- `__post_init__` hooks for viewer pose, Franka home joint pose, kitchen sub-asset extraction, and the optional UMI `object_poses_path` block.

See [`standalone_env_config_export.md`](./standalone_env_config_export.md) for the rules every env config must follow (file location, naming, `@configclass`, `task_description`, `__post_init__` ordering).

### 1.2 Task-specific MDP terms — `mdp/terminations.py`

Holds the `task_done` predicate used by the env's success termination. Per-task MDP code (terminations, rewards, events) lives under `mdp/` so it does not leak into the shared template.

### 1.3 Gym registration — `__init__.py`

The package's `__init__.py` **must** call `gym.register(...)` so Isaac Lab can resolve the task id from `--task ...`. The cup-stacking registration that ships in the tree:

```python
import gymnasium as gym


gym.register(
    id="HCIS-CupStacking-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cup_stacking_env_cfg:CupStackingEnvCfg",
    },
)
```

The `entry_point` is **always** `isaaclab.envs:ManagerBasedRLEnv` for tasks built on the manager-based template. Only the `env_cfg_entry_point` string changes per task.

### 1.4 Surfacing the task — `tasks/__init__.py`

Registration only fires on import. The **parent** package `packages/simulator/src/simulator/tasks/__init__.py` must import the task subpackage so that `import simulator.tasks` runs the `gym.register(...)` block:

```python
from . import cup_stacking  # noqa: F401
```

Forgetting this line is the most common reason `--task HCIS-CupStacking-SingleArm-v0` fails with "task is not registered" — the file exists, the gym call exists, but nothing imported it.

---

## 2. Scripted state machine

Path: `packages/simulator/src/simulator/datagen/state_machine/cup_stacking.py`.

The state machine is the scripted "policy" that produces successful demos. Subclass `leisaac.datagen.state_machine.base.StateMachineBase` and implement:

| Method | Responsibility |
|--------|---------------|
| `setup(env)` | One-time calibration. Resolve joint and body indices, write the Franka rest pose, capture the rest-pose EE world position. |
| `pre_step(env)` | Optional per-step hook before action computation (cup-stacking does not need it). |
| `get_action(env)` | Return the action tensor for this step — joint position targets + gripper command. |
| `advance()` | Tick the scripted timeline (per-phase step counter, phase index). |
| `reset()` | Per-episode reset. Clear phase state and cached targets; **keep** setup-time calibration. |
| `check_success(env)` | Return `True` iff the episode's success criterion holds (used to gate dataset export). |
| `is_episode_done` | Property — True once the scripted timeline has run through every phase. |

### 2.1 Phases

`CupStackingStateMachine` runs seven phases gated by per-phase step counts (`_events_dt`):

1. **Hover** — interpolate from the current EE pose to a point above the blue cup (`_HOVER_Z_OFFSET = 0.15`).
2. **Approach** — descend toward the blue cup (`_GRASP_Z_OFFSET = 0.08`).
3. **Grasp** — close the gripper (`_GRIPPER_CLOSE = -1.0`).
4. **Lift** — raise the cup (`_LIFT_Z_OFFSET = 0.20`).
5. **Move above pink** — translate over the pink cup at lift height.
6. **Lower / release** — descend to `_RELEASE_Z_OFFSET = 0.09` above the pink cup, open the gripper.
7. **Retreat** — lift back up, gripper open.

`advance()` increments the per-phase counter; when it hits `_events_dt[event]` the state machine moves to the next phase. Once every phase has played out, `is_episode_done` flips to `True`.

### 2.2 EE-pose → joint-target conversion

The cup-stacking env is configured with the keyboard action setup, so the action vector is `[panda_joint1, ..., panda_joint7, gripper]`. The state machine plans in **world-space EE waypoints** and converts each step's pose error into a joint delta via a damped least-squares (DLS) Jacobian IK:

1. Look up the target pose in the robot root frame.
2. Clip the position delta to `_MAX_CARTESIAN_DELTA = 0.018` and the rotation delta (axis-angle) to `_MAX_ROT_DELTA = 0.08`.
3. Pull the body Jacobian at `panda_hand` from `robot.root_physx_view.get_jacobians()`, rotate it into the root frame.
4. Solve `Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx` with `λ = _IK_DLS_LAMBDA = 0.01`.
5. Add `Δq` to the current arm joint positions and clamp to soft joint limits.
6. Concatenate the gripper command on the end.

Keeping per-step deltas small is important: the recorder samples actions at the env step rate, and policies trained on jittery, large-step demos generalize poorly.

### 2.3 Success check

`check_success(env)` mirrors the env's `success` termination: it pulls `blue_cup` and `pink_cup` world positions, subtracts the env origin, and returns True iff the blue cup is inside `_SUCCESS_X_RANGE × _SUCCESS_Y_RANGE` over the pink cup. This is the predicate the recorder consults before exporting an episode.

---

## 3. Wire into `scripts/datagen/generate.py`

The datagen entry point selects a state machine by task id from a registry. To add a task, import its state machine and add an entry to `TASK_REGISTRY`:

```python
from simulator.datagen.state_machine.cup_stacking import CupStackingStateMachine

TASK_REGISTRY = {
    ...,
    "HCIS-CupStacking-SingleArm-v0": (CupStackingStateMachine, "keyboard"),
}
```

The entry is a `(StateMachineClass, teleop_device_name)` tuple. The teleop device string is fed to `env_cfg.use_teleop_device(...)` and decides which `actions.arm_action` / `actions.gripper_action` get filled in (cup-stacking uses `"keyboard"`, which gives the joint-position action layout the state machine plans against).

The cup-stacking entry is already wired in tree — this section is the pattern to follow when adding a new task.

---

## 4. Run data generation

From inside the Isaac Lab container (see `make launch-isaaclab` in the README):

```bash
python scripts/datagen/generate.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id HF-USER/name \
    --augment_pose_factor 10 \
    --augment_global_xy_jitter 0.01 \
    --augment_local_xy_jitter 0.01 \
    --object_poses data/<demo_directory_name>/object_poses.json
```

Key flags:

| Flag | Effect |
|------|--------|
| `--task` | Gym task id. **Must match** the `id=...` passed to `gym.register(...)` in the task's `__init__.py`. |
| `--num_envs 1` | Number of parallel envs. State machines are scripted against world-space targets and run fine at `num_envs=1`. |
| `--device cuda` | Run physics + rendering on GPU. |
| `--enable_cameras` | Required for the `wrist` and `front` `TiledCameraCfg` observations to render. Without it, the recorded LeRobot frames have no image features. |
| `--record` | Turn on the recorder manager; configures `env_cfg.recorders` and the `success` termination wiring. |
| `--use_lerobot_recorder` | Swap the default `StreamingRecorderManager` for `LeRobotRecorderManager`, which writes the LeRobot dataset format on disk instead of HDF5. |
| `--lerobot_dataset_repo_id HF-USER/name` | Passed straight into `LeRobotDatasetCfg(repo_id=..., fps=args_cli.lerobot_dataset_fps)`. Names the on-disk dataset and the eventual HF Hub repo. |
| `--lerobot_dataset_fps` | Frame rate the dataset is written at. Default `30`. |
| `--object_poses data/.../object_poses.json` | Base replay set from UMI. Every `status == "full"` entry becomes one source episode before augmentation. |
| `--augment_pose_factor 10` | Multiply the replay set in Step 3. Example: `16` source entries become `160` replay episodes. |
| `--augment_global_xy_jitter`, `--augment_local_xy_jitter` | Scene-level and per-object translation jitter in meters. |
| `--augment_yaw_jitter_deg` | Optional world-yaw jitter. Keep it at `0` first if you want the safest scripted rollouts. |
| `--augment_mix_objects` | Recombine object poses across episodes before jittering. Higher diversity, slightly higher risk. |
| `--cutlery_eval_pose_fraction`, `--cutlery_eval_pose_jitter` | For the cutlery task only: rewrite part of the augmented replay set so fork/knife starts resemble the eval initial-pose distribution, while keeping the remaining episodes broader. |
| `--resume` | Append to an existing dataset (`EXPORT_SUCCEEDED_ONLY_RESUME`) instead of starting fresh. |
| `--seed` | Optional. Defaults to `int(time.time())`. |

What happens at runtime:

1. `parse_env_cfg(task_name, ...)` resolves the env config via the gym registration.
2. `_configure_env_cfg(...)` flips `env_cfg.recorders.dataset_export_mode` to `EXPORT_SUCCEEDED_ONLY` and rewires the `success` termination so the recorder controls episode endings.
3. `_replace_recorder_manager(...)` instantiates `LeRobotRecorderManager(env_cfg.recorders, LeRobotDatasetCfg(repo_id=..., fps=...), env)`.
4. The main loop calls `sm.pre_step → sm.get_action → env.step → sm.advance` until `sm.is_episode_done`. On episode end, `sm.check_success(env)` decides whether the recorder commits the episode.
5. Once every replay episode from `--object_poses` (after optional augmentation) has been attempted, the script exits cleanly and `recorder_manager.finalize()` writes the LeRobot dataset to disk.

The dataset lands locally first — the `repo_id` only names the directory at this stage. Upload is a separate step.

---

## 5. Upload to Hugging Face

`LeRobotRecorderManager` writes a directory in the LeRobot dataset format. Push it to the Hub with:

```bash
hf upload <lerobot_dataset_repo_id> --repo-type dataset
```

The user must be authenticated first:

```bash
hf auth login
```

After upload the dataset is ready for `lerobot-train` on the host machine (see the README's "LeRobot & Hugging Face Hub workflow" section for the host-side training flow).

### 5.1 Inspect the uploaded dataset

You can preview your uploaded LeRobot dataset (episodes, camera views, actions) in the browser via the Hugging Face dataset visualizer:

<https://huggingface.co/spaces/lerobot/visualize_dataset>

Paste your `repo_id` (e.g. `HF-USER/name`) into the Space to browse the dataset.

---

## Quick checklist

- [ ] `packages/simulator/src/simulator/tasks/<task>/__init__.py` calls `gym.register(...)`.
- [ ] `packages/simulator/src/simulator/tasks/__init__.py` imports the new subpackage.
- [ ] State machine subclasses `leisaac.datagen.state_machine.base.StateMachineBase` and implements `setup`, `pre_step`, `get_action`, `advance`, `reset`, `check_success`, `is_episode_done`.
- [ ] `scripts/datagen/generate.py::TASK_REGISTRY` has `<task id>: (StateMachineClass, "<teleop_device>")`.
- [ ] Run with `--record --use_lerobot_recorder --enable_cameras --object_poses ... --lerobot_dataset_repo_id HF-USER/name`.
- [ ] `hf auth login`, then `hf upload <repo_id> --repo-type dataset`.
