# Getting Started

End-to-end guide for the AI Capstone imitation-learning pipeline: from recording human demonstrations to training and evaluating a robot manipulation policy in simulation.

## Overview

This project builds imitation-learning policies for robot manipulation tasks (cup stacking, cutlery arrangement, toy blocks collection). The pipeline has five stages:

1. **Record** human demonstrations using the UMI device
2. **Process** recordings through a SLAM reconstruction pipeline
3. **Simulate** — generate synthetic training data in Isaac Lab
4. **Train** a policy model with LeRobot
5. **Evaluate** the trained policy in the simulator (rollout)

## Where to run what

| Stage | Environment | Why |
|-------|-------------|-----|
| UMI recording + SLAM processing | Local machine  | Lightweight video + SLAM pipeline |
| Simulation (datagen + rollout) | GlowsAI | Isaac Sim / Isaac Lab / Vulkan stack pinned in image |
| Training (`lerobot-train`) | GlowsAI | Docker adds I/O overhead — train natively for throughput |

> **GlowsAI VNC terminal:** When using the VNC desktop terminal on GlowsAI, switch to the `glows` user first:
> ```bash
> su - glows
> ```

For details on the repo layout and developer setup, see [Developer Introduction](dev/introduction.md).

## Prerequisites

- **Python 3.11** — required by the project (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — Python package manager used for this workspace
- **Git** — with submodule support
- **Docker + Docker Compose** — required for simulation steps (Isaac Lab runs in a container)
- **exiftool** — used by the UMI pipeline for video metadata extraction
- **ffmpeg** — used by the UMI pipeline for video processing
- **Linux machine with Nvidia GPU + driver** — required for simulation and training. Verify with `nvidia-smi`.
- **Hugging Face account** — datasets and model checkpoints are stored on the Hub. Create a token at <https://huggingface.co/docs/hub/en/security-tokens>.

> **No local GPU?** Training and simulation require a Linux machine with an Nvidia GPU.
> See [Cloud GPU Setup with GlowsAI](https://docs.google.com/presentation/d/1xPHm3gNNvLbX7aOWnY_xW5dfVLEa9XiBuskMbRsgxI8/edit?usp=sharing) for instructions on using cloud GPU instances.

### Installing prerequisites

```bash
# Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker

# exiftool and ffmpeg
sudo apt-get install -y libimage-exiftool-perl ffmpeg
```

Verify installations:

```bash
docker --version
docker compose version
exiftool -ver
ffmpeg -version
```

## Installation

### UMI Pipeline only (local machine, no GPU)

For recording and SLAM processing:

```bash
uv sync --package umi
source .venv/bin/activate
```

### Full setup (GPU machine)

For simulation, training, and rollout:

```bash
# Initialize submodules (required before Docker build)
make submodules

# Install all Python dependencies
uv sync
source .venv/bin/activate

# Build and launch the Isaac Lab container (simulation, datagen, rollout)
# Pick the target matching your GlowsAI GPU:
make launch-isaaclab-glowsai-4090   # RTX 4090
make launch-isaaclab-glowsai-l40s   # L40S
```

### Hugging Face login

Required for both setups:

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

Set `HF_USER` each terminal session — commands throughout this project reference it.

---

## Step 1: Record Human Demonstrations

> **Run on: local machine (no GPU needed).** Physical recording with the UMI device.

Each session needs three kinds of footage recorded with the same GoPro camera:

| Footage | Purpose |
|---------|---------|
| Mapping video | Build the SLAM map and locate ArUco tag #13 |
| Gripper calibration video(s) | Measure gripper finger separation range |
| Demo videos | The actual task demonstrations |

After recording:

1. Create a directory under `data/`. Suggested name: `YYYYMMDD-taskname`.
   Recommended task names:
   - `kitchen` — cup stacking
   - `dining_room` — cutlery arrangement
   - `living_room` — toy blocks collection
2. Place all recorded videos in `data/YYYYMMDD-taskname/raw_videos/`.

For detailed recording tips and failure mode guidance, see [UMI Pipeline](umi_pipeline.md).

---

## Step 2: Process Recordings (SLAM Pipeline)

**Recommended: run on your local machine.** This step does not require a GPU — it runs the SLAM reconstruction pipeline on CPU. Install the UMI package locally (see [UMI Pipeline only](#umi-pipeline-only-local-machine-no-gpu)) before proceeding.

### Device-specific pipeline configurations

Each GoPro device has its own camera intrinsics calibration. Use the pipeline configuration that matches your device:

| GoPro device | Verify config | Intrinsics file |
|--------------|---------------|-----------------|
| 交2 | `verify_pipeline_C2.yaml` | `gopro13_intrinsics_2_7k_C2.json` |
| 交6 | `verify_pipeline_C6.yaml` | `gopro13_intrinsics_2_7k_C6.json` |
| 交9 | `verify_pipeline_C9.yaml` | `gopro13_intrinsics_2_7k_C9.json` |

All configuration files are in `umi_pipeline_configs/` and reference intrinsics from `packages/umi/defaults/calibration/`. Using the wrong device config will produce incorrect ArUco detection and calibration results.

### 2a. Verify first

The SLAM mapping stage is fragile. Run verification before the full pipeline to catch bad recordings early. **Use the config matching your GoPro device:**

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/verify_pipeline_C2.yaml \
    --session-dir <demo_directory_name>
```

**If verification fails:** re-record the problematic video (usually the mapping video), replace it, and re-run verification.

Common failure:
```
RuntimeError: SLAM mapping failed. Check logs at datasets/team_asia/demos/mapping/slam_stdout.txt for details.
```

### 2b. Build dataset

Once verification passes. **Use the config matching your GoPro device:**

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/build_dataset_C2.yaml \
    --session-dir <demo_directory_name> \
    --task <kitchen|dining_room|living_room>
```

| GoPro device | Build dataset config |
|--------------|----------------------|
| 交2 | `build_dataset_C2.yaml` |
| 交6 | `build_dataset_C6.yaml` |
| 交9 | `build_dataset_C9.yaml` |

Upload the `object_poses.json` produced by the pipeline to Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_id> data/<demo_directory_name>/demos/mapping/object_poses.json
```

This file contains per-episode object poses that drive the simulator's scene setup in Step 3.

### 2c. Merge object poses (optional)

To combine object poses from multiple recording sessions into a single dataset:

```bash
uv run umi merge-object-poses data/<session_dir_1> data/<session_dir_2>
```

This creates a new session directory (e.g., `data/merged_<name1>_<name2>/`) containing the merged `demos/mapping/object_poses.json`. Use `--output`/`-o` to specify a custom output directory:

```bash
uv run umi merge-object-poses data/<session_dir_1> data/<session_dir_2> -o data/my_merged_session
```

---

## Step 3: Generate Synthetic Data in Simulation

> **Run on: GPU machine, inside Docker container.** Requires Linux + Nvidia GPU + Docker (see Prerequisites).

### Launch the Isaac Lab container

Pick the target matching your GlowsAI GPU:

| Target | GPU |
|--------|-----|
| `make launch-isaaclab-glowsai-4090` | RTX 4090 (VNC display :1) |
| `make launch-isaaclab-glowsai-l40s` | L40S (VNC display :1) |

```bash
# GlowsAI — pick one:
make launch-isaaclab-glowsai-4090
make launch-isaaclab-glowsai-l40s
```

All remaining commands in this step run **inside the container**.

### Download the UMI output

```bash
hf download ${HF_USER}/<repo_id> --local-dir data/<demo_directory_name>
```

### Run data generation

`object_poses.json` only supplies the per-episode object placements used to initialize the scene. If your UMI pipeline only gives you a small number of usable entries, expand them here with pose augmentation before training.

Available tasks:
- `HCIS-CupStacking-SingleArm-v0`
- `HCIS-CutleryArrangement-SingleArm-v0`
- `HCIS-ToyBlocksCollection-SingleArm-v0`

```bash
python scripts/datagen/generate.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/<repo_id> \
    --augment_pose_factor 10 \
    --augment_global_xy_jitter 0.01 \
    --augment_local_xy_jitter 0.01 \
    --object_poses data/<demo_directory_name>/object_poses.json
```

Recommended starting point when you only have about `16` usable UMI entries:

- `--augment_pose_factor 10` to turn `16` base entries into about `160` replay episodes.
- Keep `--augment_yaw_jitter_deg 0` at first for maximum stability.
- Add `--augment_mix_objects` only if you need more diversity after confirming the scripted policy still succeeds reliably.

### Upload the generated dataset

```bash
hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/
```

For the full data generation pipeline reference, see [Synthetic Data Generation Pipeline](synthetic_data_generation.md).

---

## Step 4: Train a Policy

> **Run on: GPU machine, host (not inside Docker).

It uses your uploaded dataset from Step 3 and produces a trained policy checkpoint.

**Quick start:**

```bash
uv sync && source .venv/bin/activate

lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --policy.type=diffusion \
  --output_dir=<your-output-dir> \
  --job_name=cupstacking \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

Training takes a few hours to over a day depending on dataset size and GPU.

For the complete flag reference, multi-GPU training, troubleshooting, and upload instructions, see [LeRobot Training Procedure](lerobot_training.md).

---

## Step 5: Evaluate in Simulation (Rollout)

> **Run on: GPU machine, inside Docker container.** Same Isaac Lab environment as Step 3.

Loads your trained policy and runs it in the simulator to evaluate performance.

```bash
# Inside the container
python scripts/rollout.py \
    --task=<eval_task_file> \
    --policy_type=lerobot-<policy_name> \
    --policy_checkpoint_path=<path/to/checkpoint> \
    --policy_action_horizon=1 \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=<num_rounds> \
    --episode_length_s=20
```

Placeholders:

| Placeholder | Meaning |
|-------------|---------|
| `<eval_task_file>` | Path to an evaluation task file under `eval/` (see table below) |
| `<policy_name>` | LeRobot policy variant — e.g. `diffusion`, `act`, `pi0` (full flag becomes `lerobot-diffusion`, etc.) |
| `<path/to/checkpoint>` | Local path to the trained policy checkpoint directory |
| `<num_rounds>` | Number of evaluation episodes to roll out |

Pick `--task` from the evaluation task files under `eval/`:

| Task | Eval file |
|------|-----------|
| Cup stacking | `eval/cup_stacking_eval.py` |
| Cutlery arrangement | `eval/cutlery_arrangement_eval.py` |
| Toy blocks collection | `eval/toy_blocks_collection_eval` |

For the full procedure including model download and flag reference, see [LeRobot Rollout (Policy Evaluation)](lerobot_rollout.md).

---

## See Also

| Document | Description |
|----------|-------------|
| [Developer Introduction](dev/introduction.md) | Repo layout, environment setup, where to run what |
| [Isaac Lab + LeIsaac Configuration Tutorial](isaaclab_leisaac_tutorial.md) | Configuring Isaac Lab with LeIsaac |
| [Keyboard Teleoperation](keyboard_teleoperation.md) | Drive the Franka from the keyboard for debugging or recording |
| [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md) | Inspecting datasets before training |
| [LeRobot Checkpoint Format](lerobot-model-format.md) | Understanding model checkpoint structure |
| [Standalone Env Config Export](standalone_env_config_export.md) | Exporting environment configs as standalone files |
| [Synthetic Data Generation Pipeline](synthetic_data_generation.md) | Full datagen pipeline reference |
| [UMI Pipeline](umi_pipeline.md) | Recording tips, SLAM details, failure modes |
