# AI Capstone

Sim-to-real imitation-learning pipeline for robot manipulation tasks. Record human demonstrations with UMI, process them through SLAM, generate synthetic data in Isaac Lab, train a diffusion policy with LeRobot, and evaluate it in simulation.

> **Platform:** Linux only.

For a complete step-by-step walkthrough, see [Getting Started](docs/getting_started.md).

# Human Demonstration Data Processing

1. **Installation**

   ```bash
   uv sync --package umi
   ```

2. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate
   ```

   This makes `hf`, `lerobot-train`, and other installed commands available in your terminal.

3. **Hugging Face login**

   Create an access token at: <https://huggingface.co/docs/hub/en/security-tokens>

   Then log in:

   ```bash
   hf auth login --token <YOUR_HF_TOKEN>
   ```

4. **Set your Hugging Face username**

   Commands throughout this project use `${HF_USER}`. Set it once per terminal session:

   ```bash
   export HF_USER=<your-huggingface-username>
   ```

## After recording the demonstration videos, follow this practice

1. Under `data/`, create a directory for this demo. Suggested name: `YYYYMMDD-taskname`. Add a `raw_videos/` subdirectory under it.
2. Place the recorded videos in `data/YYYYMMDD-taskname/raw_videos/`.

## Verify the recorded demonstration videos

The SLAM mapping stage is fragile. To save time, run the verify pipeline first:

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/verify_pipeline.yaml \
    --session-dir <demo_directory_name>
```

## If verification fails, re-record and copy into the demo directory

There are several failure modes:

### SLAM failures

Pipeline raises:

```
RuntimeError: SLAM mapping failed. Check logs at datasets/team_asia/demos/mapping/slam_stdout.txt for details.
```

Re-record the mapping video, replace the file, and re-run the verification pipeline.

## If verification succeeds, run the full pipeline

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/build_dataset.yaml \
    --session-dir <demo_directory_name> \
    --task <kitchen|dining_room|living_room>
```

Upload the whole session directory to the Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_id> data/<demo_directory_name>/demos/mapping/object_poses.json
```

# Data Creation in Simulator

## Prerequisites

1. **Linux machine with Nvidia GPU** — verify with `nvidia-smi`. Isaac Lab requires a Linux host with an Nvidia driver.
2. **Docker installed** — the simulator runs inside a container.
3. **Repository cloned** — if you haven't already:
   ```bash
   git clone https://github.com/HCIS-Lab/aicapstone.git
   cd aicapstone
   ```

## Launch Isaac Lab

```bash
make launch-isaaclab
```

This builds the Isaac Sim container. On success, the shell drops you inside the container.

Download the session directory produced by the UMI pipeline:

```bash
hf download ${HF_USER}/<repo_id> --local-dir data/<demo_directory_name>
```

## Run the data generation pipeline

The `--lerobot_dataset_repo_id` should be your own Hugging Face dataset repo.

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

`--object_poses` still provides the base UMI scene setups, but `--augment_pose_factor` lets Step 3 replay more synthetic episodes from the same small set of demonstrations. For example, `16` source entries with `--augment_pose_factor 10` become `160` replay episodes.

Upload the recorded dataset to Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/
```

# LeRobot Training

Training runs on the **host machine** (not inside Docker) and produces a policy checkpoint from your generated dataset. Requires an Nvidia GPU.

See [LeRobot Training Procedure](docs/lerobot_training.md) for the full command reference, multi-GPU setup, and troubleshooting.

# LeRobot Rollout

Rollout loads your trained policy into the Isaac Lab simulator (inside the Docker container) to evaluate robot performance.

See [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) for the full procedure.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | End-to-end pipeline walkthrough |
| [Developer Introduction](docs/dev/introduction.md) | Repo layout, environment setup, where to run what |
| [Isaac Lab + LeIsaac Configuration Tutorial](docs/isaaclab_leisaac_tutorial.md) | Configuring Isaac Lab with LeIsaac |
| [LeRobot Dataset Visualizer](docs/lerobot_dataset_visualizer.md) | Visualizing LeRobot datasets |
| [LeRobot Checkpoint Format](docs/lerobot_model_format.md) | Understanding LeRobot model checkpoint structure |
| [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) | Running trained policies in the simulator |
| [LeRobot Training Procedure](docs/lerobot_training.md) | Training imitation-learning policies |
| [Standalone Env Config Export](docs/standalone_env_config_export.md) | Exporting environment configs as standalone files |
| [Synthetic Data Generation Pipeline](docs/synthetic_data_generation.md) | Generating synthetic training data |
| [UMI Pipeline](docs/umi_pipeline.md) | Data collection and processing with UMI |

## License

MIT — see [LICENSE](LICENSE).
