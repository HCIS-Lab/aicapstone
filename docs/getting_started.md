# Getting Started

End-to-end guide for the AI Capstone imitation-learning pipeline: generating synthetic demonstrations in simulation, then training and evaluating a robot manipulation policy.

## Overview

This project builds imitation-learning policies for robot manipulation tasks (cup stacking, cutlery arrangement, toy blocks collection). The pipeline has three stages:

1. **Simulate** — generate synthetic training data in Isaac Lab with preconfigured domain randomization
2. **Train** a policy model with LeRobot
3. **Evaluate** the trained policy in the simulator (rollout)

## Where to run what

| Stage | Environment | Why |
|-------|-------------|-----|
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
```

Verify installations:

```bash
docker --version
docker compose version
```

## Installation

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

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

Set `HF_USER` each terminal session — commands throughout this project reference it.

---

## Step 1: Generate Synthetic Data in Simulation

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

### Run data generation

Each task ships with preconfigured domain randomization: object poses are re-randomized at every environment reset, so no recorded pose file is needed. `--num_demos` sets the number of randomized episodes to run, and `--use_lerobot_recorder` saves the result in LeRobot dataset format. Only successful episodes are exported.

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
    --num_demos 20
```

The dataset lands in `~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/`.

### Upload the generated dataset

```bash
hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/
```

For the full data generation pipeline reference, see [Synthetic Data Generation Pipeline](synthetic_data_generation.md).

---

## Step 2: Train a Policy

> **Run on: GPU machine, host (not inside Docker).

It uses your uploaded dataset from Step 1 and produces a trained policy checkpoint.

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

## Step 3: Evaluate in Simulation (Rollout)

> **Run on: GPU machine, inside Docker container.** Same Isaac Lab environment as Step 1.

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
