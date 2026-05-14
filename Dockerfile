# syntax = docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    TERM=xterm-256color

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80-retries

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gpg \
    && curl --retry 5 --retry-delay 10 --retry-all-errors -fsSL \
        "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /usr/share/keyrings/deadsnakes.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libasound2 \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libgtk-3-0 \
        libice6 \
        libnss3 \
        libsm6 \
        libvulkan1 \
        libx11-6 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxkbcommon-x11-0 \
        libxrandr2 \
        libxrender1 \
        libxt6 \
        libxtst6 \
        libxxf86vm1 \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-venv \
        unzip \
        vulkan-tools \
        wget \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && curl --retry 5 --retry-delay 10 --retry-all-errors -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && rm -f /tmp/get-pip.py \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/aicapstone

RUN python -m pip install --upgrade pip \
    && python -m pip install -U \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --upgrade \
        "isaacsim[all,extscache]==5.1.0" \
        --extra-index-url https://pypi.nvidia.com \
    && python -m pip install \
        pip==23 \
        setuptools==65 \
        flatdict==4.0.0 \
        huggingface-hub==0.35.3 \
        transformers==4.57.6

COPY dependencies/IsaacLab /workspace/aicapstone/dependencies/IsaacLab
COPY packages/simulator /workspace/aicapstone/packages/simulator

RUN test -x dependencies/IsaacLab/isaaclab.sh || \
    (echo "dependencies/IsaacLab is not initialized. Run 'git submodule update --init --recursive' before 'docker build'." >&2; exit 1)

RUN python -m pip install --no-deps \
        setuptools==65 \
        wheel==0.45.1 \
        toml==0.10.2 \
        packaging==23.0 \
        poetry-core==2.2.1 \
    && sed -i 's|-m pip install"|-m pip install --no-build-isolation"|' \
        dependencies/IsaacLab/isaaclab.sh \
    && cd dependencies/IsaacLab \
    && ./isaaclab.sh --install

RUN python -m pip install --no-deps numpy==1.26.0

RUN python -m pip install --upgrade \
        setuptools==80.10.2 \
        wheel==0.45.1 \
        cython==3.0.11 \
        toml==0.10.2 \
        packaging==24.2

RUN printf "numpy==1.26.0\n" > /tmp/sim-constraints.txt \
    && python -m pip install --use-deprecated=legacy-resolver \
        --no-build-isolation \
        -c /tmp/sim-constraints.txt \
        -e packages/simulator \
    && python -m pip install --no-deps numpy==1.26.0 \
    && rm -f /tmp/sim-constraints.txt

RUN python -m pip install --upgrade pip==26.0.1 \
    && python -m pip install --no-deps numpy==1.26.0

# ── Sidecar venv for lerobot >=0.5.1 policies (multi_task_dit, etc.) ─────────
# isaacsim 5.1 pins Python ==3.11 and lerobot 0.5.1 pins Python >=3.12, so
# these two cannot coexist in one interpreter. We install Python 3.12 + a
# dedicated venv whose only purpose is to host lerobot inference for policy
# types whitelisted by SIDECAR_POLICY_TYPES in scripts/rollout.py. The main
# (3.11) interpreter drives Isaac Sim and shells out to this venv via
# scripts/policy_sidecar.py, exchanging observations / actions over pipes.
# Placed at the end of the Dockerfile so unrelated changes don't bust cache.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/lerobot-py312 \
    && /opt/lerobot-py312/bin/pip install --upgrade pip \
    && /opt/lerobot-py312/bin/pip install \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/lerobot-py312/bin/pip install \
        "lerobot[multi_task_dit]==0.5.1"

CMD ["/bin/bash"]
