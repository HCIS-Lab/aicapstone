.PHONY: install install-dev test \
	submodules submodules-pull \
	build-isaaclab launch-isaaclab \
	launch-isaaclab-glowsai-4090 launch-isaaclab-glowsai-l40s \
	check-isaaclab-gpu

# ---- Config ------------------------------------------------------------------
IMAGE          ?= leisaac-isaaclab:latest
DOCKERFILE     ?= Dockerfile
GPU            ?= all
CONTAINER_NAME ?= isaaclab

# ---- Shared shell snippets ---------------------------------------------------
# Pick first existing NVIDIA Vulkan ICD and export VK_ICD_FILENAMES.
define select_vulkan_icd
unset VK_ICD_FILENAMES; \
for icd in \
	/usr/share/vulkan/icd.d/nvidia_icd.json \
	/etc/vulkan/icd.d/nvidia_icd.json; do \
	if [ -f "$$icd" ]; then \
		export VK_ICD_FILENAMES="$$icd"; \
		echo "Using Vulkan ICD: $$VK_ICD_FILENAMES"; \
		break; \
	fi; \
done; \
if [ -z "$${VK_ICD_FILENAMES:-}" ]; then \
	echo "WARNING: No NVIDIA Vulkan ICD found."; \
	echo "Check nvidia-container-toolkit / driver installation."; \
fi
endef

# Verify required GL/X/Vulkan libs are visible to ldconfig.
define require_runtime_libs
for lib in libGLU.so.1 libXt.so.6 libX11.so.6 libvulkan.so.1; do \
	if ! ldconfig -p | grep -q "$$lib"; then \
		echo "Missing $$lib in image." >&2; \
		exit 1; \
	fi; \
done
endef

# ---- Submodules --------------------------------------------------------------
submodules:
	git submodule update --init --recursive

submodules-pull:
	git submodule update --remote --recursive

# ---- Python env --------------------------------------------------------------
install: submodules
	uv sync

install-dev: submodules
	uv sync --extra dev

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev pytest \
		tests/test_repo_layout.py \
		tests/test_external_task_resolver.py

# ---- Docker image ------------------------------------------------------------
build-isaaclab: submodules
	docker build -f $(DOCKERFILE) -t $(IMAGE) .

# ---- Launch: default ---------------------------------------------------------
launch-isaaclab: build-isaaclab
	@set -e; \
	xhost +local:root >/dev/null || true; \
	trap 'xhost -local:root >/dev/null || true' EXIT; \
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		--gpus '"device=$(GPU)"' \
		--net=host \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v $(shell pwd):/workspace/aicapstone \
		-v /workspace/aicapstone/.venv \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
		-v /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro \
		-e DISPLAY=$$DISPLAY \
		-e OMNI_KIT_ACCEPT_EULA=Y \
		-e PRIVACY_CONSENT=Y \
		-e QT_X11_NO_MITSHM=1 \
		-e NVIDIA_VISIBLE_DEVICES=$(GPU) \
		-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
		$(IMAGE) \
		bash -lc '\
			set -e; \
			echo "== GPU check =="; nvidia-smi || true; \
			echo "== Vulkan ICD candidates =="; \
			ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true; \
			$(select_vulkan_icd); \
			$(require_runtime_libs); \
			cd /workspace/aicapstone; \
			exec /bin/bash \
		'

# ---- Launch: GlowsAI RTX 4090 (VNC display :1) ------------------------------
launch-isaaclab-glowsai-4090: build-isaaclab
	@set -e; \
	docker run --rm -it \
		--name $(CONTAINER_NAME)-glowsai-4090 \
		--gpus '"device=0"' \
		--device /dev/dri/card0:/dev/dri/card0 \
		--net=host \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--shm-size=16g \
		-v $(shell pwd):/workspace/aicapstone \
		-v /workspace/aicapstone/.venv \
		-v /home/glows/.Xauthority:/root/.Xauthority:ro \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /opt/VirtualGL:/opt/VirtualGL:ro \
		-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
		-e DISPLAY=:1 \
		-e USE_VNC=1 \
		-e OMNI_KIT_ACCEPT_EULA=Y \
		-e PRIVACY_CONSENT=Y \
		-e QT_X11_NO_MITSHM=1 \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
		$(IMAGE) \
		bash -lc '\
			set -e; \
			echo "=== GlowsAI RTX 4090 ==="; \
			echo "Display: $$DISPLAY"; \
			echo "== GPU check =="; nvidia-smi || true; \
			echo "== Vulkan ICD candidates =="; \
			ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true; \
			$(select_vulkan_icd); \
			$(require_runtime_libs); \
			cd /workspace/aicapstone; \
			exec /bin/bash \
		'

# ---- Launch: GlowsAI L40S (VirtualGL + VNC display :2) -----------------------
launch-isaaclab-glowsai-l40s: build-isaaclab
	@set -e; \
	docker run --rm -it \
		--name $(CONTAINER_NAME)-glowsai-l40s \
		--gpus '"device=0"' \
		--device /dev/dri/card0:/dev/dri/card0 \
		--net=host \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--shm-size=16g \
		-v $(shell pwd):/workspace/aicapstone \
		-v /workspace/aicapstone/.venv \
		-v /home/glows/.Xauthority:/root/.Xauthority:ro \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /opt/VirtualGL:/opt/VirtualGL:ro \
		-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
		-e DISPLAY=:1 \
		-e USE_VNC=1 \
		-e VGL_DISPLAY=egl0 \
		-e PATH=/opt/VirtualGL/bin:$$PATH \
		-e OMNI_KIT_ACCEPT_EULA=Y \
		-e PRIVACY_CONSENT=Y \
		-e QT_X11_NO_MITSHM=1 \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
		$(IMAGE) \
		bash -lc '\
			set -e; \
			echo "=== GlowsAI L40S ==="; \
			echo "Display: $$DISPLAY"; \
			echo "VGL_DISPLAY: $$VGL_DISPLAY"; \
			echo "== GPU check =="; nvidia-smi || true; \
			echo "== Vulkan ICD candidates =="; \
			ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true; \
			$(select_vulkan_icd); \
			$(require_runtime_libs); \
			cd /workspace/aicapstone; \
			exec /bin/bash \
		'

# ---- GPU sanity check --------------------------------------------------------
check-isaaclab-gpu:
	@docker run --rm \
		--device nvidia.com/gpu=all \
		-e ACCEPT_EULA=Y \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-e __GLX_VENDOR_LIBRARY_NAME=nvidia \
		$(IMAGE) \
		bash -lc '\
			set -e; \
			$(select_vulkan_icd); \
			if [ -z "$${VK_ICD_FILENAMES:-}" ]; then exit 1; fi; \
			nvidia-smi; \
			ldconfig -p | grep "libGLU.so.1"; \
			ldconfig -p | grep "libXt.so.6"; \
			python -c "import torch; print(\"torch cuda available:\", torch.cuda.is_available()); print(\"torch cuda device:\", torch.cuda.get_device_name(0))"; \
			ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null || true; \
			vulkaninfo --summary; \
		'
