# Ubuntu 24.04 (WSL2 or Native) — GPU + Docker Setup for HRM Phase 0

Goal
- Run Phase 0 entirely in Ubuntu 24.04 (WSL2 or native Linux) for a clean, minimal environment.

1) Install NVIDIA Drivers (Windows host)
- Use the latest Game Ready/Studio driver that supports Blackwell and WSL2 CUDA
- Reboot if prompted

2) Enable WSL2 and Install Ubuntu 24.04
- In PowerShell (Admin):
```
wsl --install -d Ubuntu-24.04
```
- After install, update:
```
sudo apt update && sudo apt -y upgrade
```

3) Install Docker and NVIDIA Container Toolkit inside Ubuntu 24.04
```
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit
distribution=$( . /etc/os-release; echo $ID$VERSION_ID )
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker || true
```

4) Validate GPU access in containers (CUDA 13 / Ubuntu 24.04 based images)
```
# Option A (PyTorch 25.06; CUDA 13)
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.06-py3 nvidia-smi
# Option B (Triton 25.06; TensorRT 10.13.x)
docker run --rm --gpus all nvcr.io/nvidia/tritonserver:25.06-py3 nvidia-smi
```

5) Clone workspace into WSL2 filesystem
- Prefer /home/<user>/workspace to avoid Windows path performance issues
- Example:
```
mkdir -p ~/workspace && cd ~/workspace
# If repo exists on Windows, you can re-clone here for speed
```

6) Build and run training container (inside Ubuntu)
```
cd ~/workspace/Augment_Agent_Evolution
docker build -f containers/hrm/Dockerfile.pytorch-cuda13 -t hrm-train:25.06 .
docker run --rm -it --gpus all -v $PWD:/workspace hrm-train:25.06 bash
```

7) Proceed with Phase 0 Runbook
- Follow docs/augment_code/22_hrm_phase0_runbook.md entirely inside Ubuntu 24.04

Tips
- Avoid mounting Windows paths (e.g., /mnt/c/...) into containers for performance
- Use WSL2 Ubuntu’s native path (~/workspace/...) for cloning and building
- Keep training container separate from Triton container; share artifacts via /workspace/models

Zen MCP Optimization
- Use Zen MCP to review runbook and environment flags (torch.compile modes, CUDA Graphs, SDPA vs FlashAttention) and suggest tweaks
- Capture W&B metrics and record exact CUDA/driver/NGC image versions for reproducibility

