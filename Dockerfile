# TRIAGE-JS artifact image.
#
# The CUDA/cuDNN/NCCL runtime is supplied by the pip `nvidia-*-cu12==12.6.*`
# wheels pinned in environment.yml (bundled with torch 2.7.1), so a plain
# Miniconda base is sufficient — the host driver is provided at run time via
# `--gpus all` and the NVIDIA Container Toolkit.
FROM continuumio/miniconda3:latest

WORKDIR /workspace

# A C compiler is required at run time by Triton (used by bitsandbytes for the
# 4-bit / LoRA and quantized zero-shot paths) to JIT-compile kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create the pinned conda environment first (cached across code changes).
COPY environment.yml /workspace/environment.yml
RUN conda env create -f environment.yml && conda clean -afy

# Self-contained, deterministic defaults for artifact evaluation.
ENV WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HUB_OFFLINE=0

# Copy the rest of the artifact (see .dockerignore for exclusions).
COPY . /workspace

# All commands run inside the `torch-pyg` environment.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "torch-pyg"]
CMD ["bash"]
