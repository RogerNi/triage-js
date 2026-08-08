# Running the artifact with Docker

These scripts run each model family from the paper **inside the container**, on
the public `dataset_demo/` subset. They mirror the SLURM scripts in
`exp_scripts/` (which target a slurm-managed non-Docker cluster and the full benchmark).

## Build

```bash
docker build -t triage-js:latest .          # from the repo root
```

Requires Docker + the **NVIDIA Container Toolkit** (for `--gpus`).

## One-command demos

| script | family | GPUs | notes |
|---|---|---|---|
| `docker/demo_gnn.sh` | GNN only | 1 | no model download |
| `docker/demo_baseline.sh` | classical ML (RF/XGBoost/LogReg/SVM) | 0 | no model download |
| `docker/demo_linear_probe.sh` | LLM linear probe | 1 | downloads LLM on first run |
| `docker/demo_lora.sh` | LLM LoRA | 1 | downloads LLM on first run |
| `docker/demo_full_multigpu.sh` | LLM full fine-tune | 2 | downloads LLM on first run |
| `docker/demo_hybrid_multigpu.sh` | GNN + LLM hybrid | 2 | downloads LLM on first run |
| `docker/demo_zero_shot.sh` | zero-shot (local instruct LLM) | 1 | no training; prompts + generates Yes/No; downloads an instruct model |

Example:

```bash
bash docker/demo_gnn.sh
CUDA=0,1 bash docker/demo_full_multigpu.sh
```

All scripts go through `docker/run.sh`, which sets the required flags:
`--gpus all`, `--ipc=host` (needed for multi-GPU NCCL), a persistent
HuggingFace cache (`$HOME/hf_cache`), and wandb disabled.

## Important notes

- The demo `code` snippets are **synthetic** (see the main README).
  Full-benchmark runs need the withheld benchmark via `exp_scripts/`.
- Single GPU is enough for the GNN/baseline/linear-probe/LoRA demos; the
  full-fine-tune / hybrid demos shard the model across the GPUs listed in `CUDA`.
