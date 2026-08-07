# TRIAGE-JS — Learning to Triage Vulnerability Reports from Program Analysis

Artifact for the paper *"Learning to Triage Vulnerability Reports from Program
Analysis: An Empirical Study in Node.js"*. It contains the training/evaluation
code for all model families evaluated in the paper (classical ML, GNN,
fine-tuned LLMs, GNN+LLM hybrid, zero-shot LLMs), together with the dataset
construction pipeline and the scripts that reproduce every configuration.

The paper's evaluation uses the TRIAGE-JS benchmark (1,883 npm packages with
taint flows reported by NodeMedic-FINE and human-reviewed exploitability
labels). **The benchmark data is not yet public.** Because the confirmed
exploitable packages are under coordinated disclosure, the benchmark will be
released publicly after the disclosure period completes; until then,
researchers may request early access for research purposes only by contacting
the authors (see "Dataset" below).

## Getting Started

The quickest way to exercise the artifact is the Docker image together with the
included public demo subset (`dataset_demo/`) — this needs no benchmark access
and no model download. Host prerequisites are x86-64, Docker with the NVIDIA
Container Toolkit, and an NVIDIA GPU with a CUDA 12.6 driver.

**Build** (from the repository root):

```bash
docker build -t triage-js:latest .
```

**GNN test** — GNN path on the demo subset, single GPU, ~1 minute, fully
offline (no model download): `bash docker/demo_gnn.sh`. The run ends with `Training
finished!` followed by a test-metrics line, e.g.:

```
{'test_accuracy': 0.7, 'test_f1': 0.81, 'test_precision': 0.76, 'test_recall': 0.87, ...}
```
(exact values vary by GPU/run.)

which confirms data loading, training, and evaluation all work. `docker/README.md`
gives one-command runners for every model family; "Claims supported and not
supported" below states what the demo does and does not reproduce.

## Repository layout

```
dataset/                    TRIAGE-JS benchmark (NOT included — see "Dataset" below)
dataset_demo/               small public demo subset for functional testing (see "Demo subset")
data_util/                  dataset construction pipeline (raw/preprocessed archives ship with the dataset)
train.py                    GNN / LLM / hybrid training and evaluation entry point
baseline.py                 classical ML baselines (RF, XGBoost, LogReg, SVM)
model.py, ggnn.py           model definitions (classification head, GGNN)
datamodule.py               dataset loading, splits, resampling
openrouter.py               zero-shot inference via API models
exp_scripts/                SLURM scripts reproducing every trained configuration
docker/                     Docker run scripts, one per model family (see docker/README.md)
Dockerfile                  container image definition
environment.yml             conda environment specification
```

## Installation (non-Docker)

```bash
conda env create -f environment.yml
conda activate torch-pyg
```

Requirements: Python 3.10+, PyTorch + PyTorch Geometric, HuggingFace
`transformers`/`peft`/`bitsandbytes` (for 4-bit LoRA), `scikit-learn`,
`xgboost`, `shap`. GPU experiments were run on NVIDIA H100 80GB (1 GPU for
LoRA/linear-probe, 2 GPUs for full fine-tuning). Zero-shot API baselines
require an OpenRouter key in `openrouter.key`.

## Dataset

`dataset/` contains the four benchmark artifacts consumed by the code:

| file | content |
|---|---|
| `basic_metadata.csv` | one row per package: id, version, vulnerability type, label (`confirmed` = exploitable) |
| `provenance_graphs.pt` | PyTorch Geometric provenance graphs (one per flagged flow) |
| `node_information_per_graph.csv` | per-node code snippets and attributes |
| `graph_features_matrix.csv` | pooled per-graph feature matrix for classical ML |

Labels: 1,250 exploitable / 633 false alarms. Of the exploitable flows, 644
were auto-confirmed by NodeMedic-FINE's exploit synthesis; the remaining 606
(plus all false alarms) were labeled by manual review. Splits are 8:1:1
train/validation/test (1,506/188/189), fixed via `--dataset_seed 42` across
all experiments.

**Data availability.** The benchmark data is **not included in this
repository.** It will be made publicly available after the responsible
disclosure period for the confirmed exploitable packages completes. Until
then, researchers may request early access for research purposes only by
contacting the authors.

Once obtained, the dataset ships with the `data_util/` raw and preprocessed
archives; unpack them and run `data_util/build_metadata_and_graph.py`
followed by `data_util/build_llm_input.py` to (re)build `dataset/`.

### Demo subset for functional testing

So that users can understand the data format and test the functionality of the
pipeline without access to the withheld benchmark, we provide a small demo
subset in `dataset_demo/`. It is a stratified 10% sample of the graphs (188
graphs, preserving the confirmed/false-alarm ratio) with the same schema and
file layout as `dataset/`, so it is a drop-in `--data_folder` for `train.py`.

It is noteworthy that the actual package source code in
`node_information_per_graph.csv` has been **replaced with synthetic,
clearly-marked snippets**, and all package-identifying fields (name, version,
file path) have been redacted, so that no code or identity of a package under
coordinated disclosure is published. The **provenance graphs and labels are the
real benchmark values**, so the GNN path produces genuine results; the code/LLM
path runs end-to-end on the synthetic snippets, which demonstrates that the
pipeline is functional but does not reproduce the paper's numbers on real code.
The real code is part of the benchmark released after the disclosure period
completes.

## Claims supported and not supported

**Supported by this artifact.** The artifact contains the complete
training/evaluation pipeline for every model family evaluated in the paper —
classical ML baselines, GNN, LLM linear-probing, LLM LoRA, LLM full
fine-tuning, and the GNN+LLM hybrid — with per-configuration scripts
(`exp_scripts/`) and one-command Docker runners (`docker/`). A reviewer can 
build the image and run each model family end-to-end on the included demo
subset (see "Getting Started" and `docker/README.md`).

**Not reproducible from the public artifact, and why.**

- **Exact table/figure numbers** cannot be reproduced from the public release:
  the full benchmark (1,883 packages) is withheld under coordinated vulnerability
  disclosure, so only a stratified 10% demo subset is public.
- **Zero-shot frontier-API models** (OpenRouter) require a paid external service 
  and API credentials and thus are not executable without API keys. The zero-shot *local* models are runnable via `docker/demo_zero_shot.sh`.


## Reproducing the paper's results

All experiments run 5 seeds (2025–2029) over the same fixed split. Scripts
are SLURM batch files (`sbatch <script>`); adapt the headers to your cluster.

| paper result | scripts |
|---|---|
| Classical ML rows (Tables 2, 7) | `exp_scripts/baseline/run_all_baselines.sh` |
| GNN rows | `train.py --model_name=gnn_only --no_llm ...` (see hybrid scripts) |
| LLM full fine-tuning | `exp_scripts/new-cls-head/full/<model>.sh` |
| LLM LoRA (r=128) | `exp_scripts/new-cls-head/lora/<model>.sh` |
| LLM linear probing | `exp_scripts/new-cls-head/linear-probe/<family>/<model>.sh` |
| GNN+LLM hybrid | `exp_scripts/new-cls-head/full-gnn/<model>.sh` |
| Zero-shot (local models) | `exp_scripts/zero-shot/local/` |
| Zero-shot (frontier API models) | `exp_scripts/zero-shot/openrouter/` |
| LoRA rank study | `exp_scripts/new-cls-head/lora_hp_tune/` |

Expected runtimes per run (H100): linear probe 10–19 min, LoRA 30–58 min,
full fine-tuning 60–70 min, GNN ~25 min, classical baselines < 1 min on CPU
after feature extraction.

## Notes

- Vulnerability disclosures for confirmed exploitable packages follow a
  coordinated disclosure process; the benchmark will be released publicly after
  the disclosure period completes. Early access for research purposes only is
  available by contacting the authors.
