# TRIAGE-JS — Learning to Triage Vulnerability Reports from Program Analysis

[![Dataset](https://img.shields.io/badge/Dataset-Request_Access-4285F4?logo=googledrive&logoColor=white)](https://forms.gle/U73oakGxk3YrmNyu9)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker/README.md)
[![Python](https://img.shields.io/badge/Python-3.13.5-3776AB?logo=python&logoColor=white)](environment.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Artifact for the paper *"Learning to Triage Vulnerability Reports from Program
Analysis: An Empirical Study in Node.js"*. It contains the training/evaluation
code for all model families evaluated in the paper (classical ML, GNN,
fine-tuned LLMs, GNN+LLM hybrid, zero-shot LLMs), together with the dataset
construction pipeline and scripts for every configuration.

The paper's evaluation uses the TRIAGE-JS benchmark (1,883 npm packages with
taint flows reported by NodeMedic-FINE and exploitability labels obtained
through exploit synthesis or manual review). The dataset is publicly available
for research use. Download access can be requested by completing the Google
Form listed in the "Dataset" section.

## 🚀 Getting started

The quickest way to exercise the artifact is the Docker image together with the
included demo subset (`dataset_demo/`). Host prerequisites are x86-64, Docker
with the NVIDIA Container Toolkit, and an NVIDIA GPU with a CUDA 12.6 driver.

### 1. Build the image

From the repository root:

```bash
docker build -t triage-js:latest .
```

### 2. Run the GNN demo

The GNN path runs on the demo subset with a single GPU in approximately one
minute. It is fully offline and requires no model download:

```bash
bash docker/demo_gnn.sh
```

The run ends with `Training finished!` followed by a test-metrics line, for
example:

```text
{'test_accuracy': 0.7, 'test_f1': 0.81, 'test_precision': 0.76, 'test_recall': 0.87, ...}
```

> [!NOTE]
> Exact values vary by GPU and run.

`docker/README.md` gives one-command runners for every model family.

## 🗂️ Repository layout

```
dataset/                    TRIAGE-JS benchmark (download separately; see "Dataset")
dataset_demo/               small public demo subset for functional testing (see "Demo subset")
data_util/                  dataset construction pipeline (raw/preprocessed archives ship with the dataset)
train.py                    GNN / LLM / hybrid training and evaluation entry point
baseline.py                 classical ML baselines (RF, XGBoost, LogReg, SVM)
model.py, ggnn.py           model definitions (classification head, GGNN)
datamodule.py               dataset loading, splits, resampling
openrouter.py               zero-shot inference via API models
exp_scripts/                SLURM scripts for every trained configuration
docker/                     Docker run scripts, one per model family (see docker/README.md)
Dockerfile                  container image definition
environment.yml             conda environment specification
THIRD_PARTY_NOTICES.md      attribution and licenses for adapted code
```

## 🛠️ Installation without Docker

```bash
conda env create -f environment.yml
conda activate torch-pyg
```

- **Python:** The provided Conda environment and Docker image pin Python 3.13.5.
  Other environments are not tested.
- **Dependencies:** PyTorch, PyTorch Geometric, Hugging Face
  `transformers`/`peft`/`bitsandbytes` (for 4-bit LoRA), `scikit-learn`,
  `xgboost`, and `shap`.
- **Hardware:** GPU experiments were run on NVIDIA H100 80GB GPUs: one GPU for
  LoRA/linear probing and two GPUs for full fine-tuning.
- **API models:** Zero-shot API baselines require an OpenRouter key in
  `openrouter.key`.

## 📦 Dataset

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

> [!IMPORTANT]
> **Dataset access:** The TRIAGE-JS dataset is publicly downloadable for
> research purposes. Request access using an institutional email address by
> completing the [Google Form](https://forms.gle/U73oakGxk3YrmNyu9).

Once obtained, the dataset ships with the `data_util/` raw and preprocessed
archives; unpack them and run `data_util/build_metadata_and_graph.py`
followed by `data_util/build_llm_input.py` to (re)build `dataset/`.

### 🧪 Demo subset for functional testing

`dataset_demo/` contains a small subset for testing the pipeline. It is a
stratified 10% sample of the graphs (188 graphs, preserving the
confirmed/false-alarm ratio). It contains the three files required by
`train.py` and `baseline.py`, with the same schema as their full-dataset
counterparts, so it is a drop-in `--data_folder` for both entry points. The
derived `graph_features_matrix.csv` file is not needed by these commands and is
therefore omitted from the demo.

In `node_information_per_graph.csv`, package source code is replaced with
clearly marked synthetic snippets, and identifying fields such as package name,
version, and file path are redacted. The provenance graphs and labels retain
the benchmark values.

## 📊 Running the paper's experiment configurations

All experiments run 5 seeds (2025–2029) over the same fixed split. Scripts
are SLURM batch files (`sbatch <script>`); adapt the headers to your cluster.

| paper result | scripts |
|---|---|
| Classical ML rows (Tables 2, 7) | `exp_scripts/baseline/run_all_baselines.sh` |
| GNN rows | `exp_scripts/new-cls-head/gnn/gnn_only.sh` |
| LLM full fine-tuning | `exp_scripts/new-cls-head/full/<model>.sh` |
| LLM LoRA (r=128) | `exp_scripts/new-cls-head/lora/<model>.sh` |
| LLM linear probing | `exp_scripts/new-cls-head/linear-probe/<family>/<model>.sh` |
| GNN+LLM hybrid | `exp_scripts/new-cls-head/full-gnn/<model>.sh` |
| Zero-shot (local models) | `exp_scripts/zero-shot/local/` |
| Zero-shot (frontier API models) | `exp_scripts/zero-shot/openrouter/` |
| LoRA rank study | `exp_scripts/new-cls-head/lora_hp_tune/` |

### Expected runtime per run

| Configuration | Expected runtime |
|---|---:|
| Linear probe | 10–19 min on H100 |
| LoRA | 30–58 min on H100 |
| Full fine-tuning | 60–70 min on H100 |
| GNN | ~25 min on H100 |
| Classical baselines | < 1 min on CPU after feature extraction |
