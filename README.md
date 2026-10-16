# TAFI: Feature Filtering and Targeted Adversarial Augmentation for Node-Level Graph OOD Generalization

Official implementation of **TAFI (Targeted Adversarial Feature Filtering for Invariance)** for node-level graph out-of-distribution generalization.

TAFI combines **Adaptive Feature Filtering (AFF)**, which gates source-node feature channels before neighborhood aggregation, with **Targeted Adversarial Augmentation (TAA)**, which learns structural protection and allocates adversarial edge perturbations under a fixed budget. Training alternates between the graph editor and the predictor/protection modules; inference uses the predictor.

## Requirements

Python 3.10+ is required; a CUDA-capable GPU is recommended. Install a compatible [PyTorch build](https://pytorch.org/get-started/locally/) for your system, followed by:

```bash
python -m pip install -r requirements.txt
```

Dependencies are not version-pinned. Use mutually compatible PyTorch/PyG versions; exact reproduction may require the original experiment environment.

## Data preparation

Download datasets from [GOOD](https://github.com/divelab/GOOD), [MUSAE/Twitch](https://github.com/benedekrozemberczki/MUSAE), [OGB](https://ogb.stanford.edu/docs/nodeprop/), and [Elliptic](https://www.kaggle.com/ellipticco/elliptic-data-set). Place them under `data/`, or specify another location with `--data-root`.

Expected layout (showing representative paths):

```text
data/
├── GOODCBAS/<domain>/processed/{concept,covariate}.pt
├── GOODCora/<domain>/processed/{concept,covariate}.pt
├── GOODWebKB/<domain>/processed/{concept,covariate}.pt
├── twitch/<language>/musae_<language>_{target.csv,edges.csv,features.json}
├── ogbn_arxiv/
└── Elliptic/{0.pkl,...,48.pkl}
```

- **GOOD:** Include each domain's PyG `.pt` files with node features, edges, labels, and train/validation/test masks. Domain folders are discovered automatically.
- **Twitch-Explicit:** Supply CSV/JSON files for DE (train), ENGB (validation), and ES/FR/PTBR/RU/TW (test).
- **OGB-Arxiv:** Create `data/ogbn_arxiv/`; the OGB loader downloads missing raw data when network access is available. This repository uses chronological splits, not the official OGB leaderboard split.
- **Elliptic:** Supply 49 preprocessed snapshots (`0.pkl`–`48.pkl`) under `data/Elliptic/`. Each contains `(adjacency, labels, features)`. See [SING's Elliptic preprocessing implementation](https://github.com/Codeyrw/KDD2025-SING/blob/main/elliptic/load_data.py) for conversion from the original CSVs. The evaluation follows the SING temporal protocol.

Load serialized `.pt` and `.pkl` files only from trusted sources.

## Running experiments

The commands below use the dataset profiles in `parsers.py` and default to five seeds (`0,1,2,3,4`).

```bash
# GOOD-CBAS, concept shift
python run_tafi.py --datasets GOODCBAS --good-shifts concept --data-root ./data --device cuda --output-dir ./results/GOODCBAS_concept

# Twitch-Explicit
python run_tafi.py --datasets twitch --data-root ./data --device cuda --output-dir ./results/twitch

# OGB-Arxiv
python run_tafi.py --datasets ogbn_arxiv --data-root ./data --device cuda --output-dir ./results/ogbn_arxiv

# Elliptic
python run_tafi.py --datasets Elliptic --data-root ./data --device cuda --output-dir ./results/Elliptic
```

For the complete real-world benchmark configurations (including dataset-specific tuned settings), run:

```bash
bash run.sh ./data
```

If Conda is installed, `run.sh` uses the `oodv1` environment by default. Override it with, for example, `CONDA_ENV=my_tafi_env bash run.sh ./data`. Without Conda, it uses `PYTHON_BIN` (default: `python`). Set `DEVICE=cpu` or pass `--device cpu` for CPU execution. Run `python run_tafi.py --help` for additional options.

Check output summaries and run-status/error records before interpreting the results. Hyperparameter profiles are in `parsers.py`, with two tuned configurations specified in `run.sh`.

## Synthetic graph generation

`generate_croe.py` provides `generate_relational(...)` to construct controlled synthetic graphs with known invariant and spurious edge types, returning graph arrays, edge metadata, and generation statistics.

## Citation

Citation information will be added upon publication.
