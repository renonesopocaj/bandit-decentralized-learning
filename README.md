# BanditDL — Adaptive Neighbor Sampling in Decentralized Learning

Code accompanying the EPFL *Optimization for Machine Learning* mini-project
**"Impact of Adaptive Neighbor Sampling on Optimization in Decentralized
Federated Learning"** (Bombarda, Busato, Senoner).

## 1. Overview

In decentralized (serverless) learning, every node trains locally and must spread
information through peer-to-peer exchanges. Dense gossip costs `O(n²)` messages
per round; pull-based epidemic learning instead samples a small **uniform**
random set of peers each round. This project asks an empirical question:

> **When does *adaptive* neighbor sampling beat uniform sampling at a fixed
> communication budget, and how does the answer depend on data heterogeneity?**

We cast per-round neighbor selection at each node as a **combinatorial
semi-bandit**: each candidate peer is an arm, a node selects `s` arms per round
and observes one reward per selected peer. We compare uniform sampling against
ε-greedy, EXP3, combinatorial UCB (CUCB), combinatorial Thompson sampling (CTS),
and their discounted variants, across Dirichlet, pathological-label, and natural
FEMNIST-writer partitions. Besides accuracy and loss we measure the *learned
sampling structure*, *consensus*, and *reward regret*.

### Research questions

- **Q1** Does bandit-based selection improve over uniform sampling at equal
  communication, and how does this depend on heterogeneity?
- **Q2** Does the converged sampling distribution recover the latent cluster
  structure of the partition?
- **Q3** Which reward signal best drives the bandit, and does decreasing realized
  regret correspond to better learning?

## 2. Repository structure

```
BanditDL/
├── README.md                 # this file
├── reproduce.sh              # entry point: runs the paper's experiment grids
├── pyproject.toml            # dependencies (PEP 621) + ruff config
├── uv.lock                   # pinned dependency versions
│
├── banditdl/                 # the Python package
│   ├── __main__.py           # `python -m banditdl` -> Hydra entry point
│   ├── core/
│   │   ├── sampling.py       # reward strategies + neighbor samplers (the bandits)
│   │   ├── worker/           # honest (DynamicWorker) and Byzantine workers
│   │   └── robustness/       # attacks + robust aggregators
│   ├── data/
│   │   ├── providers.py      # MNIST/CIFAR-10 (torchvision) and FEMNIST loaders
│   │   ├── partitioning.py   # synthetic (Dirichlet/pathological) + natural splits
│   │   ├── dataset.py        # builds per-node train/val/test loaders
│   │   └── models.py         # cnn_mnist, cnn_femnist, cnn_cifar (Table II archs)
│   ├── experiments/
│   │   ├── hydra_run.py      # composes config, runs seeds, plots
│   │   ├── engine.py         # the decentralized training/evaluation loop
│   │   ├── config_schema.py  # typed BanditDLConfig
│   │   └── sweep.py          # Optuna sweep orchestration
│   └── utils/                # metrics, plotting, seed averaging, math helpers
│
├── conf/                     # Hydra configuration groups (see §8)
│   ├── config.yaml           # main config; composes all groups below
│   ├── dataset/  optimization/  heterogeneity/  sampler/
│   ├── topology/ adversary/ aggregator/ evaluation/ optuna/
│
├── scripts/                  # offline figure/table generators (see §7)
├── docs/                     # extended config and sweep reference
└── tests/                    # pytest unit tests
```

## 3. Installation

The project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                   # creates .venv and installs deps
# if the uv cache directory is not writable:
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

Dependencies (declared in `pyproject.toml`): `torch`, `torchvision`,
`hydra-core`, `optuna`, `mabwiser`, `datasets`, `numpy`, `matplotlib`,
`networkx`.

## 4. Datasets

- **MNIST** and **CIFAR-10** download automatically via `torchvision` on first
  use into `banditdl/datasets/` (override with `BANDITDL_DATASET_ROOT`).
- **FEMNIST** is pulled from the HuggingFace `datasets` hub on first use
  (≈700 MB into `~/.cache/huggingface/datasets`; override with
  `HF_DATASETS_CACHE`). Pre-fetch it with:
  ```bash
  uv run python -c "from banditdl.data.femnist import _load_hf_dataset; _load_hf_dataset()"
  ```

## 5. Quick start

Check the install with a tiny run:

```bash
bash reproduce.sh smoke           # 10 MNIST nodes, 5 rounds
```

Run a single configuration directly (any `conf/` field is a Hydra override):

```bash
uv run -m banditdl \
  dataset=femnist_pool optimization=opt_femnist \
  sampler=cts sampler.reward=parameter_distance \
  heterogeneity=pathological_5g_2c topology.nodes=30 topology.sampling=0.2 \
  optimization.rounds=500 evaluation.evaluation_delta=20 seed=0
```

Inspect the fully resolved config without running:

```bash
uv run -m banditdl --cfg job
```

Each run writes a self-contained folder:

```
.hydra_runs/<date>/<time>_local/
├── .hydra/            # resolved config + overrides
├── hydra_run.log
├── results/           # metric .npy arrays + audit.json
│   └── seeds/seed_<n>/results/   # raw per-seed artifacts + partition audit
└── plots/             # auto-generated figures
```

Public arrays under `results/` are seed-averaged; `*_by_seed.npy` arrays keep the
seed dimension. Metric arrays of shape `(evaluations, nodes)` include
`validation_accuracy` (each node on its own local test set — the paper's "local
accuracy"), `global_accuracy` (every node on the shared 10 % holdout),
`validation_loss`, and `train_loss`; diagnostics include
`sampler_probabilities` / `sampler_weights` `(rounds, nodes, nodes)`,
`reward_algorithm`, `reward_oracle`, `regret`, `neighbor_disagreement`,
`consensus_drift`, and `gradient_norms`. See [docs/config.md](docs/config.md) for
the full list.

## 6. Reproducing the experiments

`reproduce.sh` encodes the exact grids behind the paper. Each named experiment
loops `uv run -m banditdl` over the reported configurations:

```bash
bash reproduce.sh femnist_main      # main study (Fig. 1, 5-7; Table I; App. C)
bash reproduce.sh cifar_discount    # discount ablation (Tables III-IV; Fig. 8-10; App. D)
bash reproduce.sh dirichlet         # Dirichlet alpha sweep (Fig. 3-4; Table I rows)
```

**Fixed factors (match the report):** `n = 30` honest nodes, mean aggregation,
one local SGD step per round, evaluation every 20 rounds, a 10 % global test
holdout, and a 20 % per-node local test holdout. Optimizer hyperparameters live
in `conf/optimization/`:

| Dataset  | Loss              | LR    | Momentum | Weight decay | Batch | Rounds (paper) |
|----------|-------------------|-------|----------|--------------|-------|----------------|
| MNIST    | NLL               | 0.05* | 0.9      | 1e-4         | 25    | 200            |
| FEMNIST  | NLL               | 0.05  | 0.9      | 1e-4         | 25    | 500            |
| CIFAR-10 | CrossEntropy      | 0.5   | 0.99     | 1e-2         | 50    | 500–2000       |

`*` MNIST uses the engine default LR (0.5) unless overridden; set
`optimization.learning_rate=0.05` to match the report exactly.

Reward signals: `parameter_distance` (inverse model distance, "IMD"),
`cosine_similarity` (cosine on weights), and `update_cosine_similarity` (cosine
on local updates, the paper's "COS"). Seeds default to `{0, 1, 2}` averaged
in-process via `num_seeds=3`.

**Runtime.** Cost is dominated by `nodes × rounds` forward/backward passes.
MNIST/FEMNIST configs take minutes to ~1 h on a single CPU; CIFAR-10 (deeper
model, 500–2000 rounds) takes a few hours per config (~15–20 s/round). The full
grids are large — shrink `SEEDS`/`ROUNDS` in `reproduce.sh`, or run a single
`uv run -m banditdl ...` line, while iterating. Add `device=cuda` (or
`device=mps`) to use an accelerator.

## 7. Generating figures and tables

The figure/table generators read finished run folders (no retraining):

| Report artifact                                   | Script |
|---------------------------------------------------|--------|
| Per-run training curves — gradient norm, neighbor disagreement, regret, loss, accuracy (Fig. 6–8, 10b/d) | auto-written to each run's `plots/`, or `scripts/plot_results.py` |
| Sampling-probability network graphs (Fig. 2, 3, 9, 10a/c) | `scripts/plot_clustering_graph.py` |
| Cluster-purity summary (does sampling recover clusters, Q2) | `scripts/analyze_clustering.py` |
| Sweep heatmaps over reward × sampler / cluster × sampler (Fig. 5) | `scripts/plot_sweep.py` |
| Discount comparison tables, γ × reward (Tables III–IV) | `scripts/plot_discount_tables.py` |

Examples (point them at the run folders produced in §6):

```bash
# Training curves for one or several runs
uv run python scripts/plot_results.py .hydra_runs/<date>/<time>_local/results \
  --metric validation_accuracy -o curves.png

# Final sampling-probability graph (cluster recovery)
uv run python scripts/plot_clustering_graph.py .hydra_runs/<date>/<time>_local \
  --weight sampler_probability --top-edges 4

# Cluster purity over the trailing window
uv run python scripts/analyze_clustering.py .hydra_runs/<date>/*_local \
  --partition pathological_5g_2c --tail 200

# Discount tables over a directory of discount-ablation runs
uv run python scripts/plot_discount_tables.py .hydra_runs/<date>
```

The Pareto-frontier scatter (Fig. 1) and reward-delta heatmaps (Fig. 4) are
assembled from the per-config scalar metrics produced above (local vs global
accuracy/loss, and the COS-minus-IMD differences); see
[docs/sweeps.md](docs/sweeps.md) for the sweep-plotting configuration.

## 8. Configuration

Hydra composes one config from the groups in `conf/`. `conf/config.yaml` sets the
defaults; override any field on the command line.

| Group           | Selects                | Key fields |
|-----------------|------------------------|------------|
| `dataset`       | dataset + model        | `mnist`, `cifar10`, `femnist` (natural writers), `femnist_pool` (synthetic) |
| `optimization`  | training schedule      | `opt_mnist`, `opt_cifar10`, `opt_femnist`; `rounds`, `learning_rate`, … |
| `heterogeneity` | partition strategy     | `dirichlet_alpha{0.5,1,10,100}`, `pathological_*`; `clusters`, `classes_per_group`, `group_overlap` |
| `sampler`       | neighbor sampler       | `uniform`, `epsilon_greedy`/`bandit`, `exp3`, `cucb`, `cts`, `discounted_{cucb,cts}`; `reward`, `params.{epsilon,gamma,exploration}` |
| `topology`      | network               | `nodes`, `sampling` (fraction of peers pulled per round) |
| `adversary`     | Byzantine setup        | `none`, `alie`; `byzcount`, `attack` |
| `aggregator`    | robust aggregation     | `mean`, `robust_trmean`; `aggregator`, `pre_aggregator`, `rag` |
| `evaluation`    | eval cadence/holdouts  | `evaluation_delta`, `global_test_ratio`, `local_test_ratio` |
| `seed` / `num_seeds` | base seed + in-process seed averaging | — |
| `device`        | `cpu`, `cuda`, `mps`, or `auto` | — |

An explicit `heterogeneity.clusters` must divide the number of honest nodes.
Full reference: [docs/config.md](docs/config.md). Sweep system (Optuna +
heatmaps): [docs/sweeps.md](docs/sweeps.md).

## 9. Tests

```bash
uv run pytest
```

Unit tests cover the samplers, partitioning, the training engine, robust
aggregation, metrics, and the plotting helpers.
