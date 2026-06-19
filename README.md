# Bandit-Based Neighbor Selection for Decentralized Learning under Data Heterogeneity

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

[Read the report](Bandit%20Based%20Neighbor%20Selection%20for%20Decentralized%20Learning%20under%20Data%20Heterogeneity.pdf)

### Research questions

- **Q1** Does bandit-based selection improve over uniform sampling at equal
  communication, and how does this depend on heterogeneity?
- **Q2** Does the converged sampling distribution recover the latent cluster
  structure of the partition?
- **Q3** Which reward signal best drives the bandit, and does decreasing realized regret correspond to better learning?
- **Q4** How does discounting reposition a sampler on the
personalization--generalization front?

## 2. Repository structure

```
BanditDL/
├── README.md                 # this file
├── reproduce.sh              # entry point: runs the predefined experiment grids
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
│   │   └── models.py         # cnn_mnist, cnn_femnist, cnn_cifar (per-dataset CNNs)
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
├── scripts/                  # offline plotting & analysis scripts (see §7)
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
`validation_accuracy` (each node on its own local test set — its local
accuracy), `global_accuracy` (every node on the shared 10 % holdout),
`validation_loss`, and `train_loss`; diagnostics include
`sampler_probabilities` / `sampler_weights` `(rounds, nodes, nodes)`,
`reward_algorithm`, `reward_oracle`, `regret`, `neighbor_disagreement`,
`consensus_drift`, and `gradient_norms`. See [docs/config.md](docs/config.md) for
the full list.

## 6. Experiment grids

`reproduce.sh` bundles a few predefined experiment grids. Each named target
loops `uv run -m banditdl` over a set of configurations:

```bash
bash reproduce.sh femnist_main      # FEMNIST: 3 cluster profiles x all samplers x 2 rewards
bash reproduce.sh cifar_discount    # CIFAR-10 discount ablation: {cucb,cts} x gamma x 2 rewards
bash reproduce.sh dirichlet         # Dirichlet heterogeneity sweep (alpha in {0.5, 1, 10})
```

## 7. Plotting and analysis

Every run auto-writes plots for all available metrics to its own `plots/`
folder. The scripts in `scripts/` regenerate or extend these offline from
finished run folders (no retraining):

| Script | What it produces |
|--------|------------------|
| `scripts/plot_results.py` | Per-run / multi-run training curves: accuracy, loss, regret, neighbor disagreement, gradient norm, sampler aggressiveness, … |
| `scripts/plot_clustering_graph.py` | Network graph of the final sampling probabilities (how concentrated each node's neighbor selection is) |
| `scripts/analyze_clustering.py` | Per-node cluster-purity summary of the converged neighbor selections |
| `scripts/plot_sweep.py` | Heatmaps from a completed Optuna sweep study |
| `scripts/plot_discount_tables.py` | Discount comparison tables (γ × reward) over a directory of runs |

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

# Discount tables over a directory of runs
uv run python scripts/plot_discount_tables.py .hydra_runs/<date>
```

Cross-configuration comparisons (e.g. local vs global accuracy, or the gap
between two reward signals) can be assembled from the scalar metrics these
scripts expose; see [docs/sweeps.md](docs/sweeps.md) for the declarative
sweep-plotting configuration.

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
