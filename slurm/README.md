# Running BanditDL on EPFL Izar (Slurm)

Two scripts:

| File | Where to run | Purpose |
| --- | --- | --- |
| `slurm/setup.sh` | **Login node** | One-time: install `uv`, sync deps, optionally pre-download FEMNIST. |
| `slurm/sbatch_banditdl.sh` | **Login node** (via `sbatch`) | Submits one training run. Takes Hydra overrides as args. |

## 1. One-time bootstrap (login node)

```bash
# CIFAR / MNIST only — no FEMNIST download yet
bash slurm/setup.sh

# CIFAR / MNIST + pre-download FEMNIST (~700 MB into ~/.cache/huggingface/datasets)
bash slurm/setup.sh --femnist
```

The bootstrap loads `gcc` + `cuda`, installs `uv` to `~/.local/bin` if missing, and runs `uv sync` to populate `.venv`. Re-run safely; each step is idempotent.

If your `$HOME` quota is tight, redirect the HuggingFace cache once and re-run:

```bash
export HF_DATASETS_CACHE=$SCRATCH/banditdl/hf-cache
bash slurm/setup.sh --femnist
```

You will also need to `export HF_DATASETS_CACHE=...` in any sbatch invocation — the easiest path is to put that `export` in `~/.bashrc` (Izar uses `bash -l` for jobs, so login-shell init files are sourced).

## 2. Submitting a single run

```bash
# CIFAR-10, epsilon-greedy bandit, seed 0
sbatch slurm/sbatch_banditdl.sh dataset=cifar10 sampler=bandit seed=0

# FEMNIST writer-per-node, default sampler, 30 nodes
sbatch slurm/sbatch_banditdl.sh dataset=femnist topology.nodes=30

# CIFAR-10 with the grouped clustering partition
sbatch slurm/sbatch_banditdl.sh \
    dataset=cifar10 sampler=bandit heterogeneity=grouped_5x2 topology.nodes=30 seed=0
```

All positional args are forwarded to `uv run -m banditdl`. `device=cuda` is added automatically unless you provide your own `device=` override.

Override sbatch directives via the CLI:

```bash
sbatch --time=04:00:00 --job-name=fast_cifar slurm/sbatch_banditdl.sh dataset=cifar10
```

## 3. Sweeps

Two patterns work; pick by sweep size.

**Pattern A — one sbatch per combination (best for >1h runs, parallelizes across GPUs):**

```bash
for seed in 0 1 2; do
  for sampler in uniform bandit exp3; do
    sbatch slurm/sbatch_banditdl.sh \
        dataset=cifar10 topology=dynamic sampler=$sampler seed=$seed
  done
done
```

Each combination becomes its own job and runs in parallel as cluster capacity allows. Output dirs are independent (Hydra writes `.hydra_runs/<date>/<time>/`).

**Pattern B — single sbatch with Hydra multirun (best for sweeps that fit in one walltime):**

```bash
sbatch --time=24:00:00 slurm/sbatch_banditdl.sh -m \
    dataset=cifar10 topology=dynamic \
    sampler=uniform,bandit,exp3 seed=0,1,2
```

Hydra's `-m` flag enumerates the Cartesian product **sequentially** inside one job. Output goes to `.hydra_multirun/<date>/<time>/<idx>/`. Use this when total walltime is manageable; otherwise prefer Pattern A.

**Pattern C — Optuna grid sweep (for the dedicated sweep entry point):**

```bash
sbatch --time=24:00:00 slurm/sbatch_banditdl.sh \
    --module banditdl.experiments.sweep optuna=sweep
```

…but note that `sbatch_banditdl.sh` calls `uv run -m banditdl`, so for `banditdl.experiments.sweep` you'd want a tiny variant. If you need this, copy `sbatch_banditdl.sh` to `sbatch_sweep.sh` and change the final `srun` line to:

```bash
srun --cpu-bind=cores uv run python -m banditdl.experiments.sweep "$@"
```

## 4. Output

- **Job logs**: `job_output/banditdl_<jobid>.txt` (created by `--output=job_output/...`).
- **Experiment artifacts**: `.hydra_runs/<date>/<time>/` (single runs) or `.hydra_multirun/<date>/<time>/<idx>/` (Pattern B). Inside each: `results/` (numpy arrays), `plots/` (auto-generated PNGs), `.hydra/` (resolved config).

Both `job_output/` and the Hydra dirs are gitignored.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ERROR: 'uv' not on PATH` | Forgot the bootstrap | `bash slurm/setup.sh` on a login node |
| `CUDA out of memory` | Too-large batch on V100 | Override: `optimization.batch_size=16` |
| `Could not load 'override'` | Missing `conf/override.yaml` | Create one per the root README, or pass overrides via CLI |
| FEMNIST stalls on first run | First call downloads the HF dataset | Pre-download with `bash slurm/setup.sh --femnist` |
| Job dies at 12h walltime | Default time limit | Override: `sbatch --time=24:00:00 slurm/sbatch_banditdl.sh ...` |

## 6. Defaults at a glance

The sbatch directives in `sbatch_banditdl.sh` mirror your `test.sh` conventions:

- `--partition=gpu --gres=gpu:1`
- `--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G`
- `--time=12:00:00` (shorter than your 48h default — CIFAR 2000 rounds finishes in well under that on a V100)
- `--mail-user=mattea.busato@epfl.ch` and `--mail-type=END,FAIL`
- `srun --cpu-bind=cores`

All overridable via the `sbatch` CLI.
