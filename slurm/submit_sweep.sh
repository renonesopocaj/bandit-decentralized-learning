#!/bin/bash -l
# Submit a parameterized sweep on Izar by looping sbatch.
#
# Usage:
#   bash slurm/submit_sweep.sh <sweep_name>
#
# Sweep names defined below: cifar_dirichlet, femnist_pool_dirichlet,
# cifar_grouped, femnist_pool_grouped, femnist_writer.
#
# This is a thin wrapper around `sbatch slurm/sbatch_banditdl.sh ...`.
# Each combination becomes its own job. Edit the axis arrays below to adjust scope.

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SWEEP="${1:-}"
if [ -z "$SWEEP" ]; then
    echo "usage: $0 <cifar_dirichlet|femnist_pool_dirichlet|cifar_grouped|femnist_pool_grouped|femnist_writer>"
    exit 1
fi

# Shared axes
SAMPLERS=(uniform bandit exp3)
SAMPLINGS=(0.1 0.2)
SEEDS=(0 1 2)

submit_one() {
    local name="$1"; shift
    local override="$1"; shift
    echo "[submit] $name -- $override"
    sbatch --job-name="$name" --time=02:00:00 slurm/sbatch_banditdl.sh $override
}

count=0
case "$SWEEP" in
  cifar_dirichlet)
    NODES=30
    ROUNDS=500
    for alpha in alpha0.5 alpha1 alpha10.0; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="cifar_${alpha}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=cifar10 sampler=$sampler heterogeneity=$alpha topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  femnist_pool_dirichlet)
    NODES=30
    ROUNDS=300
    for alpha in alpha0.5 alpha1 alpha10.0; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="fmpool_${alpha}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=femnist dataset.mode=pool sampler=$sampler heterogeneity=$alpha topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  cifar_grouped)
    NODES=30
    ROUNDS=500
    for group in grouped_5x2 grouped_2x5; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="cifar_${group}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=cifar10 sampler=$sampler heterogeneity=$group topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  femnist_pool_grouped)
    NODES=30
    ROUNDS=300
    for group in grouped_5x2 grouped_2x5; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="fmpool_${group}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=femnist dataset.mode=pool sampler=$sampler heterogeneity=$group topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  femnist_writer)
    NODES=30
    ROUNDS=300
    for sampler in "${SAMPLERS[@]}"; do
      for sampling in "${SAMPLINGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          name="fmwriter_${sampler}_s${sampling}_seed${seed}"
          submit_one "$name" \
            "dataset=femnist sampler=$sampler topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
          count=$((count + 1))
        done
      done
    done
    ;;
  *)
    echo "unknown sweep: $SWEEP"
    exit 1
    ;;
esac

echo "[submit] queued $count jobs"
