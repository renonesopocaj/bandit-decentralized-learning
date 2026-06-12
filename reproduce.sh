#!/usr/bin/env bash
# Reproduce the experiments reported in
#   "Impact of Adaptive Neighbor Sampling on Optimization in Decentralized
#    Federated Learning" (EPFL OptML mini-project).
#
# Every experiment is a plain local run of `uv run -m banditdl <hydra overrides>`.
# Each invocation writes a self-contained run folder under
#   .hydra_runs/<date>/<time>_local/
# containing results/ (metric .npy arrays + audit.json) and plots/.
# The figure/table generators in scripts/ then read those run folders
# (see the "Generate figures and tables" section of the README).
#
# Usage:
#   bash reproduce.sh <experiment>
#
# Experiments:
#   femnist_main        Main FEMNIST study: 3 cluster profiles x all samplers x
#                       2 rewards  (Fig. 1 Pareto, Fig. 5-7 heatmaps/curves,
#                       Table I, Appendix C).
#   cifar_discount      CIFAR-10 discount ablation: {cucb,cts} x gamma in
#                       {1.0,0.99,0.95,0.9} x {cosine,model-distance} rewards
#                       (Tables III-IV, Fig. 8-10, Appendix D).
#   dirichlet           Dirichlet heterogeneity sweep on FEMNIST and CIFAR-10
#                       (alpha in {0.5,1,10}), all samplers (Fig. 3-4, Table I).
#   smoke               Tiny 5-round MNIST run to check the install works.
#
# Notes:
#   * All studies use n = 30 honest nodes, mean aggregation, 1 local SGD step.
#   * Hyperparameters (lr, batch size, momentum, weight decay, rounds) come from
#     conf/optimization/opt_{mnist,cifar10,femnist}.yaml and match the report.
#   * Override SEEDS / ROUNDS below to shrink a run while developing.
#   * These grids are large (hundreds of node-rounds each). They were originally
#     run on a compute cluster; a single config takes minutes (MNIST/FEMNIST) to
#     a few hours (CIFAR-10, 500-2000 rounds) on one CPU/GPU.

set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

run() {  # run <hydra overrides...>
  echo "[reproduce] uv run -m banditdl $*"
  uv run -m banditdl "$@"
}

NODES=30
SEEDS=(0 1 2)

femnist_main() {
  # Main study (Section IV, Appendix C). FEMNIST pooled, pathological clusters.
  # 3 cluster profiles, each with its matched sampling fraction (sampled ~=
  # cluster size + 1). num_seeds=3 averages seeds in-process per config.
  local ROUNDS=500 NUM_SEEDS=3
  # profile = clusters:classes_per_group:sampling
  local PROFILES=("3:15:0.3" "5:9:0.2" "10:4:0.1")
  local REWARDS=(parameter_distance update_cosine_similarity)
  for prof in "${PROFILES[@]}"; do
    IFS=: read -r clusters cpg sampling <<<"$prof"
    local BASE="dataset=femnist_pool optimization=opt_femnist \
      heterogeneity=pathological_5g_2c heterogeneity.clusters=$clusters \
      heterogeneity.classes_per_group=$cpg heterogeneity.group_overlap=0 \
      adversary=none topology.nodes=$NODES topology.sampling=$sampling \
      optimization.rounds=$ROUNDS evaluation.evaluation_delta=20 \
      num_seeds=$NUM_SEEDS seed=0"
    run $BASE sampler=uniform
    for reward in "${REWARDS[@]}"; do
      run $BASE sampler=epsilon_greedy   sampler.reward="$reward" sampler.params.epsilon=0.1
      run $BASE sampler=exp3             sampler.reward="$reward" sampler.params.gamma=auto
      run $BASE sampler=cucb             sampler.reward="$reward" sampler.params.exploration=1.0
      run $BASE sampler=cts              sampler.reward="$reward"
      run $BASE sampler=discounted_cucb  sampler.reward="$reward" sampler.params.exploration=1.0 sampler.params.gamma=0.99
      run $BASE sampler=discounted_cts   sampler.reward="$reward" sampler.params.gamma=0.99
    done
  done
}

cifar_discount() {
  # Discount ablation (Appendix D, Tables III-IV, Fig. 8-10). CIFAR-10,
  # pathological_5g_2c (5 clusters x 2 labels), single seed. opt_cifar10 uses
  # CrossEntropyLoss + lr decay, which cnn_cifar requires.
  local ROUNDS=500 SAMPLING=0.2 SEED=0
  local GAMMAS=(0.9 0.95 0.99)
  local BASE="dataset=cifar10 optimization=opt_cifar10 \
    heterogeneity=pathological_5g_2c adversary=none topology.nodes=$NODES \
    topology.sampling=$SAMPLING optimization.rounds=$ROUNDS \
    evaluation.evaluation_delta=20 seed=$SEED"
  for reward in cosine_similarity parameter_distance; do
    for family in cucb cts; do
      run $BASE sampler=$family sampler.reward="$reward"          # gamma=1.0 baseline
      for gamma in "${GAMMAS[@]}"; do
        run $BASE sampler=discounted_$family sampler.reward="$reward" sampler.params.gamma=$gamma
      done
    done
  done
}

dirichlet() {
  # Dirichlet heterogeneity sweep (Table I alpha rows, Fig. 3-4).
  local ROUNDS_FM=300 ROUNDS_CIFAR=500
  local SAMPLERS=(uniform epsilon_greedy exp3 cucb cts)
  for alpha in dirichlet_alpha0.5 dirichlet_alpha1 dirichlet_alpha10; do
    for sampler in "${SAMPLERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run dataset=femnist_pool optimization=opt_femnist sampler=$sampler \
          heterogeneity=$alpha topology.nodes=$NODES topology.sampling=0.2 \
          optimization.rounds=$ROUNDS_FM seed=$seed
        run dataset=cifar10 optimization=opt_cifar10 sampler=$sampler \
          heterogeneity=$alpha topology.nodes=$NODES topology.sampling=0.2 \
          optimization.rounds=$ROUNDS_CIFAR seed=$seed
      done
    done
  done
}

smoke() {
  run dataset=mnist sampler=uniform topology.nodes=10 optimization.rounds=5 \
    evaluation.evaluation_delta=5 seed=0
}

case "${1:-}" in
  femnist_main)   femnist_main ;;
  cifar_discount) cifar_discount ;;
  dirichlet)      dirichlet ;;
  smoke)          smoke ;;
  *)
    echo "usage: bash reproduce.sh <femnist_main|cifar_discount|dirichlet|smoke>" >&2
    exit 1
    ;;
esac
