# coding: utf-8
###
 # @file   dataset.py
 # @author John stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Dataset wrappers/helpers.
###

import torch, torchvision, random
import torchvision.transforms as T
import numpy as np
from .dataset_utils import (
  get_default_root,
  draw_indices,
  pathological_classes_per_worker,
  pathological_grouped_classes,
  pathological_shards_per_worker,
)


def _is_femnist(dataset_name):
  return str(dataset_name).lower() == "femnist"


def _build_underlying_train(dataset_name):
  """Return (torch Dataset, targets tensor) for the training pool.

  Branches on dataset_name so FEMNIST can use its custom loader instead of
  torchvision.datasets. For FEMNIST in pool mode we use the full pooled
  training set (no writer-level holdout): the global-test holdout is now
  carved out uniformly inside `make_train_validation_test_datasets`.
  """
  if _is_femnist(dataset_name):
    from .femnist import build_femnist_pool_dataset
    train_dataset = build_femnist_pool_dataset()
    return train_dataset, train_dataset.targets
  dataset = getattr(torchvision.datasets, dict_names[dataset_name])(
    root=get_default_root(), train=True, download=True, transform=transforms[dataset_name][0]
  )
  targets = dataset.targets
  if isinstance(targets, list):
    targets = torch.FloatTensor(targets)
  return dataset, targets

# ---------------------------------------------------------------------------- #
# Collection of default transforms
transforms_horizontalflip = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
# Transforms from "A Little is Enough" (https://github.com/moranant/attacking_distributed_learning)
transforms_mnist = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
# Transforms from https://github.com/kuangliu/pytorch-cifar
transforms_cifar = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Per-dataset image transformations (automatically completed, see 'Dataset._get_datasets')
transforms = {
  "mnist":        (transforms_mnist, transforms_mnist),
  "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
  "cifar10":      (transform_train, transform_test),
  "cifar100":     (transforms_cifar, transforms_cifar),
  "imagenet":     (transforms_horizontalflip, transforms_horizontalflip) }

#JS: Dataset names in pytorch
dict_names = {
  "mnist":        "MNIST",
  "fashionmnist": "FashionMNIST",
  "emnist":       "EMNIST",
  "cifar10":      "CIFAR10",
  "cifar100":     "CIFAR100",
  "imagenet":     "ImageNet"}


def _partition_worker_indices(
    targets,
    *,
    honest_workers,
    numb_labels,
    heterogeneity=False,
    distinct_datasets=False,
    gamma_similarity=None,
    alpha_dirichlet=None,
    nb_datapoints=None,
    partition_method=None,
    partition_style=None,
    classes_per_worker=None,
    nb_shards=None,
    shards_per_worker=None,
    nb_groups=None,
    classes_per_group=None,
    group_overlap=0,
):
  """Return `{worker_id: [sample_index, ...]}` over `targets` according to the
  selected partitioning scheme.

  This is the pure partitioning logic split off from `Dataset.__init__` so that
  the caller can drive both the train and local-test DataLoader construction
  from the same partition.
  """
  numb_samples = len(targets)

  if heterogeneity:
    ordered_indices = []
    for label in range(numb_labels):
      label_indices = (targets == label).nonzero().tolist()
      label_indices = [item for sublist in label_indices for item in sublist]
      ordered_indices += label_indices
    splits = np.array_split(ordered_indices, honest_workers)
    return {worker_id: splits[worker_id].tolist() for worker_id in range(honest_workers)}

  if distinct_datasets and gamma_similarity is not None:
    numb_samples_iid = int(gamma_similarity * numb_samples)
    homogeneous_indices = list(range(numb_samples))
    random.shuffle(homogeneous_indices)
    homogeneous_indices = homogeneous_indices[:numb_samples_iid]
    homogeneous_set = set(homogeneous_indices)
    split_homogeneous = np.array_split(homogeneous_indices, honest_workers)

    ordered_heterogeneous = []
    for label in range(numb_labels):
      label_indices = (targets == label).nonzero().tolist()
      label_indices = [item for sublist in label_indices for item in sublist]
      ordered_heterogeneous += [i for i in label_indices if i not in homogeneous_set]
    split_heterogeneous = np.array_split(ordered_heterogeneous, honest_workers)
    return {
      worker_id: list(split_homogeneous[worker_id]) + list(split_heterogeneous[worker_id])
      for worker_id in range(honest_workers)
    }

  if distinct_datasets:
    sample_indices = list(range(numb_samples))
    random.shuffle(sample_indices)
    if nb_datapoints is None:
      splits = np.array_split(sample_indices, honest_workers)
      return {worker_id: list(splits[worker_id]) for worker_id in range(honest_workers)}
    return {
      worker_id: sample_indices[worker_id * nb_datapoints : (worker_id + 1) * nb_datapoints]
      for worker_id in range(honest_workers)
    }

  if partition_method == "pathological":
    if partition_style == "classes_per_worker":
      return pathological_classes_per_worker(
        targets, numb_labels, honest_workers, classes_per_worker
      )
    if partition_style == "shards_per_worker":
      return pathological_shards_per_worker(
        targets, numb_labels, honest_workers, shards_per_worker, nb_shards=nb_shards
      )
    if partition_style == "grouped_classes":
      return pathological_grouped_classes(
        targets, numb_labels, honest_workers, nb_groups, classes_per_group,
        overlap=group_overlap,
      )
    raise ValueError(
      f"Unknown pathological partition_style: {partition_style!r}. "
      "Expected 'classes_per_worker', 'shards_per_worker', or 'grouped_classes'."
    )

  if alpha_dirichlet is not None:
    indices_per_label = {}
    for label in range(numb_labels):
      label_indices = (targets == label).nonzero().tolist()
      indices_per_label[label] = [item for sublist in label_indices for item in sublist]
    samples_distribution = np.random.dirichlet(
      np.repeat(alpha_dirichlet, honest_workers), size=numb_labels
    )
    return draw_indices(samples_distribution, indices_per_label, honest_workers)

  raise ValueError(
    "No partitioning scheme selected: provide alpha_dirichlet, partition_method='pathological',"
    " distinct_datasets=True, or heterogeneity=True."
  )


def _uniform_train_test_split(indices, test_ratio, rng):
  """Uniform shuffle + split `indices` into (train_indices, test_indices).

  Guarantees at least 1 train sample if any indices were provided; if the
  partition is empty, returns two empty lists.
  """
  indices = list(indices)
  if not indices:
    return [], []
  rng.shuffle(indices)
  test_size = int(round(len(indices) * test_ratio))
  test_size = max(0, min(test_size, len(indices) - 1)) if len(indices) > 1 else 0
  if test_size == 0:
    return indices, []
  return indices[test_size:], indices[:test_size]


# ---------------------------------------------------------------------------- #
def make_train_validation_test_datasets(
    dataset, *, heterogeneity=False, numb_labels=None, distinct_datasets=False,
    gamma_similarity=None, alpha_dirichlet=None, nb_datapoints=None,
    honest_workers=None, train_batch=None, test_batch=None,
    global_test_ratio=0.1, local_test_ratio=0.2, split_seed=0,
    partition_method=None, partition_style=None,
    classes_per_worker=None, nb_shards=None, shards_per_worker=None,
    nb_groups=None, classes_per_group=None, group_overlap=0,
    dataset_mode=None, nb_writers_limit=None):
  """Build per-worker training + per-worker local test DataLoaders, plus a shared
  global test DataLoader.

  The official train pool is split in two stages:
    1. Uniformly hold out `global_test_ratio` of it as the global test set.
    2. Partition the remainder among honest workers using the selected
       heterogeneity scheme.
  Each worker's resulting partition is then split into local train / local test
  uniformly using `local_test_ratio`, so the local test mirrors the client's
  (skewed) distribution.

  The official test split is intentionally discarded for non-FEMNIST datasets;
  the global test set comes entirely from the train pool.

  Returns:
    (train_loaders: dict[int, DataLoader],
     local_test_loaders: dict[int, DataLoader],
     global_test_loader: DataLoader)
  """
  if dataset_mode == "writer_per_node":
    if not _is_femnist(dataset):
      raise ValueError(
        f"dataset_mode='writer_per_node' is only supported for FEMNIST (got dataset={dataset!r})"
      )
    from .femnist import load_femnist_writer_loaders
    return load_femnist_writer_loaders(
      nb_honest=honest_workers,
      train_batch=train_batch,
      test_batch=test_batch,
      global_test_ratio=global_test_ratio,
      local_test_ratio=local_test_ratio,
      split_seed=split_seed,
      writers_cap=nb_writers_limit,
    )

  full_train_dataset, full_targets = _build_underlying_train(dataset)

  # Stage 1: carve a uniform global test set out of the train pool.
  total = len(full_train_dataset)
  if total < 2:
    raise ValueError("Need at least 2 training samples to carve a global test split")
  rng = np.random.default_rng(split_seed)
  permuted = rng.permutation(total)
  global_test_size = int(round(total * global_test_ratio))
  global_test_size = min(max(global_test_size, 1), total - honest_workers)
  global_test_indices = permuted[:global_test_size].tolist()
  client_pool_indices = permuted[global_test_size:].tolist()

  # Stage 2: partition the remaining client pool across honest workers.
  client_pool_indices_arr = np.asarray(client_pool_indices)
  client_pool_targets = full_targets[client_pool_indices_arr] \
    if isinstance(full_targets, torch.Tensor) \
    else torch.as_tensor([full_targets[i] for i in client_pool_indices_arr])

  local_indices_per_worker = _partition_worker_indices(
    client_pool_targets,
    honest_workers=honest_workers,
    numb_labels=numb_labels,
    heterogeneity=heterogeneity,
    distinct_datasets=distinct_datasets,
    gamma_similarity=gamma_similarity,
    alpha_dirichlet=alpha_dirichlet,
    nb_datapoints=nb_datapoints,
    partition_method=partition_method,
    partition_style=partition_style,
    classes_per_worker=classes_per_worker,
    nb_shards=nb_shards,
    shards_per_worker=shards_per_worker,
    nb_groups=nb_groups,
    classes_per_group=classes_per_group,
    group_overlap=group_overlap,
  )

  # Stage 3: per-worker uniform local train/test split, then build loaders.
  train_loaders: dict[int, torch.utils.data.DataLoader] = {}
  local_test_loaders: dict[int, torch.utils.data.DataLoader] = {}
  for worker_id in range(honest_workers):
    local_pool = local_indices_per_worker.get(worker_id, [])
    # Map partition-local indices back to absolute indices in the full dataset.
    absolute_indices = [int(client_pool_indices_arr[i]) for i in local_pool]
    worker_rng = np.random.default_rng(split_seed + 1 + worker_id)
    train_idx, local_test_idx = _uniform_train_test_split(
      absolute_indices, local_test_ratio, worker_rng
    )
    train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
    train_loaders[worker_id] = torch.utils.data.DataLoader(
      train_subset, batch_size=train_batch, shuffle=True
    )
    local_test_subset = torch.utils.data.Subset(full_train_dataset, local_test_idx)
    local_test_loaders[worker_id] = torch.utils.data.DataLoader(
      local_test_subset, batch_size=test_batch, shuffle=False
    )

  global_test_subset = torch.utils.data.Subset(full_train_dataset, global_test_indices)
  global_test_loader = torch.utils.data.DataLoader(
    global_test_subset, batch_size=test_batch, shuffle=False
  )

  return train_loaders, local_test_loaders, global_test_loader
