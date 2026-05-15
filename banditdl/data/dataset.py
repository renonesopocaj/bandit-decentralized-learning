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
  """Return (torch Dataset, targets tensor) for the training split.

  Branches on dataset_name so FEMNIST can use its custom loader instead of
  torchvision.datasets.
  """
  if _is_femnist(dataset_name):
    from .femnist import build_femnist_pool_dataset
    train_dataset, _ = build_femnist_pool_dataset()
    return train_dataset, train_dataset.targets
  dataset = getattr(torchvision.datasets, dict_names[dataset_name])(
    root=get_default_root(), train=True, download=True, transform=transforms[dataset_name][0]
  )
  targets = dataset.targets
  if isinstance(targets, list):
    targets = torch.FloatTensor(targets)
  return dataset, targets


def _build_underlying_eval(dataset_name):
  """Return a torch Dataset for the eval (official test) split."""
  if _is_femnist(dataset_name):
    from .femnist import build_femnist_pool_dataset
    _, eval_dataset = build_femnist_pool_dataset()
    return eval_dataset
  return getattr(torchvision.datasets, dict_names[dataset_name])(
    root=get_default_root(), train=False, download=True, transform=transforms[dataset_name][1]
  )

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

# ---------------------------------------------------------------------------- #
# Dataset wrapper class
class Dataset:
  """ Dataset wrapper class."""

  def __init__(self, dataset_name, heterogeneity=False, numb_labels=None, distinct_datasets=False,
               gamma_similarity=None, alpha_dirichlet=None, nb_datapoints=None, honest_workers=None, batch_size=None,
               partition_method=None, partition_style=None,
               classes_per_worker=None, nb_shards=None, shards_per_worker=None,
               nb_groups=None, classes_per_group=None, group_overlap=0):
    """ Training Dataset builder constructor.
    Args:
      dataset_name          Dataset string name
      heterogeneity         Boolean that is true in heterogeneous setting
      numb_labels           Number of labels of the dataset in question
      distinct_datasets     Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
      gamma_similarity      Float for distributing the datasets among honest workers
      alpha_dirichlet       Value of parameter alpha for dirichlet distribution
      nb_datapoints         Number of datapoints per honest worker in case of distinct datasets
      honest_workers        Number of honest workers in the system
      batch_size            Batch size used during the training or testing
      partition_method      Optional partition method, e.g. "pathological"
      partition_style       Pathological sub-style: "classes_per_worker" or "shards_per_worker"
      classes_per_worker    Number of classes assigned to each worker (classes_per_worker style)
      nb_shards             Total number of shards (shards_per_worker style; defaults to nb_workers * shards_per_worker)
      shards_per_worker     Shards assigned to each worker (shards_per_worker style)
      nb_groups             Number of node groups (grouped_classes style)
      classes_per_group     Number of labels assigned to each group (grouped_classes style)
      group_overlap         Labels shared between adjacent groups (grouped_classes style, default 0)
    """

    #JS: Load the initial training dataset (branches on dataset_name; non-torchvision datasets
    # like FEMNIST use a custom loader).
    dataset, targets = _build_underlying_train(dataset_name)

    #JS: extreme heterogeneity setting while training
    if heterogeneity:
      labels = range(numb_labels)
      ordered_indices = []
      for label in labels:
        label_indices = (targets == label).nonzero().tolist()
        label_indices = [item for sublist in label_indices for item in sublist]
        ordered_indices += label_indices

      self.dataset_dict = {}
      split_indices = np.array_split(ordered_indices, honest_workers)
      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id].tolist())
        dataset_worker = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = dataset_worker


    #JS: distinct datasets for honest workers with gamma similarity
    elif distinct_datasets and gamma_similarity is not None:
      numb_samples = len(targets)
      numb_samples_iid = int(gamma_similarity * numb_samples)

      #JS: Sample gamma_similarity % of the dataset, and build homogeneous dataset
      homogeneous_dataset, _ = torch.utils.data.random_split(dataset, [numb_samples_iid, numb_samples - numb_samples_iid])

      #JS: Split the indices of the homogeneous dataset onto the honest workers
      split_indices_homogeneous = np.array_split(homogeneous_dataset.indices, honest_workers)

      #JS: Rearrange the entire dataset by sorted labels
      labels = range(numb_labels)
      ordered_indices = []
      for label in labels:
        label_indices = (targets == label).nonzero().tolist()
        label_indices = [item for sublist in label_indices for item in sublist]
        ordered_indices += label_indices
      #JS: split the (sorted) heterogeneous indices equally among the honest workers
      indices_heterogeneous = [index for index in ordered_indices if index not in homogeneous_dataset.indices]
      split_indices_heterogeneous = np.array_split(indices_heterogeneous, honest_workers)

      self.dataset_dict = {}
      for worker_id in range(honest_workers):
        homogeneous_dataset_worker = torch.utils.data.Subset(dataset, split_indices_homogeneous[worker_id])
        heterogeneous_dataset_worker = torch.utils.data.Subset(dataset, split_indices_heterogeneous[worker_id])
        concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker, heterogeneous_dataset_worker])
        dataset_worker = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size, shuffle=True)
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = dataset_worker


    #JS: distinct datasets for honest workers, homogeneous setting
    elif distinct_datasets:
      numb_samples = len(targets)
      sample_indices = list(range(numb_samples))
      random.shuffle(sample_indices)

      self.dataset_dict = {}
      if nb_datapoints is None:
        #JS: split the whole dataset equally among the honest workers
        split_indices = np.array_split(sample_indices, honest_workers)
      else:
        #JS: give every honest worker nb_datapoints samples
        split_indices = [sample_indices[i:i + nb_datapoints] for i in range(0, nb_datapoints*honest_workers, nb_datapoints)]

      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id])
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)


    #JS: pathological non-IID partitioning (FedAvg-style)
    elif partition_method == "pathological":
      if partition_style == "classes_per_worker":
        worker_samples = pathological_classes_per_worker(
          targets, numb_labels, honest_workers, classes_per_worker
        )
      elif partition_style == "shards_per_worker":
        worker_samples = pathological_shards_per_worker(
          targets, numb_labels, honest_workers, shards_per_worker, nb_shards=nb_shards
        )
      elif partition_style == "grouped_classes":
        worker_samples = pathological_grouped_classes(
          targets, numb_labels, honest_workers, nb_groups, classes_per_group,
          overlap=group_overlap,
        )
      else:
        raise ValueError(
          f"Unknown pathological partition_style: {partition_style!r}. "
          "Expected 'classes_per_worker', 'shards_per_worker', or 'grouped_classes'."
        )

      self.dataset_dict = {}
      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, worker_samples[worker_id])
        self.dataset_dict[worker_id] = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)

    #JS: distribute data among honest workers using Dirichlet distribution
    elif alpha_dirichlet is not None:

      #JS: store in indices_per_label the list of indices of each label (0 then 1 then 2 ...)
      indices_per_label = dict()
      for label in range(numb_labels):
        label_indices = (targets == label).nonzero().tolist()
        label_indices = [item for sublist in label_indices for item in sublist]
        indices_per_label[label] = label_indices

      #JS: compute number of samples of each worker for each class, using a Dirichlet distribution of parameter alpha_dirichlet
      samples_distribution = np.random.dirichlet(np.repeat(alpha_dirichlet, honest_workers), size=numb_labels)
      #JS: get the indices of the samples belonging to each worker (stored in dict worker_samples)
      worker_samples = draw_indices(samples_distribution, indices_per_label, honest_workers)

      self.dataset_dict = {}
      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, worker_samples[worker_id])
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------- #
def make_train_validation_test_datasets(dataset, heterogeneity=False, numb_labels=None, distinct_datasets=False, gamma_similarity=None, alpha_dirichlet=None,
    nb_datapoints=None, honest_workers=None, train_batch=None, test_batch=None, validation_ratio=0.5, split_seed=0,
    partition_method=None, partition_style=None, classes_per_worker=None, nb_shards=None, shards_per_worker=None,
    nb_groups=None, classes_per_group=None, group_overlap=0,
    dataset_mode=None, nb_writers_limit=None):
  """ Helper to make new instance of train, validation and test datasets.
  Args:
    dataset             Case-sensitive dataset name
    heterogeneity       Boolean that is true in heterogeneous setting
    numb_labels         Number of labels of dataset
    distinct_datasets   Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
    gamma_similarity    Float for distributing the datasets among honest workers
    alpha_dirichlet     Value of parameter alpha for dirichlet distribution
    nb_datapoints       Number of datapoints per honest worker in case of distinct datasets
    honest_workers      Number of honest workers in the system
    train_batch         Training batch size
    test_batch          Validation/test batch size
    validation_ratio    Fraction of the official test split to use as validation split
    split_seed          Seed used for deterministic validation/test split
    partition_method    Optional partition method, e.g. "pathological"
    partition_style     Pathological sub-style: "classes_per_worker" or "shards_per_worker"
    classes_per_worker  Number of classes per worker (classes_per_worker style)
    nb_shards           Total number of shards (shards_per_worker style)
    shards_per_worker   Shards assigned to each worker (shards_per_worker style)
  Returns:
    (Dictionary of training datasets for honest workers, data loader for validation dataset, data loader for held-out test dataset)
  """
  # FEMNIST writer-per-node short-circuit: bypass the synthetic partitioning entirely;
  # each honest worker gets one FEMNIST writer's DataLoader.
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
      validation_ratio=validation_ratio,
      split_seed=split_seed,
      writers_cap=nb_writers_limit,
    )

  # Make the training dataset (pool path: FEMNIST falls through here and gets partitioned
  # by the existing heterogeneity machinery, like torchvision datasets).
  trainset = Dataset(dataset, heterogeneity=heterogeneity, numb_labels=numb_labels,
                     distinct_datasets=distinct_datasets, gamma_similarity=gamma_similarity, alpha_dirichlet=alpha_dirichlet,
                     nb_datapoints=nb_datapoints, honest_workers=honest_workers, batch_size=train_batch,
                     partition_method=partition_method, partition_style=partition_style,
                     classes_per_worker=classes_per_worker, nb_shards=nb_shards, shards_per_worker=shards_per_worker,
                     nb_groups=nb_groups, classes_per_group=classes_per_group, group_overlap=group_overlap)

  # Build validation/test splits from the official train=False split (FEMNIST uses its own loader).
  dataset_eval_full = _build_underlying_eval(dataset)
  total_eval_samples = len(dataset_eval_full)
  if total_eval_samples < 2:
    raise ValueError("Need at least 2 samples to create validation/test splits")
  validation_size = int(round(total_eval_samples * validation_ratio))
  validation_size = min(max(validation_size, 1), total_eval_samples - 1)
  test_size = total_eval_samples - validation_size
  generator = torch.Generator().manual_seed(split_seed)
  dataset_validation, dataset_test = torch.utils.data.random_split(dataset_eval_full, [validation_size, test_size], generator=generator)
  data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=test_batch, shuffle=False)
  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch, shuffle=False)

  # Return the data loaders
  return trainset.dataset_dict, data_loader_validation, data_loader_test


def make_train_test_datasets(dataset, heterogeneity=False, numb_labels=None, distinct_datasets=False, gamma_similarity=None, alpha_dirichlet=None,
    nb_datapoints=None, honest_workers=None, train_batch=None, test_batch=None, validation_ratio=0.5, split_seed=0,
    partition_method=None, partition_style=None, classes_per_worker=None, nb_shards=None, shards_per_worker=None,
    nb_groups=None, classes_per_group=None, group_overlap=0,
    dataset_mode=None, nb_writers_limit=None):
  """ Backward-compatible helper returning train datasets and validation loader.
  Args:
    dataset             Case-sensitive dataset name
    heterogeneity       Boolean that is true in heterogeneous setting
    numb_labels         Number of labels of dataset
    distinct_datasets   Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
    gamma_similarity    Float for distributing the datasets among honest workers
    alpha_dirichlet     Value of parameter alpha for dirichlet distribution
    nb_datapoints       Number of datapoints per honest worker in case of distinct datasets
    honest_workers      Number of honest workers in the system
    train_batch         Training batch size
    test_batch          Validation/test batch size
    validation_ratio    Fraction of the official test split to use as validation split
    split_seed          Seed used for deterministic validation/test split
  Returns:
    (Dictionary of training datasets for honest workers, data loader for validation dataset)
  """
  trainset, data_loader_validation, _ = make_train_validation_test_datasets(
    dataset=dataset, heterogeneity=heterogeneity, numb_labels=numb_labels, distinct_datasets=distinct_datasets,
    gamma_similarity=gamma_similarity, alpha_dirichlet=alpha_dirichlet, nb_datapoints=nb_datapoints,
    honest_workers=honest_workers, train_batch=train_batch, test_batch=test_batch,
    validation_ratio=validation_ratio, split_seed=split_seed,
    partition_method=partition_method, partition_style=partition_style,
    classes_per_worker=classes_per_worker, nb_shards=nb_shards, shards_per_worker=shards_per_worker,
    nb_groups=nb_groups, classes_per_group=classes_per_group, group_overlap=group_overlap,
    dataset_mode=dataset_mode, nb_writers_limit=nb_writers_limit)
  return trainset, data_loader_validation
