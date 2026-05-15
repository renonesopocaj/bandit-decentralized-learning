"""Dataset utility functions."""

import pathlib
import numpy as np


def get_default_root():
    """Lazy-initialize and return the default dataset root directory path."""
    default_root = pathlib.Path(__file__).parent.parent / "datasets" / "cache"
    default_root.mkdir(parents=True, exist_ok=True)
    return default_root


def draw_indices(samples_distribution, indices_per_label, nb_workers):
    """Return the indices of the training datapoints selected for each honest worker.

    Used in case of Dirichlet distribution.
    """
    worker_samples = {worker: [] for worker in range(nb_workers)}

    for label, label_distribution in enumerate(samples_distribution):
        last_sample = 0
        number_samples_label = len(indices_per_label[label])
        for worker, worker_proportion in enumerate(label_distribution):
            samples_for_worker = int(worker_proportion * number_samples_label)
            worker_samples[worker].extend(
                indices_per_label[label][last_sample:last_sample + samples_for_worker]
            )
            last_sample += samples_for_worker

    return worker_samples


def _indices_per_label(targets, numb_labels):
    """Return a {label: [sample_index, ...]} dict over a torch/numpy targets tensor."""
    indices_per_label = {}
    for label in range(numb_labels):
        label_indices = (targets == label).nonzero().tolist()
        indices_per_label[label] = [item for sublist in label_indices for item in sublist]
    return indices_per_label


def pathological_classes_per_worker(targets, numb_labels, nb_workers, classes_per_worker, rng=None):
    """Pathological non-IID partition: each worker is assigned `classes_per_worker` random labels.

    Each class's samples are split evenly among the workers that drew it. Returns the same
    `{worker_id: [sample_index, ...]}` shape as `draw_indices`.
    """
    if classes_per_worker < 1:
        raise ValueError("classes_per_worker must be >= 1")
    if classes_per_worker > numb_labels:
        raise ValueError(
            f"classes_per_worker ({classes_per_worker}) cannot exceed numb_labels ({numb_labels})"
        )
    if rng is None:
        rng = np.random

    indices_per_label = _indices_per_label(targets, numb_labels)

    classes_assigned = [
        list(rng.choice(numb_labels, size=classes_per_worker, replace=False))
        for _ in range(nb_workers)
    ]
    workers_per_class = {label: [] for label in range(numb_labels)}
    for worker_id, labels in enumerate(classes_assigned):
        for label in labels:
            workers_per_class[int(label)].append(worker_id)

    worker_samples = {worker_id: [] for worker_id in range(nb_workers)}
    for label, owners in workers_per_class.items():
        if not owners:
            continue
        label_indices = list(indices_per_label[label])
        rng.shuffle(label_indices)
        chunks = np.array_split(label_indices, len(owners))
        for owner, chunk in zip(owners, chunks):
            worker_samples[owner].extend(int(i) for i in chunk)

    return worker_samples


def pathological_grouped_classes(
    targets, numb_labels, nb_workers, nb_groups, classes_per_group, overlap=0, rng=None
):
    """Grouped pathological partition for studying cluster formation.

    Workers are split into `nb_groups` consecutive groups (first `nb_workers % nb_groups`
    groups receive one extra worker). Group `g` is assigned the label range
    `[g * stride, g * stride + classes_per_group)` where `stride = classes_per_group - overlap`,
    so adjacent groups share `overlap` labels. For each label, samples are split evenly
    across all groups that own it; within a group, the resulting pool is shuffled and split
    evenly across the group's workers.

    Returns the same `{worker_id: [sample_index, ...]}` shape as `draw_indices`.
    """
    if nb_groups < 1:
        raise ValueError("nb_groups must be >= 1")
    if classes_per_group < 1:
        raise ValueError("classes_per_group must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= classes_per_group:
        raise ValueError("overlap must be < classes_per_group")
    if nb_workers < nb_groups:
        raise ValueError(
            f"nb_workers ({nb_workers}) must be >= nb_groups ({nb_groups})"
        )

    stride = classes_per_group - overlap
    last_label = (nb_groups - 1) * stride + classes_per_group
    if last_label > numb_labels:
        raise ValueError(
            f"label range [0, {last_label}) does not fit in numb_labels ({numb_labels}). "
            f"Reduce nb_groups, classes_per_group, or increase overlap."
        )
    if rng is None:
        rng = np.random

    group_labels = [
        list(range(g * stride, g * stride + classes_per_group))
        for g in range(nb_groups)
    ]
    workers_per_group = [nb_workers // nb_groups] * nb_groups
    for i in range(nb_workers % nb_groups):
        workers_per_group[i] += 1

    group_workers = []
    cursor = 0
    for size in workers_per_group:
        group_workers.append(list(range(cursor, cursor + size)))
        cursor += size

    indices_per_label = _indices_per_label(targets, numb_labels)
    owners_per_label = {label: [] for label in range(numb_labels)}
    for g, labels in enumerate(group_labels):
        for label in labels:
            owners_per_label[label].append(g)

    group_pool = {g: [] for g in range(nb_groups)}
    for label, owners in owners_per_label.items():
        if not owners:
            continue
        label_indices = list(indices_per_label[label])
        rng.shuffle(label_indices)
        chunks = np.array_split(label_indices, len(owners))
        for owner, chunk in zip(owners, chunks):
            group_pool[owner].extend(int(i) for i in chunk)

    worker_samples = {worker_id: [] for worker_id in range(nb_workers)}
    for g in range(nb_groups):
        pool = group_pool[g]
        rng.shuffle(pool)
        chunks = np.array_split(pool, len(group_workers[g]))
        for worker_id, chunk in zip(group_workers[g], chunks):
            worker_samples[worker_id].extend(int(i) for i in chunk)
    return worker_samples


def pathological_shards_per_worker(
    targets, numb_labels, nb_workers, shards_per_worker, nb_shards=None, rng=None
):
    """Pathological non-IID partition à la McMahan 2017: sort by label, cut into `nb_shards`
    shards, each worker draws `shards_per_worker` distinct shards.

    If `nb_shards` is None it defaults to `nb_workers * shards_per_worker` so that the shards
    are consumed exactly. Returns the same `{worker_id: [sample_index, ...]}` shape.
    """
    if shards_per_worker < 1:
        raise ValueError("shards_per_worker must be >= 1")
    if nb_shards is None:
        nb_shards = nb_workers * shards_per_worker
    if nb_shards < nb_workers * shards_per_worker:
        raise ValueError(
            f"nb_shards ({nb_shards}) must be >= nb_workers * shards_per_worker "
            f"({nb_workers * shards_per_worker})"
        )
    if rng is None:
        rng = np.random

    indices_per_label = _indices_per_label(targets, numb_labels)
    ordered_indices = []
    for label in range(numb_labels):
        ordered_indices.extend(indices_per_label[label])

    shards = np.array_split(ordered_indices, nb_shards)
    shard_ids = np.arange(nb_shards)
    rng.shuffle(shard_ids)

    worker_samples = {worker_id: [] for worker_id in range(nb_workers)}
    cursor = 0
    for worker_id in range(nb_workers):
        for _ in range(shards_per_worker):
            shard = shards[shard_ids[cursor]]
            worker_samples[worker_id].extend(int(i) for i in shard)
            cursor += 1
    return worker_samples
