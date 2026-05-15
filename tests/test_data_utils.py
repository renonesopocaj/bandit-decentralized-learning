import numpy as np
import torch

from banditdl.data.dataset_utils import (
    draw_indices,
    pathological_classes_per_worker,
    pathological_grouped_classes,
    pathological_shards_per_worker,
)


def test_draw_indices_advances_offsets_cumulatively():
    samples_distribution = np.array([[0.2, 0.3, 0.5]])
    indices_per_label = {0: list(range(10))}

    samples = draw_indices(samples_distribution, indices_per_label, nb_workers=3)

    assert samples == {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8, 9]}


def _toy_targets(samples_per_label=20, numb_labels=5):
    return torch.tensor([label for label in range(numb_labels) for _ in range(samples_per_label)])


def test_pathological_classes_per_worker_respects_class_budget():
    targets = _toy_targets(samples_per_label=20, numb_labels=5)
    rng = np.random.default_rng(0)

    worker_samples = pathological_classes_per_worker(
        targets, numb_labels=5, nb_workers=4, classes_per_worker=2, rng=rng
    )

    assert set(worker_samples) == {0, 1, 2, 3}
    for worker_id, indices in worker_samples.items():
        labels_seen = set(int(targets[i].item()) for i in indices)
        assert len(labels_seen) <= 2, f"worker {worker_id} saw {labels_seen}"
    all_indices = [i for indices in worker_samples.values() for i in indices]
    assert len(set(all_indices)) == len(all_indices), "samples must not be duplicated across workers"


def test_pathological_shards_per_worker_consumes_all_shards_by_default():
    targets = _toy_targets(samples_per_label=20, numb_labels=5)
    rng = np.random.default_rng(1)

    worker_samples = pathological_shards_per_worker(
        targets, numb_labels=5, nb_workers=4, shards_per_worker=2, rng=rng
    )

    all_indices = sorted(i for indices in worker_samples.values() for i in indices)
    assert all_indices == list(range(20 * 5)), "shards_per_worker default should cover the whole dataset"
    for indices in worker_samples.values():
        assert len(indices) == 20 * 5 // 4


def test_pathological_shards_per_worker_rejects_too_few_shards():
    targets = _toy_targets(samples_per_label=20, numb_labels=5)
    try:
        pathological_shards_per_worker(
            targets, numb_labels=5, nb_workers=4, shards_per_worker=2, nb_shards=4
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when nb_shards < nb_workers * shards_per_worker")


def test_pathological_grouped_classes_disjoint_keeps_labels_within_group():
    targets = _toy_targets(samples_per_label=20, numb_labels=10)
    rng = np.random.default_rng(0)

    worker_samples = pathological_grouped_classes(
        targets, numb_labels=10, nb_workers=10, nb_groups=5, classes_per_group=2, rng=rng
    )

    expected_group_labels = [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}]
    for worker_id, indices in worker_samples.items():
        group = worker_id // 2
        labels_seen = set(int(targets[i].item()) for i in indices)
        assert labels_seen <= expected_group_labels[group], (
            f"worker {worker_id} (group {group}) saw {labels_seen}, "
            f"expected subset of {expected_group_labels[group]}"
        )
    all_indices = [i for indices in worker_samples.values() for i in indices]
    assert len(set(all_indices)) == len(all_indices)
    assert len(all_indices) == 20 * 10, "disjoint case should consume all samples"


def test_pathological_grouped_classes_distributes_remainder_to_first_groups():
    targets = _toy_targets(samples_per_label=20, numb_labels=10)
    rng = np.random.default_rng(0)

    worker_samples = pathological_grouped_classes(
        targets, numb_labels=10, nb_workers=7, nb_groups=3, classes_per_group=2, rng=rng
    )

    expected_group_sizes = [3, 2, 2]
    worker_groups = []
    cursor = 0
    for size in expected_group_sizes:
        worker_groups.append(list(range(cursor, cursor + size)))
        cursor += size
    group_labels = [{0, 1}, {2, 3}, {4, 5}]
    for g, members in enumerate(worker_groups):
        for worker_id in members:
            labels_seen = set(int(targets[i].item()) for i in worker_samples[worker_id])
            assert labels_seen <= group_labels[g]


def test_pathological_grouped_classes_overlap_shares_labels_across_groups():
    targets = _toy_targets(samples_per_label=20, numb_labels=10)
    rng = np.random.default_rng(0)

    worker_samples = pathological_grouped_classes(
        targets, numb_labels=10, nb_workers=6, nb_groups=3, classes_per_group=3,
        overlap=1, rng=rng,
    )

    expected_group_labels = [{0, 1, 2}, {2, 3, 4}, {4, 5, 6}]
    for worker_id, indices in worker_samples.items():
        group = worker_id // 2
        labels_seen = set(int(targets[i].item()) for i in indices)
        assert labels_seen <= expected_group_labels[group]
    all_indices = [i for indices in worker_samples.values() for i in indices]
    assert len(set(all_indices)) == len(all_indices), "samples must not be duplicated"


def test_pathological_grouped_classes_rejects_too_many_labels():
    targets = _toy_targets(samples_per_label=20, numb_labels=10)
    try:
        pathological_grouped_classes(
            targets, numb_labels=10, nb_workers=6, nb_groups=3, classes_per_group=4
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when nb_groups * classes_per_group > numb_labels")
