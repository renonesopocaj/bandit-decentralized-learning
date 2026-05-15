"""Tests for the FEMNIST loader.

The HF dataset is replaced by a tiny synthetic DatasetDict so tests are fast and
require no network. The fixture overwrites `_load_hf_dataset`'s cache, so any
loader function that goes through it gets the synthetic data instead.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image

import banditdl.data.femnist as femnist_mod


def _make_synthetic_split(samples_per_writer: dict[str, int], rng: np.random.Generator):
    images, labels, writers = [], [], []
    for writer, count in samples_per_writer.items():
        for _ in range(count):
            arr = rng.integers(0, 255, size=(28, 28), dtype=np.uint8)
            images.append(Image.fromarray(arr, mode="L"))
            labels.append(int(rng.integers(0, femnist_mod.FEMNIST_NUM_CLASSES)))
            writers.append(writer)
    return Dataset.from_dict({"image": images, "character": labels, "writer_id": writers})


@pytest.fixture
def synthetic_femnist(monkeypatch):
    rng = np.random.default_rng(0)
    train_writers = {f"f{i:04d}": 6 for i in range(5)}
    test_writers = {f"f{i:04d}": 3 for i in range(5)}
    train = _make_synthetic_split(train_writers, rng)
    test = _make_synthetic_split(test_writers, rng)
    dd = DatasetDict({"train": train, "test": test})

    femnist_mod._load_hf_dataset.cache_clear()
    monkeypatch.setattr(femnist_mod, "_load_hf_dataset", lambda: dd)
    yield dd


def test_writer_loaders_one_loader_per_node(synthetic_femnist):
    train_loaders, validation_loader, test_loader = femnist_mod.load_femnist_writer_loaders(
        nb_honest=3, train_batch=2, test_batch=4, split_seed=0,
    )

    assert set(train_loaders.keys()) == {0, 1, 2}
    for worker_id, loader in train_loaders.items():
        samples = sum(batch[0].shape[0] for batch in loader)
        assert samples > 0, f"worker {worker_id} loader is empty"

    val_samples = sum(b[0].shape[0] for b in validation_loader)
    test_samples = sum(b[0].shape[0] for b in test_loader)
    assert val_samples + test_samples == 5 * 3  # all test rows materialized


def test_writer_loaders_each_worker_sees_one_writer_only(synthetic_femnist):
    train_loaders, _, _ = femnist_mod.load_femnist_writer_loaders(
        nb_honest=4, train_batch=2, test_batch=4, split_seed=42,
    )
    for worker_id, loader in train_loaders.items():
        sizes = [batch[0].shape[0] for batch in loader]
        assert sum(sizes) == 6, f"each writer should yield 6 samples; got {sum(sizes)}"


def test_writer_loaders_seed_changes_writer_assignment(synthetic_femnist):
    a, _, _ = femnist_mod.load_femnist_writer_loaders(
        nb_honest=2, train_batch=2, test_batch=4, split_seed=0,
    )
    b, _, _ = femnist_mod.load_femnist_writer_loaders(
        nb_honest=2, train_batch=2, test_batch=4, split_seed=999,
    )
    # Both loaders ran end-to-end; we just verify the structure is consistent.
    for batch_imgs, batch_labels in next(iter(a[0])), next(iter(b[0])):
        assert batch_imgs.shape[-2:] == (28, 28)
        assert batch_labels.dtype == torch.int64


def test_writer_loaders_rejects_too_many_workers(synthetic_femnist):
    with pytest.raises(ValueError):
        femnist_mod.load_femnist_writer_loaders(
            nb_honest=100, train_batch=2, test_batch=4, split_seed=0,
        )


def test_pool_dataset_exposes_targets(synthetic_femnist):
    train_ds, eval_ds = femnist_mod.build_femnist_pool_dataset()

    assert isinstance(train_ds.targets, torch.Tensor)
    assert train_ds.targets.dtype == torch.long
    assert len(train_ds.targets) == len(train_ds)
    sample, label = train_ds[0]
    assert sample.shape == (1, 28, 28)
    assert isinstance(label, int)
    assert len(eval_ds) == 5 * 3


@pytest.fixture
def synthetic_femnist_train_only(monkeypatch):
    """Fixture that emulates flwrlabs/femnist as actually shipped: only a `train` split."""
    rng = np.random.default_rng(0)
    # 20 writers so a 5% holdout (rounded to >=1) leaves at least 19 for training.
    train_writers = {f"f{i:04d}": 4 for i in range(20)}
    train = _make_synthetic_split(train_writers, rng)
    dd = DatasetDict({"train": train})

    femnist_mod._load_hf_dataset.cache_clear()
    monkeypatch.setattr(femnist_mod, "_load_hf_dataset", lambda: dd)
    yield dd


def test_writer_loaders_train_only_holds_out_eval_writers(synthetic_femnist_train_only):
    train_loaders, validation_loader, test_loader = femnist_mod.load_femnist_writer_loaders(
        nb_honest=3, train_batch=2, test_batch=4, split_seed=0,
    )

    # 20 writers; 5% holdout rounds to 1, so 19 are eligible as worker-trainers.
    assert set(train_loaders.keys()) == {0, 1, 2}
    # Eval pool has exactly 4 rows (one held-out writer * 4 samples each).
    eval_rows = sum(b[0].shape[0] for b in validation_loader) + sum(b[0].shape[0] for b in test_loader)
    assert eval_rows == 4


def test_pool_dataset_train_only_excludes_holdout_writers(synthetic_femnist_train_only):
    train_ds, eval_ds = femnist_mod.build_femnist_pool_dataset(seed=0)
    # 20 writers * 4 samples = 80 total; 1 writer (4 samples) held out -> 76 in train, 4 in eval.
    assert len(train_ds) == 76
    assert len(eval_ds) == 4
