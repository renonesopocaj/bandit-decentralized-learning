"""FEMNIST data loading.

FEMNIST is the LEAF Federated EMNIST benchmark: 62 classes (digits + lowercase +
uppercase letters), naturally non-IID by writer. This module reads the
`flwrlabs/femnist` HuggingFace dataset and exposes two integration paths:

- `load_femnist_writer_loaders` — one writer per node. Each honest worker gets the
  DataLoader of one FEMNIST writer; validation/test pools the official test split.
  The `heterogeneity=*` configuration is bypassed in this path because the writers
  already provide natural non-IID structure.

- `build_femnist_pool_dataset` — concatenate all writers' training samples into one
  `torch.utils.data.Dataset` exposing a `targets` attribute, so the existing
  Dirichlet / pathological / grouped partitioning machinery in `Dataset.__init__`
  can be applied on top.

The HF dataset is cached transparently under `HF_DATASETS_CACHE` (defaults to
~/.cache/huggingface/datasets). On the EPFL GPU cluster, override that env var to
a shared/scratch path and pre-download once from the login node.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as T
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

FEMNIST_HF_NAME = "flwrlabs/femnist"
FEMNIST_NUM_CLASSES = 62
# Fraction of writers (from the single `train` split) reserved as the held-out
# eval pool when the HF dataset has no separate `test` split. Standard FL setup:
# the eval pool is a fresh set of writers the workers never train on.
EVAL_WRITER_FRACTION = 0.05

_DEFAULT_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


def _label_column(hf_split) -> str:
    for candidate in ("character", "label", "labels"):
        if candidate in hf_split.column_names:
            return candidate
    raise ValueError(
        f"FEMNIST split has no label column among 'character'/'label'/'labels'; "
        f"found {hf_split.column_names}"
    )


def _writer_column(hf_split) -> str:
    for candidate in ("writer_id", "user", "client_id"):
        if candidate in hf_split.column_names:
            return candidate
    raise ValueError(
        f"FEMNIST split has no writer column among 'writer_id'/'user'/'client_id'; "
        f"found {hf_split.column_names}"
    )


@lru_cache(maxsize=2)
def _load_hf_dataset() -> DatasetDict:
    """Load (and cache) the FEMNIST HuggingFace dataset.

    Cached at module level so repeat calls within one process are cheap.
    """
    return load_dataset(FEMNIST_HF_NAME)


class _FEMNISTWrappedDataset(Dataset):
    """Wrap one HF split as a torch Dataset with a `.targets` attribute."""

    def __init__(self, hf_split, transform=None):
        self._hf = hf_split
        self._label_col = _label_column(hf_split)
        self._transform = transform if transform is not None else _DEFAULT_TRANSFORM
        self.targets = torch.tensor(self._hf[self._label_col], dtype=torch.long)

    def __len__(self):
        return len(self._hf)

    def __getitem__(self, idx):
        row = self._hf[int(idx)]
        return self._transform(row["image"]), int(row[self._label_col])


def _resolve_split(hf, name: str):
    if name in hf:
        return hf[name]
    raise KeyError(f"FEMNIST HF dataset has no split named {name!r}; available={list(hf.keys())}")


def _split_writers_train_eval(
    train_split, eval_writer_fraction: float, seed: int
) -> tuple[dict[str, list[int]], list[int]]:
    """Return ({train_writer: [indices], ...}, [eval_indices_in_train_split]).

    Deterministically holds out `eval_writer_fraction` of writers from the train
    split and returns their row indices as the eval pool. Used when the HF dataset
    has no separate `test` split.
    """
    all_writers = _writer_groups(train_split)
    writer_ids = sorted(all_writers.keys())
    rng = np.random.default_rng(seed)
    nb_eval = max(1, int(round(eval_writer_fraction * len(writer_ids))))
    eval_choice = set(rng.choice(len(writer_ids), size=nb_eval, replace=False).tolist())
    train_writers: dict[str, list[int]] = {}
    eval_indices: list[int] = []
    for i, writer in enumerate(writer_ids):
        if i in eval_choice:
            eval_indices.extend(all_writers[writer])
        else:
            train_writers[writer] = all_writers[writer]
    return train_writers, eval_indices


def _get_eval_dataset(hf, train_split, seed: int) -> Dataset:
    """Return the eval dataset: hf['test'] if present, else a Subset of writers held out from train."""
    if "test" in hf:
        return _FEMNISTWrappedDataset(hf["test"])
    _, eval_indices = _split_writers_train_eval(train_split, EVAL_WRITER_FRACTION, seed)
    return Subset(_FEMNISTWrappedDataset(train_split), eval_indices)


def _train_writers_only(hf, train_split, seed: int) -> dict[str, list[int]]:
    """Return {writer_id: [indices]} for writers eligible to be worker-trainers.

    If hf has a separate test split, all writers in train_split are eligible.
    Otherwise the eval-holdout writers are removed.
    """
    if "test" in hf:
        return _writer_groups(train_split)
    train_writers, _ = _split_writers_train_eval(train_split, EVAL_WRITER_FRACTION, seed)
    return train_writers


def _writer_groups(hf_split) -> dict[str, list[int]]:
    """Return {writer_id: [sample_index, ...]} over the given HF split."""
    writer_col = _writer_column(hf_split)
    writers = hf_split[writer_col]
    groups: dict[str, list[int]] = {}
    for idx, w in enumerate(writers):
        groups.setdefault(str(w), []).append(idx)
    return groups


def _select_writers(
    writers_to_indices: dict[str, list[int]], nb_writers: int, seed: int, writers_cap: int | None = None
) -> list[str]:
    """Pick `nb_writers` writer IDs deterministically from seed; capped by writers_cap if given."""
    pool = sorted(writers_to_indices.keys())
    if writers_cap is not None:
        pool = pool[: int(writers_cap)]
    if nb_writers > len(pool):
        raise ValueError(
            f"Requested {nb_writers} FEMNIST writers but only {len(pool)} are available "
            f"(writers_cap={writers_cap})"
        )
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pool), size=nb_writers, replace=False)
    return [pool[int(i)] for i in chosen_idx]


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_femnist_writer_loaders(
    nb_honest: int,
    train_batch: int,
    test_batch: int,
    validation_ratio: float = 0.5,
    split_seed: int = 0,
    writers_cap: int | None = None,
) -> tuple[dict[int, DataLoader], DataLoader, DataLoader]:
    """Build per-writer training DataLoaders plus pooled validation/test loaders.

    Returns the same shape as `make_train_validation_test_datasets`:
    `(train_loaders_dict, validation_loader, test_loader)`.
    """
    hf = _load_hf_dataset()
    train_split = _resolve_split(hf, "train")

    train_groups = _train_writers_only(hf, train_split, seed=split_seed)
    selected = _select_writers(train_groups, nb_honest, seed=split_seed, writers_cap=writers_cap)

    train_wrapped = _FEMNISTWrappedDataset(train_split)
    train_loaders: dict[int, DataLoader] = {}
    for worker_id, writer in enumerate(selected):
        indices = train_groups[writer]
        train_loaders[worker_id] = _make_loader(
            Subset(train_wrapped, indices), batch_size=train_batch, shuffle=True
        )

    test_wrapped = _get_eval_dataset(hf, train_split, seed=split_seed)
    total = len(test_wrapped)
    if total < 2:
        raise ValueError("FEMNIST test split needs at least 2 samples")
    val_size = int(round(total * validation_ratio))
    val_size = min(max(val_size, 1), total - 1)
    test_size = total - val_size
    generator = torch.Generator().manual_seed(split_seed)
    val_set, test_set = torch.utils.data.random_split(
        test_wrapped, [val_size, test_size], generator=generator
    )
    validation_loader = _make_loader(val_set, batch_size=test_batch, shuffle=False)
    test_loader = _make_loader(test_set, batch_size=test_batch, shuffle=False)

    return train_loaders, validation_loader, test_loader


def build_femnist_pool_dataset(seed: int = 0) -> tuple[Dataset, Dataset]:
    """Return (train_dataset, eval_dataset) as plain torch Datasets exposing `.targets`.

    Intended for the pool mode, where the existing Dirichlet/pathological/grouped
    partitioners in `Dataset.__init__` operate on the pooled training set. When
    the HF dataset has no separate `test` split, the eval pool is built from a
    deterministic writer-level holdout (`EVAL_WRITER_FRACTION`).
    """
    hf = _load_hf_dataset()
    train_split = _resolve_split(hf, "train")
    if "test" in hf:
        train_for_workers = train_split
    else:
        train_writers = _train_writers_only(hf, train_split, seed=seed)
        train_indices = [i for indices in train_writers.values() for i in indices]
        train_for_workers = train_split.select(train_indices)
    return (
        _FEMNISTWrappedDataset(train_for_workers),
        _get_eval_dataset(hf, train_split, seed=seed),
    )
