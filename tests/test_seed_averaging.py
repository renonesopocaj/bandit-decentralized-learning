import numpy as np
import pytest

from banditdl.utils.metrics import read_eval
from banditdl.utils.seed_averaging import aggregate_seed_results, seed_result_dir


def _write_seed_result(result_dir, seed, validation_values, accuracy_array, selected_neighbors):
    seed_dir = seed_result_dir(result_dir, seed)
    seed_dir.mkdir(parents=True)
    lines = ["# Step number\tCross-accuracy"]
    for step, value in zip([0, 2], validation_values, strict=True):
        lines.append(f"{step}\t{value}")
    (seed_dir / "validation").write_text("\n".join(lines) + "\n")
    np.save(seed_dir / "validation_accuracies.npy", accuracy_array)
    np.save(seed_dir / "selected_neighbors.npy", selected_neighbors)


def test_aggregate_seed_results_writes_public_mean_and_by_seed_arrays(tmp_path):
    result_dir = tmp_path / "results"
    _write_seed_result(
        result_dir,
        10,
        [0.2, 0.4],
        np.array([[0.1, 0.3], [0.5, 0.7]]),
        np.array([[[1], [0]]]),
    )
    _write_seed_result(
        result_dir,
        11,
        [0.4, 0.8],
        np.array([[0.3, 0.5], [0.7, 0.9]]),
        np.array([[[0], [1]]]),
    )

    aggregate_seed_results(result_dir, [10, 11])

    steps, validation = read_eval(result_dir / "validation")
    np.testing.assert_allclose(steps, np.array([0.0, 2.0]))
    np.testing.assert_allclose(validation, np.array([0.3, 0.6]))
    np.testing.assert_allclose(
        np.load(result_dir / "validation_accuracies.npy"),
        np.array([[0.2, 0.4], [0.6, 0.8]]),
    )
    assert np.load(result_dir / "validation_accuracies_by_seed.npy").shape == (2, 2, 2)
    assert np.load(result_dir / "selected_neighbors_by_seed.npy").shape == (2, 1, 2, 1)
    assert not (result_dir / "selected_neighbors.npy").exists()


def test_aggregate_seed_results_rejects_partial_metric_files(tmp_path):
    result_dir = tmp_path / "results"
    _write_seed_result(
        result_dir,
        10,
        [0.2, 0.4],
        np.array([[0.1, 0.3], [0.5, 0.7]]),
        np.array([[[1], [0]]]),
    )
    seed_result_dir(result_dir, 11).mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        aggregate_seed_results(result_dir, [10, 11])
