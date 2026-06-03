import pytest
import numpy as np

from banditdl.utils.metrics import MetricKey, MetricLoader, TimeAverage, min_, scalar_reduce_seed_outer


def test_normalized_regret_is_derived_from_regret(tmp_path):
    regret = np.array([[2.0, 4.0], [6.0, 8.0], [9.0, 12.0]])
    np.save(tmp_path / "regret.npy", regret)

    values = MetricLoader(tmp_path).load_values(MetricKey.NORMALIZED_REGRET)

    np.testing.assert_allclose(
        values,
        np.array([[2.0, 4.0], [3.0, 4.0], [3.0, 4.0]]),
    )


def test_validation_accuracy_can_interpolate_to_full_round_axis(tmp_path):
    (tmp_path / "validation").write_text("0\t0.1\n2\t0.5\n4\t0.9\n")
    np.save(
        tmp_path / "validation_accuracies.npy",
        np.array([[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]),
    )
    np.save(tmp_path / "regret.npy", np.zeros((5, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_ACCURACIES,
        interpolate_eval=True,
    )

    np.testing.assert_allclose(data.x, np.arange(5))
    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [0.0, 0.2],
                [0.2, 0.4],
                [0.4, 0.6],
                [0.6, 0.8],
                [0.8, 1.0],
            ]
        ),
    )


def test_seed_stacked_validation_accuracy_interpolates_on_time_axis(tmp_path):
    (tmp_path / "validation").write_text("0\t0.1\n2\t0.5\n4\t0.9\n")
    np.save(
        tmp_path / "validation_accuracies_by_seed.npy",
        np.array(
            [
                [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]],
                [[0.1, 0.3], [0.5, 0.7], [0.9, 1.1]],
            ]
        ),
    )
    np.save(tmp_path / "regret.npy", np.zeros((5, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_ACCURACIES,
        interpolate_eval=True,
    )

    np.testing.assert_allclose(data.x, np.arange(5))
    assert data.values.shape == (2, 5, 2)
    np.testing.assert_allclose(data.values[0, :, 0], np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
    np.testing.assert_allclose(data.values[1, :, 1], np.array([0.3, 0.5, 0.7, 0.9, 1.1]))


def test_node_reduction_averages_seeds_outermost():
    values = np.array(
        [
            [[0.0, 100.0], [5.0, 10.0]],
            [[100.0, 0.0], [20.0, 1.0]],
        ]
    )

    np.testing.assert_allclose(min_.fn(values), np.array([0.0, 3.0]))


def test_scalar_worse_reduction_averages_per_seed_worst_values():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[100.0, 0.0], [0.0, 0.0]],
        ]
    )

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_LOSSES, values, "worse")

    assert reduced == pytest.approx(52.0)


def test_time_average_uses_time_axis_after_seed_axis():
    values = np.array([[[2.0], [6.0], [12.0]], [[4.0], [10.0], [18.0]]])

    averaged = TimeAverage()(values)

    np.testing.assert_allclose(
        averaged,
        np.array([[[2.0], [3.0], [4.0]], [[4.0], [5.0], [6.0]]]),
    )


def test_metric_loader_loads_selected_reward_extrema(tmp_path):
    values = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.save(tmp_path / "reward_selected_min.npy", values)

    loaded = MetricLoader(tmp_path).load_values(MetricKey.REWARD_SELECTED_MIN)

    np.testing.assert_allclose(loaded, values)
