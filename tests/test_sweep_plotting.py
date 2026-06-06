from pathlib import Path

import optuna

from banditdl.utils.experiment_table import ExperimentTable, SweepRow
from banditdl.utils.plot_sweep_base import (
    STUDY_NAME,
    load_sweep_study,
    optuna_storage_url,
)
from banditdl.utils.sweep_plotting import SweepPlotter


def test_heatmap_renders_file(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 10, "g": "a"}, {"metric__avg": 1.0}),
        SweepRow({"x": 2, "y": 20, "g": "a"}, {"metric__avg": 5.0}),
    ]
    table = ExperimentTable(rows)
    plotter = SweepPlotter(table, tmp_path)

    plotter.plot_heatmap("metric", "avg", "x", "y", {"g": "a"})

    assert (tmp_path / "heatmap" / "metric_avg_x_y.png").exists()


def test_optuna_storage_url_is_loadable(tmp_path: Path):
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=optuna_storage_url(tmp_path),
        direction="maximize",
    )
    study.set_user_attr("smoke", True)

    loaded = load_sweep_study(tmp_path)

    assert loaded.study_name == STUDY_NAME
    assert loaded.user_attrs["smoke"] is True
