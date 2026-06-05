from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.experiment_table import ExperimentTable
from banditdl.utils.plot_sweep_base import column_key_for


def _sanitize_label(text):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    return cleaned.strip("_") or "axis"


def _cycle_color(index):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


class SweepPlotter:
    def __init__(self, table: ExperimentTable, output_dir: Path):
        self.table = table
        self.output_dir = output_dir

    def plot_per_parameter(self, metric: str, direction: str):
        """WAY 2: sweep line plots per secondary axis with remaining dims fixed."""
        axes = getattr(self.table, "axes_meta", [])
        if not axes:
            return

        column_key = column_key_for(metric, direction)

        for x_axis in axes:
            # We want to plot X-axis vs Metric, for every possible combination of other axes
            other_axes = [ax for ax in axes if ax.path != x_axis.path]

            # For "per_parameter", we typically pick ONE axis to be the "Series/Curves"
            # and fix all others.
            for curve_axis in other_axes:
                fixed_axes = [ax for ax in other_axes if ax.path != curve_axis.path]
                fixed_combos = self.table.get_combinations([ax.path for ax in fixed_axes]) or [{}]

                for fixed_params in fixed_combos:
                    self._draw_line_plot(
                        metric, direction, column_key,
                        x_axis, curve_axis, fixed_params
                    )

    def _draw_line_plot(self, metric, direction, column_key, x_axis, curve_axis, fixed_params):
        # Filter table by fixed params
        subset = self.table.filter(fixed_params)
        pivoted = subset.pivot(x_axis.path, curve_axis.path)

        if not pivoted:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (curve_val, points) in enumerate(pivoted.items()):
            xs = [p[0] for p in points]
            ys = [p[1].get(column_key, np.nan) for p in points]
            ax.plot(xs, ys, marker="o", label=f"{curve_axis.display_name}={curve_val}", color=_cycle_color(i))

        ax.set_xlabel(x_axis.display_name)
        ax.set_ylabel(f"{metric} ({direction})")

        fixed_desc = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in fixed_params.items())
        title = f"{metric} | {x_axis.display_name}"
        if fixed_desc:
            title += f"\nfixed: {fixed_desc}"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

        # Path management
        subdir = self.output_dir / "per_parameter" / f"x_{_sanitize_label(x_axis.display_name)}"
        if fixed_params:
            fixed_str = "_".join(f"{_sanitize_label(k)}={_sanitize_label(v)}" for k, v in fixed_params.items())
            subdir = subdir / fixed_str

        subdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(subdir / f"{metric}_{direction}_{_sanitize_label(curve_axis.display_name)}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def plot_heatmap(self, metric: str, direction: str, x_axis_path: str, y_axis_path: str, fixed_params: dict | None = None):
        """Plot a heatmap for two parameters."""
        fixed_params = fixed_params or {}
        subset = self.table.filter(fixed_params)

        # This is a bit more complex as we need a 2D grid
        x_vals = self.table.get_unique_values(x_axis_path)
        y_vals = self.table.get_unique_values(y_axis_path)

        z = np.full((len(y_vals), len(x_vals)), np.nan)
        column_key = column_key_for(metric, direction)

        for row in subset.rows:
            xv = row.params.get(x_axis_path)
            yv = row.params.get(y_axis_path)
            if xv in x_vals and yv in y_vals:
                xi = x_vals.index(xv)
                yi = y_vals.index(yv)
                z[yi, xi] = row.metrics.get(column_key, np.nan)

        if np.isnan(z).all():
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(z, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, label=f"{metric} ({direction})")

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=45)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])

        ax.set_xlabel(x_axis_path.rsplit('.', maxsplit=1)[-1])
        ax.set_ylabel(y_axis_path.rsplit('.', maxsplit=1)[-1])
        ax.set_title(f"Heatmap: {metric}\n{fixed_params}")

        output_path = self.output_dir / "heatmap" / f"{metric}_{direction}_{_sanitize_label(x_axis_path)}_{_sanitize_label(y_axis_path)}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
