from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.plot_sweep_base import BaseSweepPlotter


class PerParamPlotter(BaseSweepPlotter):
    """WAY 2: sweep line plots per secondary axis with remaining dims fixed."""

    def _plot_metric_column(self, metric, direction, column_key):
        """
        Plot per-parameter grids as described in WAY 2.

        Args:
          metric: str
            Metric stem name.
          direction: str
            Canonical direction name (avg, worse, or best).
          column_key: str
            Column key inside SweepRow.metrics.
        """
        if not self.axes:
            return
        if len(self.axes) == 1:
            self._single_axis_plot(metric, direction, column_key, self.axes[0])
            return
        for x_axis in self.axes:
            curve_candidates = [ax for ax in self.axes if ax.path != x_axis.path]
            for curve_axis in curve_candidates:
                remainder = [ax for ax in self.axes if ax.path not in (x_axis.path, curve_axis.path)]
                value_lists = [ax.values for ax in remainder]
                if not remainder:
                    combos = [()]
                else:
                    combos = list(itertools.product(*value_lists))
                for combo in combos:
                    fixed_parts = []
                    involved = [x_axis.path, curve_axis.path]
                    axis_value_pairs = list(zip(remainder, combo, strict=False)) if remainder else []
                    for axis_obj, fixed_val in axis_value_pairs:
                        fixed_parts.append((axis_obj.display_name, fixed_val))
                        involved.append(axis_obj.path)
                    rows = self.table.filtered_rows(involved)
                    filtered = []
                    for row in rows:
                        mismatch = False
                        for axis_obj, fixed_val in axis_value_pairs:
                            if axis_obj.path not in row.params:
                                mismatch = True
                                break
                            if row.params[axis_obj.path] != fixed_val:
                                mismatch = True
                                break
                        if not mismatch:
                            filtered.append(row)
                    fig, ax = plt.subplots(figsize=(7.5, 4.8))
                    color_idx = 0
                    drew_anything = False
                    for curve_val in curve_axis.values:
                        pts = []
                        for row in filtered:
                            if curve_axis.path not in row.params:
                                continue
                            if row.params[curve_axis.path] != curve_val:
                                continue
                            if column_key not in row.metrics:
                                continue
                            if x_axis.path not in row.params:
                                continue
                            xv = row.params[x_axis.path]
                            yv = row.metrics[column_key]
                            pts.append((xv, yv))
                        if not pts:
                            continue
                        pts.sort(key=lambda pair: _sort_key(pair[0]))
                        xs = [pair[0] for pair in pts]
                        ys = [pair[1] for pair in pts]
                        label = f"{curve_axis.display_name}={curve_val}"
                        ax.plot(_numeric_or_cat(xs), ys, marker="o", linewidth=1.6, label=label, color=_cycle_color(color_idx))
                        color_idx += 1
                        drew_anything = True
                    folder_name = self.auto_subfolder_name(fixed_parts) if fixed_parts else "no_fixed_axes"
                    subdir = (
                        self.output_dir
                        / f"x={_sanitize_label(x_axis.display_name)}"
                        / f"curves={_sanitize_label(curve_axis.display_name)}"
                        / folder_name
                    )
                    outfile = subdir / f"{metric}__{direction}.png"
                    if not drew_anything:
                        plt.close(fig)
                        print(
                            f"[plot_sweep] WARNING: skip empty per_parameter plot {outfile} "
                            f"(metric={metric}, direction={direction}, x={x_axis.display_name}, "
                            f"curves={curve_axis.display_name}, fixed={fixed_parts or 'none'})"
                        )
                        continue
                    ax.set_xlabel(x_axis.display_name)
                    ax.set_ylabel(f"{metric} ({direction})")
                    title_core = f"{metric} ({direction}) | x={x_axis.display_name} | curves={curve_axis.display_name}"
                    ax.set_title(title_core)
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="best", fontsize=8)
                    subdir.mkdir(parents=True, exist_ok=True)
                    fig.tight_layout()
                    fig.savefig(outfile, dpi=160, bbox_inches="tight")
                    plt.close(fig)

    def _single_axis_plot(self, metric, direction, column_key, x_axis):
        """
        Plot a single-axis sweep as one metric curve.

        Args:
          metric: str
            Metric stem name.
          direction: str
            Canonical direction name (avg, worse, or best).
          column_key: str
            Column key inside SweepRow.metrics.
          x_axis: SweepAxis
            Sole swept axis metadata.
        """
        rows = self.table.filtered_rows([x_axis.path])
        pts = []
        for row in rows:
            if column_key not in row.metrics:
                continue
            if x_axis.path not in row.params:
                continue
            xv = row.params[x_axis.path]
            yv = row.metrics[column_key]
            pts.append((xv, yv))
        subdir = self.output_dir / f"x={_sanitize_label(x_axis.display_name)}"
        outfile = subdir / f"{metric}__{direction}.png"
        if not pts:
            print(
                f"[plot_sweep] WARNING: skip empty per_parameter plot {outfile} "
                f"(metric={metric}, direction={direction}, x={x_axis.display_name})"
            )
            return
        pts.sort(key=lambda pair: _sort_key(pair[0]))
        xs = [pair[0] for pair in pts]
        ys = [pair[1] for pair in pts]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(_numeric_or_cat(xs), ys, marker="o", linewidth=1.8, color=_cycle_color(0))
        ax.set_xlabel(x_axis.display_name)
        ax.set_ylabel(f"{metric} ({direction})")
        ax.set_title(f"{metric} ({direction}) | x={x_axis.display_name}")
        ax.grid(True, alpha=0.25)
        subdir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        plt.close(fig)


def _sanitize_label(text):
    """
    Normalize labels for directories.

    Args:
      text: str
        Raw label text.

    return: str
      Safe directory token.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "axis"
    return cleaned


def _cycle_color(index):
    """
    Pick color from matplotlib cycle.

    Args:
      index: int
        Color index.

    return: Any
      Color token.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


def _sort_key(value):
    """
    Sort helper for sweep values.

    Args:
      value: Any
        Axis value.

    return: tuple
      Comparison tuple.
    """
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    return (1, str(value))


def _numeric_or_cat(xs):
    """
    Convert x values for matplotlib.

    Args:
      xs: list
        X values.

    return: list | ndarray
      Plottable sequence.
    """
    if not xs:
        return xs
    if all(isinstance(v, (int, float, np.floating)) for v in xs):
        return np.asarray(xs, dtype=float)
    return xs
