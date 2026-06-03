from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.plot_sweep_base import BaseSweepPlotter


class AllTogetherPlotter(BaseSweepPlotter):
    """WAY 1: overlay curves for each tuple of non-x-axis parameter values."""

    def _plot_metric_column(self, metric, direction, column_key):
        """
        Plot metric slices with shared x-axis per swept dimension.

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
        for x_axis in self.axes:
            others = [ax for ax in self.axes if ax.path != x_axis.path]
            involved = [x_axis.path] + [ax.path for ax in others]
            rows = self.table.filtered_rows(involved)
            grouped = defaultdict(list)
            for row in rows:
                if any(ax.path not in row.params for ax in others):
                    continue
                group_key = tuple(row.params[ax.path] for ax in others)
                grouped[group_key].append(row)
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            color_idx = 0
            drew_anything = False
            for group_key in sorted(grouped.keys(), key=lambda item: tuple(map(str, item))):
                bucket = grouped[group_key]
                points = []
                for row in bucket:
                    if column_key not in row.metrics:
                        continue
                    if x_axis.path not in row.params:
                        continue
                    xv = row.params[x_axis.path]
                    yv = row.metrics[column_key]
                    points.append((xv, yv))
                if not points:
                    continue
                points.sort(key=lambda pair: _sort_key(pair[0]))
                xs = [pair[0] for pair in points]
                ys = [pair[1] for pair in points]
                label = _legend_tuple_label(list(others), group_key) if others else f"{x_axis.display_name}"
                color = _cycle_color(color_idx)
                color_idx += 1
                ax.plot(_numeric_or_cat(xs), ys, marker="o", linewidth=1.6, label=label, color=color)
                drew_anything = True
            subdir = self.output_dir / f"x={_sanitize_label(x_axis.display_name)}"
            outfile = subdir / f"{metric}__{direction}.png"
            if not drew_anything:
                plt.close(fig)
                print(
                    f"[plot_sweep] WARNING: skip empty all_together plot {outfile} "
                    f"(metric={metric}, direction={direction}, x={x_axis.display_name})"
                )
                continue
            ax.set_xlabel(x_axis.display_name)
            ax.set_ylabel(f"{metric} ({direction})")
            title_bits = [f"{metric} ({direction})", f"x={x_axis.display_name}"]
            ax.set_title("\n".join(title_bits))
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            subdir.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(outfile, dpi=160, bbox_inches="tight")
            plt.close(fig)


def _sanitize_label(text):
    """
    Normalize axis labels for filesystem paths.

    Args:
      text: str
        Raw label text.

    return: str
      Shortened safe label.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "axis"
    return cleaned


def _legend_tuple_label(axes, values):
    """
    Build legend text for fixed parameter tuple.

    Args:
      axes: list
        SweepAxis instances corresponding to tuple entries.
      values: tuple
        Parameter values aligned with axes order.

    return: str
      comma-separated label string.
    """
    parts = []
    for axis_obj, val in zip(axes, values, strict=True):
        parts.append(f"{axis_obj.display_name}={val}")
    return ", ".join(parts)


def _cycle_color(index):
    """
    Pick a matplotlib color from the default cycle.

    Args:
      index: int
        Color index.

    return: Any
      Matplotlib color token.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


def _sort_key(value):
    """
    Build a sort key for mixed-type sweep values.

    Args:
      value: Any
        Parameter value on x-axis.

    return: tuple
      Comparable tuple for sorting.
    """
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    return (1, str(value))


def _numeric_or_cat(xs):
    """
    Convert x values to plottable array preserving categorical order.

    Args:
      xs: list
        Original x values.

    return: list | ndarray
      Values suitable for matplotlib.plot.
    """
    if not xs:
        return xs
    if all(isinstance(v, (int, float, np.floating)) for v in xs):
        return np.asarray(xs, dtype=float)
    return xs
