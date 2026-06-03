from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.plot_sweep_base import BaseSweepPlotter


class HeatmapPlotter(BaseSweepPlotter):
    """WAY 3: heatmaps for axis pairs with remaining parameters fixed."""

    def _plot_metric_column(self, metric, direction, column_key):
        """
        Render heatmaps for each axis pair and metric scalar.

        Args:
          metric: str
            Metric stem name.
          direction: str
            Canonical direction name (avg, worse, or best).
          column_key: str
            Column key inside SweepRow.metrics.
        """
        if len(self.axes) < 2:
            return
        for idx_x in range(len(self.axes)):
            for idx_y in range(len(self.axes)):
                if idx_y <= idx_x:
                    continue
                x_axis = self.axes[idx_x]
                y_axis = self.axes[idx_y]
                remainder = [ax for ax in self.axes if ax.path not in (x_axis.path, y_axis.path)]
                value_lists = [ax.values for ax in remainder]
                if not remainder:
                    combos = [()]
                else:
                    combos = list(itertools.product(*value_lists))
                for combo in combos:
                    fixed_parts = []
                    involved = [x_axis.path, y_axis.path]
                    pairs = list(zip(remainder, combo)) if remainder else []
                    for axis_obj, fixed_val in pairs:
                        fixed_parts.append((axis_obj.display_name, fixed_val))
                        involved.append(axis_obj.path)
                    rows = self.table.filtered_rows(involved)
                    filtered = []
                    for row in rows:
                        bad = False
                        for axis_obj, fixed_val in pairs:
                            if axis_obj.path not in row.params:
                                bad = True
                                break
                            if row.params[axis_obj.path] != fixed_val:
                                bad = True
                                break
                        if not bad:
                            filtered.append(row)
                    matrix = np.full((len(y_axis.values), len(x_axis.values)), np.nan, dtype=float)
                    for yi, y_val in enumerate(y_axis.values):
                        for xi, x_val in enumerate(x_axis.values):
                            picked = None
                            for row in filtered:
                                if x_axis.path not in row.params or y_axis.path not in row.params:
                                    continue
                                if row.params[x_axis.path] != x_val:
                                    continue
                                if row.params[y_axis.path] != y_val:
                                    continue
                                if column_key not in row.metrics:
                                    continue
                                picked = row.metrics[column_key]
                                break
                            if picked is not None:
                                matrix[yi, xi] = picked
                    folder_name = self.auto_subfolder_name(fixed_parts) if fixed_parts else "no_fixed_axes"
                    subdir = (
                        self.output_dir
                        / f"axes={_sanitize_label(x_axis.display_name)}_{_sanitize_label(y_axis.display_name)}"
                        / folder_name
                    )
                    outfile = subdir / f"{metric}__{direction}.png"
                    if np.all(np.isnan(matrix)):
                        print(
                            f"[plot_sweep] WARNING: skip empty heatmap {outfile} "
                            f"(metric={metric}, direction={direction}, x={x_axis.display_name}, "
                            f"y={y_axis.display_name}, fixed={fixed_parts or 'none'})"
                        )
                        continue
                    fig, ax = plt.subplots(figsize=(7.2, 5.6))
                    cmap = plt.cm.viridis
                    cmap.set_bad(color="0.85")
                    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
                    ax.set_xticks(range(len(x_axis.values)))
                    ax.set_xticklabels([str(v) for v in x_axis.values], rotation=35, ha="right")
                    ax.set_yticks(range(len(y_axis.values)))
                    ax.set_yticklabels([str(v) for v in y_axis.values])
                    ax.set_xlabel(x_axis.display_name)
                    ax.set_ylabel(y_axis.display_name)
                    title_bits = [
                        f"{metric} ({direction})",
                        f"x={x_axis.display_name}, y={y_axis.display_name}",
                    ]
                    if fixed_parts:
                        title_bits.append(self.auto_subfolder_name(fixed_parts))
                    ax.set_title("\n".join(title_bits))
                    for yi in range(matrix.shape[0]):
                        for xi in range(matrix.shape[1]):
                            val = matrix[yi, xi]
                            if np.isnan(val):
                                text = ""
                            else:
                                text = f"{val:.3g}"
                            ax.text(xi, yi, text, ha="center", va="center", color="black", fontsize=8)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    subdir.mkdir(parents=True, exist_ok=True)
                    fig.tight_layout()
                    fig.savefig(outfile, dpi=160, bbox_inches="tight")
                    plt.close(fig)


def _sanitize_label(text):
    """
    Normalize labels for filesystem paths.

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
