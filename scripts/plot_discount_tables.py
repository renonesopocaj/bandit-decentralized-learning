"""Discount comparison tables (discounted vs non-discounted samplers).

Renders annotated heatmap-style tables in the spirit of
``banditdl.utils.sweep_plotting.SweepPlotter.plot_heatmap`` but driven by a
directory of plain Hydra runs instead of an Optuna study.

For each sampler family (cucb, cts) we build a table with the discount factor
gamma on the rows and the reward signal on the columns. The non-discounted
sampler (plain ``cucb`` / ``cts``) is shown as the ``gamma = 1.0`` baseline row.
A combined table stacks both families with a separator line between them.

Cells are scored with the same scalar reduction the sweep plotter uses
(``scalar_reduce_seed_outer``), so values are comparable to the heatmaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from banditdl.utils.metrics import (
    HIGHER_IS_WORSE,
    MetricLoader,
    resolve_metric,
    scalar_reduce_seed_outer,
)

# Families and their discounted counterparts. The bare sampler is the gamma=1
# (no discounting) baseline.
FAMILIES = {
    "cucb": "discounted_cucb",
    "cts": "discounted_cts",
}
BASELINE_GAMMA = 1.0

DEFAULT_METRICS = (
    "validation_accuracy",
    "global_accuracy",
    "validation_loss",
    "normalized_regret",
    "regret",
)


def _read_overrides(run_dir: Path) -> dict[str, str]:
    path = run_dir / ".hydra" / "overrides.yaml"
    if not path.exists():
        return {}
    return dict(item.split("=", 1) for item in yaml.safe_load(path.read_text()))


def collect_rows(runs_root: Path) -> list[dict]:
    """One record per run, tagging family / gamma / reward and the results dir."""
    rows = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        ov = _read_overrides(run_dir)
        sampler = ov.get("sampler")
        if sampler is None:
            continue

        if sampler in FAMILIES:  # bare sampler -> baseline
            family, gamma = sampler, BASELINE_GAMMA
        elif sampler in FAMILIES.values():  # discounted_* -> family
            family = next(k for k, v in FAMILIES.items() if v == sampler)
            gamma = float(ov.get("sampler.params.gamma"))
        else:
            continue

        results = run_dir / "results"
        if not results.exists():
            continue
        rows.append(
            {
                "run": run_dir.name,
                "family": family,
                "gamma": gamma,
                "reward": ov.get("sampler.reward"),
                "results": results,
            }
        )
    return rows


def metric_scalar(results_dir: Path, metric: str, direction: str) -> float:
    try:
        raw = MetricLoader(results_dir).load_seed_values(metric)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[discount_tables] skip {metric} for {results_dir}: {exc}")
        return np.nan
    return scalar_reduce_seed_outer(metric, raw, direction)


def _draw_table(grid, row_labels, col_labels, metric, direction, title, save_path,
                separator_after=None):
    """Annotated imshow grid -- a colored, value-labelled table."""
    higher_worse = resolve_metric(metric) in HIGHER_IS_WORSE
    # Fixed scale: yellow = high value, dark purple = low value (viridis), to
    # stay consistent with the other plots in the project.
    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(2.4 + 1.8 * n_cols, 1.2 + 0.62 * n_rows))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label=f"{metric} ({direction})")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("reward")
    ax.set_ylabel("discount gamma")

    finite = grid[np.isfinite(grid)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "-", ha="center", va="center", color="0.6")
                continue
            txt_color = "white" if val < mid else "black"
            ax.text(j, i, f"{val:.4g}", ha="center", va="center",
                    color=txt_color, fontsize=9)

    if separator_after is not None:
        ax.axhline(separator_after + 0.5, color="white", linewidth=3)
        ax.axhline(separator_after + 0.5, color="black", linewidth=1.2)

    direction_note = "higher is worse" if higher_worse else "higher is better"
    ax.set_title(f"{title}\n{metric} ({direction}, {direction_note})")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _gamma_label(gamma: float) -> str:
    return "1.0 (baseline)" if gamma == BASELINE_GAMMA else f"{gamma:g}"


def build_tables(rows, metrics, direction, output_dir: Path):
    rewards = sorted({r["reward"] for r in rows})
    gammas = sorted({r["gamma"] for r in rows})  # 1.0 first

    def lookup(family, gamma, reward):
        # Multiple runs can share a cell (e.g. short smoke tests alongside the
        # full run); take the first one that actually has the metric.
        value = np.nan
        for r in rows:
            if r["family"] == family and r["gamma"] == gamma and r["reward"] == reward:
                value = metric_scalar(r["results"], metric, direction)
                if np.isfinite(value):
                    return value
        return value

    written = []
    for metric in metrics:
        # Per-family tables.
        for family in FAMILIES:
            grid = np.array([[lookup(family, g, rw) for rw in rewards] for g in gammas])
            if not np.isfinite(grid).any():
                continue
            save = output_dir / metric / f"{family}_discount_table.png"
            _draw_table(
                grid, [_gamma_label(g) for g in gammas], rewards, metric, direction,
                f"{family.upper()}: discounted vs non-discounted", save,
            )
            written.append(save)

        # Combined table: families stacked with a separator.
        combined_rows, row_labels = [], []
        sep_index = None
        for fam_idx, family in enumerate(FAMILIES):
            for g in gammas:
                combined_rows.append([lookup(family, g, rw) for rw in rewards])
                row_labels.append(f"{family} | g={_gamma_label(g)}")
            if fam_idx == 0:
                sep_index = len(combined_rows) - 1
        grid = np.array(combined_rows)
        if np.isfinite(grid).any():
            save = output_dir / metric / "combined_discount_table.png"
            _draw_table(
                grid, row_labels, rewards, metric, direction,
                "All samplers: discounted vs non-discounted", save,
                separator_after=sep_index,
            )
            written.append(save)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path,
                        help="Directory of Hydra runs, e.g. .hydra_runs/2026-06-10")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--direction", default="avg", choices=["avg", "worse", "best"])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.runs_root / "comparison_tables")
    rows = collect_rows(args.runs_root)
    if not rows:
        raise SystemExit(f"No usable runs found under {args.runs_root}")
    print(f"[discount_tables] {len(rows)} runs, rewards/gammas/families detected")
    written = build_tables(rows, args.metrics, args.direction, output_dir)
    print(f"[discount_tables] wrote {len(written)} tables under {output_dir}")
    for path in written:
        print("  ", path)


if __name__ == "__main__":
    main()
