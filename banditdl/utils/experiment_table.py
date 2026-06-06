from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SweepRow:
    params: dict[str, Any]
    metrics: dict[str, float]


class ExperimentTable:
    """A high-level table for experimental data with slicing and pivoting support."""

    def __init__(self, rows: list[SweepRow]):
        self.rows = rows

    @property
    def param_keys(self) -> list[str]:
        if not self.rows:
            return []
        return sorted(self.rows[0].params.keys())

    @property
    def metric_keys(self) -> list[str]:
        if not self.rows:
            return []
        return sorted(self.rows[0].metrics.keys())

    def get_unique_values(self, param_path: str) -> list[Any]:
        values = {row.params[param_path] for row in self.rows if param_path in row.params}
        return sorted(list(values), key=lambda x: (isinstance(x, str), x))

    def filter(self, fixed_params: dict[str, Any]) -> ExperimentTable:
        filtered = []
        for row in self.rows:
            match = True
            for k, v in fixed_params.items():
                if row.params.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(row)
        return ExperimentTable(filtered)

    def pivot(self, x_param: str, series_param: str | None = None) -> dict[Any, list[tuple[Any, dict[str, float]]]]:
        """
        Pivot the table for plotting.

        Returns:
            A mapping from series_value to a list of (x_value, metrics_dict).
        """
        result = {}
        for row in self.rows:
            x_val = row.params.get(x_param)
            s_val = row.params.get(series_param) if series_param else "default"
            if x_val is None:
                continue
            if s_val not in result:
                result[s_val] = []
            result[s_val].append((x_val, row.metrics))

        # Sort by x_val
        for s_val in result:
            result[s_val].sort(key=lambda pair: (isinstance(pair[0], (int, float)), pair[0]))

        return result

    def get_combinations(self, param_paths: list[str]) -> list[dict[str, Any]]:
        """Return all existing combinations of values for the given parameter paths."""
        combos = []
        seen = set()
        for row in self.rows:
            combo = {p: row.params.get(p) for p in param_paths}
            # Freeze for set check
            frozen = tuple(sorted(combo.items()))
            if frozen not in seen:
                seen.add(frozen)
                combos.append(combo)
        return combos
