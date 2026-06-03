from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.seed_averaging import run_seed_averaged, seed_result_dir
from banditdl.utils.plot_sweep_base import (
    build_axis_metadata,
    enumerate_valid_param_dicts,
    normalize_directions,
    normalize_plot_modes,
    plot_sweep,
    trial_folder_name,
)


def _read_metric_file_max(metric_file: Path) -> float:
    if not metric_file.exists():
        raise FileNotFoundError(f"Missing metric file: {metric_file}")

    metric_values = []
    for line in metric_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) >= 2:
            metric_values.append(float(fields[1]))
    if not metric_values:
        raise ValueError(f"No metric values found in: {metric_file}")
    return max(metric_values)


def _read_seed_metric_file_max(result_dir: Path, seeds: list[int], metric_name: str) -> tuple[float, list[float]]:
    seed_values = [
        _read_metric_file_max(seed_result_dir(result_dir, seed) / metric_name)
        for seed in seeds
    ]
    return float(np.mean(seed_values)), seed_values


def _copy_dict_config(cfg: DictConfig) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(copied, DictConfig):
        raise TypeError("Expected a DictConfig copy")
    return copied


def _resolved_trial_params(trial) -> dict:
    if trial.params:
        return dict(trial.params)
    resolved = trial.user_attrs.get("resolved_params")
    if isinstance(resolved, dict):
        return dict(resolved)
    return {}


def _objective(trial, base_cfg: DictConfig, trials_root: Path, axis_lookup: dict, combos: list) -> float:
    trial_index = int(trial.number)
    if trial_index >= len(combos):
        raise IndexError(
            f"Trial index {trial_index} out of bounds for {len(combos)} combinations"
        )
    trial_params = dict(combos[trial_index])

    trial_cfg = _copy_dict_config(base_cfg)
    for path, value in trial_params.items():
        OmegaConf.update(trial_cfg, path, value, merge=False)

    folder_name = trial_folder_name(trial_params, axis_lookup)
    trial_result_dir = trials_root / folder_name / "results"
    trial_result_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = build_engine_config(trial_cfg)
    num_seeds = int(trial_cfg.num_seeds)
    device = resolve_device(trial_cfg)
    run_once = run_dynamic if run_cfg.run_mode == "dynamic" else run_fixed

    seeds = run_seed_averaged(
        run_once=run_once,
        params=run_cfg.params,
        result_dir=trial_result_dir,
        base_seed=int(trial_cfg.seed),
        num_seeds=num_seeds,
        device=device,
    )

    validation_metric, validation_by_seed = _read_seed_metric_file_max(
        trial_result_dir, seeds, "validation"
    )
    trial.set_user_attr("validation_accuracy", validation_metric)
    trial.set_user_attr("validation_accuracy_by_seed", validation_by_seed)
    trial.set_user_attr("result_dir", str(trial_result_dir))
    trial.set_user_attr("seeds", seeds)
    trial.set_user_attr("num_seeds", num_seeds)
    trial.set_user_attr("resolved_params", trial_params)
    return validation_metric


def _run_best_trial_test_evaluation(best_trial, base_cfg: DictConfig, output_root: Path) -> float:
    best_params = _resolved_trial_params(best_trial)
    best_cfg = _copy_dict_config(base_cfg)
    for param_path, sampled_value in best_params.items():
        OmegaConf.update(best_cfg, param_path, sampled_value, merge=False)

    best_result_dir = output_root / "best_trial_test_eval" / "results"
    best_result_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = build_engine_config(best_cfg)
    run_cfg.params["evaluate-test"] = True
    num_seeds = int(best_cfg.num_seeds)
    device = resolve_device(best_cfg)
    run_once = run_dynamic if run_cfg.run_mode == "dynamic" else run_fixed

    seeds = run_seed_averaged(
        run_once=run_once,
        params=run_cfg.params,
        result_dir=best_result_dir,
        base_seed=int(best_cfg.seed),
        num_seeds=num_seeds,
        device=device,
    )

    test_metric, _ = _read_seed_metric_file_max(best_result_dir, seeds, "test")
    return test_metric


def _metrics_list_from_cfg(cfg: DictConfig) -> list[str]:
    raw = cfg.get("plot_metrics")
    if raw is None:
        return []
    values = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(values, list):
        raise ValueError("plot_metrics must be a list")
    return [str(value) for value in values]


@hydra.main(version_base=None, config_path="../../conf", config_name="sweep")
def main(cfg: DictConfig) -> None:
    output_root = Path(HydraConfig.get().runtime.output_dir)
    trials_root = output_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    if "optuna" not in cfg:
        raise ValueError("Missing 'optuna' section in Hydra config")
    optuna_cfg = cfg.optuna
    if "search_space" not in optuna_cfg:
        raise ValueError("Missing 'optuna.search_space' in Hydra config")

    raw_search_space = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
    if not isinstance(raw_search_space, dict) or not raw_search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")
    search_space = {str(path): spec for path, spec in raw_search_space.items()}

    combos = enumerate_valid_param_dicts(cfg, search_space)
    if not combos:
        raise ValueError(
            "No categorical grid combinations found. Use categorical sweeps or add list-style search_space entries."
        )

    _, axis_meta = build_axis_metadata(search_space)
    axis_lookup = {path: axis_meta.get(path, {}) for path in search_space.keys()}

    direction = str(optuna_cfg.direction)
    study = optuna.create_study(direction=direction)
    total_trials = len(combos)
    print(
        f"[optuna] grid trials={total_trials} | seeds_per_trial={int(cfg.num_seeds)} | "
        f"metric=validation_accuracy | trials_dir={trials_root}"
    )
    study.optimize(lambda trial: _objective(trial, cfg, trials_root, axis_lookup, combos), n_trials=total_trials)

    best = study.best_trial
    best_dir = best.user_attrs.get("result_dir")
    print(f"[optuna] best trial: {best.number}")
    print(f"[optuna] best validation_accuracy: {best.value:.6f}")
    if best_dir:
        print(f"[optuna] best result directory: {best_dir}")
    print("[optuna] best parameters:")
    for name, value in _resolved_trial_params(best).items():
        print(f"  - {name}: {value}")

    final_test_accuracy = _run_best_trial_test_evaluation(best, cfg, output_root)
    print(f"[optuna] best trial final test directory: {output_root / 'best_trial_test_eval' / 'results'}")
    print(f"[optuna] best trial final test_accuracy: {final_test_accuracy:.6f}")

    metrics_list = _metrics_list_from_cfg(cfg)
    plot_modes = normalize_plot_modes(cfg.get("plot_mode"))
    plot_directions = normalize_directions(cfg.get("direction"))
    sweep_plot_root = output_root / "sweep_artifacts"
    plot_sweep(
        plot_modes,
        plot_directions,
        trials_root,
        study,
        search_space,
        metrics_list,
        sweep_plot_root,
    )
    print(
        f"[optuna] sweep plots written to: {sweep_plot_root} | "
        f"modes={plot_modes} directions={plot_directions}"
    )


if __name__ == "__main__":
    main()
