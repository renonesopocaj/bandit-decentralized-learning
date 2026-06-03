from __future__ import annotations

import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.seed_averaging import run_seed_averaged
from banditdl.utils.plotting import plot_all


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n")
    run_cfg = build_engine_config(cfg)
    device = resolve_device(cfg)

    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    result_dir = output_dir / "results"
    run_once = run_dynamic if run_cfg.run_mode == "dynamic" else run_fixed
    run_seed_averaged(
        run_once=run_once,
        params=run_cfg.params,
        result_dir=result_dir,
        base_seed=int(cfg.seed),
        num_seeds=int(cfg.num_seeds),
        device=device,
    )

    plot_all(
        run_dir=result_dir,
        plots_dir=output_dir / "plots",
        run_label=run_cfg.run_name,
    )


if __name__ == "__main__":
    main()
