from __future__ import annotations

import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.config_schema import BanditDLConfig
from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.plotting import plot_all
from banditdl.utils.seed_averaging import run_seed_averaged


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Use the structured merge pattern for type safety
    merged = OmegaConf.merge(OmegaConf.structured(BanditDLConfig), cfg)
    OmegaConf.resolve(merged)
    config: BanditDLConfig = OmegaConf.to_object(merged)

    print("\n" + OmegaConf.to_yaml(config) + "\n")

    run_cfg = build_engine_config(merged) # Pass the merged DictConfig to allow to_object internally
    device = resolve_device(cfg)

    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    result_dir = output_dir / "results"
    run_once = run_dynamic if run_cfg.run_mode == "dynamic" else run_fixed

    run_seed_averaged(
        run_once=run_once,
        config=config,
        result_dir=result_dir,
        base_seed=config.seed,
        num_seeds=config.num_seeds,
        device=device,
    )

    plot_all(
        run_dir=result_dir,
        plots_dir=output_dir / "plots",
        run_label=run_cfg.run_name,
    )


if __name__ == "__main__":
    main()
