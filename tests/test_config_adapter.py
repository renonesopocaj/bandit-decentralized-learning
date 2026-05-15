import pytest
from omegaconf import OmegaConf

from banditdl.experiments.config_adapter import build_engine_config


def _base_cfg():
    return OmegaConf.create(
        {
            "dataset": {"dataset": "mnist", "model": "cnn_mnist"},
            "topology": {"nodes": 30, "sampling": 0.2},
            "sampler": {
                "name": "epsilon_greedy",
                "reward": "parameter_distance",
                "params": {"epsilon": 0.1, "initial_value": 0.0},
            },
            "adversary": {"byzcount": 0, "byzantine_budget": 0, "attack": None},
            "aggregator": {"pre-aggregator": "nnm", "aggregator": "average", "rag": True},
            "heterogeneity": {"alpha": 0.5, "numb_labels": 10},
            "optimization": {
                "batch_size": 25,
                "loss": "NLLLoss",
                "weight_decay": 1e-4,
                "momentum_worker": 0.9,
                "rounds": 200,
                "nb_local_steps": 1,
            },
            "evaluation": {"evaluation_delta": 20},
            "seed": 123,
            "device": "cpu",
        }
    )


def test_build_dynamic_engine_config_uses_sampler_group():
    run_cfg = build_engine_config(_base_cfg())

    assert run_cfg.run_mode == "dynamic"
    assert run_cfg.nb_neighbors == 6
    assert run_cfg.params["neighbor-sampler"] == "epsilon_greedy"
    assert run_cfg.params["sampler-params"] == {"epsilon": 0.1, "initial_value": 0.0}
    assert run_cfg.params["rounds"] == 200


def test_build_fixed_engine_config_uses_topology_method():
    cfg = _base_cfg()
    del cfg.topology.sampling
    cfg.topology.degree = 15
    cfg.topology.method = "cs+"

    run_cfg = build_engine_config(cfg)

    assert run_cfg.run_mode == "fixed"
    assert run_cfg.nb_neighbors == 15
    assert run_cfg.params["method"] == "cs+"


def test_build_engine_config_dirichlet_default_method():
    run_cfg = build_engine_config(_base_cfg())

    assert run_cfg.params["partition-method"] == "dirichlet"
    assert run_cfg.params["dirichlet-alpha"] == 0.5
    assert run_cfg.params["classes-per-worker"] is None
    assert run_cfg.params["shards-per-worker"] is None
    assert "alpha_0.5" in run_cfg.run_name


def test_build_engine_config_pathological_classes_per_worker():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create({
        "method": "pathological",
        "partition": "classes_per_worker",
        "classes_per_worker": 2,
        "numb_labels": 10,
        "alpha": None,
    })

    run_cfg = build_engine_config(cfg)

    assert run_cfg.params["partition-method"] == "pathological"
    assert run_cfg.params["partition-style"] == "classes_per_worker"
    assert run_cfg.params["classes-per-worker"] == 2
    assert run_cfg.params["dirichlet-alpha"] is None
    assert "pathological_c_2" in run_cfg.run_name


def test_build_engine_config_pathological_shards_per_worker():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create({
        "method": "pathological",
        "partition": "shards_per_worker",
        "shards_per_worker": 2,
        "nb_shards": None,
        "numb_labels": 10,
        "alpha": None,
    })

    run_cfg = build_engine_config(cfg)

    assert run_cfg.params["partition-method"] == "pathological"
    assert run_cfg.params["partition-style"] == "shards_per_worker"
    assert run_cfg.params["shards-per-worker"] == 2
    assert run_cfg.params["nb-shards"] is None


def test_build_engine_config_pathological_grouped_classes():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create({
        "method": "pathological",
        "partition": "grouped_classes",
        "nb_groups": 5,
        "classes_per_group": 2,
        "group_overlap": 0,
        "numb_labels": 10,
        "alpha": None,
    })

    run_cfg = build_engine_config(cfg)

    assert run_cfg.params["partition-method"] == "pathological"
    assert run_cfg.params["partition-style"] == "grouped_classes"
    assert run_cfg.params["nb-groups"] == 5
    assert run_cfg.params["classes-per-group"] == 2
    assert run_cfg.params["group-overlap"] == 0
    assert "pathological_g_5x2" in run_cfg.run_name


def test_build_engine_config_pathological_grouped_classes_with_overlap_token():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create({
        "method": "pathological",
        "partition": "grouped_classes",
        "nb_groups": 3,
        "classes_per_group": 3,
        "group_overlap": 1,
        "numb_labels": 10,
        "alpha": None,
    })

    run_cfg = build_engine_config(cfg)

    assert run_cfg.params["group-overlap"] == 1
    assert "pathological_g_3x3_ov_1" in run_cfg.run_name


def test_build_engine_config_grouped_classes_rejects_nodes_lt_groups():
    cfg = _base_cfg()
    cfg.topology.nodes = 3
    cfg.heterogeneity = OmegaConf.create({
        "method": "pathological",
        "partition": "grouped_classes",
        "nb_groups": 5,
        "classes_per_group": 2,
        "numb_labels": 10,
        "alpha": None,
    })

    try:
        build_engine_config(cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError when topology.nodes < nb_groups")


def test_build_engine_config_femnist_writer_mode_bypasses_alpha():
    cfg = _base_cfg()
    cfg.dataset = OmegaConf.create({
        "dataset": "femnist",
        "model": "cnn_femnist",
        "numb_labels": 62,
        "mode": "writer_per_node",
        "nb_writers_limit": None,
    })
    cfg.heterogeneity = OmegaConf.create({
        "method": "dirichlet",
        "alpha": None,
        "numb_labels": 10,
    })

    run_cfg = build_engine_config(cfg)

    assert run_cfg.params["dataset"] == "femnist"
    assert run_cfg.params["dataset-mode"] == "writer_per_node"
    assert run_cfg.params["numb-labels"] == 62
    assert run_cfg.params["dirichlet-alpha"] is None
    assert "femnist_writers" in run_cfg.run_name


def test_build_engine_config_femnist_pool_mode_uses_heterogeneity():
    cfg = _base_cfg()
    cfg.dataset = OmegaConf.create({
        "dataset": "femnist",
        "model": "cnn_femnist",
        "numb_labels": 62,
        "mode": "pool",
        "nb_writers_limit": None,
    })

    run_cfg = build_engine_config(cfg)

    assert run_cfg.params["dataset-mode"] == "pool"
    assert run_cfg.params["dirichlet-alpha"] == 0.5
    assert run_cfg.params["numb-labels"] == 62
    assert "alpha_0.5" in run_cfg.run_name


def test_build_engine_config_writer_mode_rejects_non_femnist():
    cfg = _base_cfg()
    cfg.dataset.mode = "writer_per_node"

    with pytest.raises(ValueError):
        build_engine_config(cfg)


def test_build_engine_config_pathological_rejects_missing_partition_style():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create({
        "method": "pathological",
        "numb_labels": 10,
        "alpha": None,
    })

    try:
        build_engine_config(cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError when partition style is missing")
