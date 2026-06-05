from __future__ import annotations

import copy
import os
import pathlib
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from banditdl.core.sampling import (
    SamplerContext,
    make_neighbor_sampler,
    make_reward_strategy,
)
from banditdl.core.topology.fxgraph import generate_connected_graph
from banditdl.core.topology.graph import CommunicationNetwork
from banditdl.core.worker.byzantine import ByzantineWorker, DecByzantineWorker
from banditdl.core.worker.config import WorkerConfig
from banditdl.core.worker.dynamic import DynamicWorker
from banditdl.core.worker.fixed import FixedGraphWorker
from banditdl.data import dataset
from banditdl.utils.math_utils import consensus_drift, neighbor_disagreement
from banditdl.utils.results import make_result_file, store_result


def _setup_seed(seed: int) -> None:
    reproducible = seed >= 0
    if reproducible:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark = not reproducible


def _progress_interval(rounds: int) -> int:
    return max(1, rounds // 20)


def _should_log_step(current_step: int, rounds: int) -> bool:
    return (
        current_step in (0, rounds) or current_step % _progress_interval(rounds) == 0
    )


def _log_start(mode: str, args, result_dir: pathlib.Path) -> None:
    print(
        f"[banditdl] starting {mode} run: "
        f"dataset={args.dataset}, model={args.model}, nodes={args.nb_workers}, "
        f"honest={args.nb_honests}, byzantine={args.nb_real_byz}, "
        f"rounds={args.rounds}, seed={args.seed}, device={args.device}",
        flush=True,
    )
    print(f"[banditdl] results: {result_dir}", flush=True)


def _log_progress(mode: str, current_step: int, args, accuracy=None, validation_loss=None, train_loss=None) -> None:
    message = f"[banditdl] {mode} round {current_step}/{args.rounds}"
    if accuracy is not None:
        message += f" | mean_accuracy={accuracy:.4f}"
    if validation_loss is not None:
        message += f" | val_loss={validation_loss:.4f}"
    if train_loss is not None:
        message += f" | train_loss={train_loss:.4f}"
    print(message, flush=True)


def _log_done(mode: str) -> None:
    print(f"[banditdl] finished {mode} run", flush=True)


def _raise_if_nonfinite_weights(workers, current_step: int, mode: str) -> None:
    for worker in workers:
        weights = worker.pull(None)
        if not torch.isfinite(weights).all():
            raise FloatingPointError(
                f"{mode} produced non-finite weights at round {current_step} "
                f"for worker {worker.worker_id}"
            )


def _record_evaluation(
    workers,
    fd_validation,
    fd_validation_loss,
    fd_train_loss,
    current_step,
    validation_steps,
    validation_accuracies,
    validation_losses,
    train_losses,
):
    accs = [w.compute_validation_accuracy() for w in workers]
    validation_losses_round = [w.compute_validation_loss() for w in workers]
    train_losses_round = [w.compute_train_loss() for w in workers]
    mean_validation_accuracy = sum(accs) / len(accs)
    mean_validation_loss = sum(validation_losses_round) / len(validation_losses_round)
    mean_train_loss = sum(train_losses_round) / len(train_losses_round)

    validation_steps.append(current_step)
    validation_accuracies.append(accs)
    validation_losses.append(validation_losses_round)
    train_losses.append(train_losses_round)
    store_result(fd_validation, current_step, mean_validation_accuracy)
    store_result(fd_validation_loss, current_step, mean_validation_loss)
    store_result(fd_train_loss, current_step, mean_train_loss)

    return mean_validation_accuracy, mean_validation_loss, mean_train_loss


def _record_final_evaluation_if_needed(
    args,
    workers,
    fd_validation,
    fd_validation_loss,
    fd_train_loss,
    validation_steps,
    validation_accuracies,
    validation_losses,
    train_losses,
):
    if args.evaluation_delta <= 0:
        return None, None, None
    if validation_steps and validation_steps[-1] == args.rounds:
        return None, None, None
    return _record_evaluation(
        workers,
        fd_validation,
        fd_validation_loss,
        fd_train_loss,
        args.rounds,
        validation_steps,
        validation_accuracies,
        validation_losses,
        train_losses,
    )


class ResultTracker:
    """Consolidates metrics tracking, evaluation, and saving results."""

    def __init__(self, args, result_dir: pathlib.Path, test_loader=None):
        self.args = args
        self.result_dir = result_dir
        self.test_loader = test_loader
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.fd_validation = (result_dir / "validation").open("w")
        self.fd_validation_worst = (result_dir / "validation_worst").open("w")
        self.fd_validation_loss = (result_dir / "validation_loss").open("w")
        self.fd_train_loss = (result_dir / "train_loss").open("w")

        make_result_file(self.fd_validation, ["Step number", "Cross-accuracy"])
        make_result_file(self.fd_validation_worst, ["Step number", "Cross-accuracy"])
        make_result_file(self.fd_validation_loss, ["Step number", "Cross-loss"])
        make_result_file(self.fd_train_loss, ["Step number", "Cross-loss"])

        self.validation_steps = []
        self.validation_accuracies = []
        self.validation_losses = []
        self.train_losses = []
        self.neighbor_disagreement_history = []
        self.consensus_drift_history = []
        self.gradient_norm_history = []

    def evaluate_step(self, current_step, workers, mode):
        mean_acc = None
        mean_val_loss = None
        mean_train_loss = None

        if self.args.evaluation_delta > 0 and current_step % self.args.evaluation_delta == 0:
            (
                mean_acc,
                mean_val_loss,
                mean_train_loss,
            ) = _record_evaluation(
                workers,
                self.fd_validation,
                self.fd_validation_loss,
                self.fd_train_loss,
                current_step,
                self.validation_steps,
                self.validation_accuracies,
                self.validation_losses,
                self.train_losses,
            )

        if _should_log_step(current_step, self.args.rounds):
            _log_progress(
                mode,
                current_step,
                self.args,
                mean_acc,
                mean_val_loss,
                mean_train_loss,
            )
        return mean_acc, mean_val_loss, mean_train_loss

    def record_gradient_norms(self, workers):
        self.gradient_norm_history.append(
            np.array([w.last_gradient_norm for w in workers], dtype=float)
        )

    def record_drift(self, disagreement, consensus):
        self.neighbor_disagreement_history.append(disagreement.cpu().numpy())
        self.consensus_drift_history.append(consensus.cpu().numpy())

    def finalize(self, workers, mode):
        final_accuracy, final_val_loss, final_train_loss = _record_final_evaluation_if_needed(
            self.args,
            workers,
            self.fd_validation,
            self.fd_validation_loss,
            self.fd_train_loss,
            self.validation_steps,
            self.validation_accuracies,
            self.validation_losses,
            self.train_losses,
        )
        if _should_log_step(self.args.rounds, self.args.rounds):
            _log_progress(
                mode,
                self.args.rounds,
                self.args,
                final_accuracy,
                final_val_loss,
                final_train_loss,
            )

        if self.validation_accuracies:
            worst_idx = min(range(len(workers)), key=lambda i: self.validation_accuracies[-1][i])
            for step, accs in zip(self.validation_steps, self.validation_accuracies, strict=True):
                store_result(self.fd_validation_worst, step, accs[worst_idx])

        if self.args.evaluate_test and self.test_loader:
            fd_test = (self.result_dir / "test").open("w")
            make_result_file(fd_test, ["Step number", "Cross-accuracy"])
            test_accuracies = [w.compute_accuracy_on_loader(self.test_loader) for w in workers]
            store_result(fd_test, self.args.rounds, sum(test_accuracies) / len(test_accuracies))
            fd_test.close()

        self.fd_validation.close()
        self.fd_validation_worst.close()
        self.fd_validation_loss.close()
        self.fd_train_loss.close()

        np.save(os.path.join(self.result_dir, "validation_accuracies.npy"), np.array(self.validation_accuracies))
        np.save(os.path.join(self.result_dir, "validation_losses.npy"), np.array(self.validation_losses))
        np.save(os.path.join(self.result_dir, "train_losses.npy"), np.array(self.train_losses))
        np.save(os.path.join(self.result_dir, "neighbor_disagreement.npy"), np.array(self.neighbor_disagreement_history))
        np.save(os.path.join(self.result_dir, "consensus_drift.npy"), np.array(self.consensus_drift_history))
        np.save(os.path.join(self.result_dir, "gradient_norms.npy"), np.array(self.gradient_norm_history))

        with torch.no_grad():
            final_weights = torch.stack([w.pull(None) for w in workers])
            pairwise_distance_final = torch.cdist(final_weights, final_weights).cpu().numpy()
        np.save(os.path.join(self.result_dir, "pairwise_model_distance_final.npy"), pairwise_distance_final)
        _log_done(mode)


def _make_args(
    params: dict, result_dir: pathlib.Path, seed: int, device: str
) -> SimpleNamespace:
    args = dict(params)
    args.setdefault("hetero", False)
    args.setdefault("distinct-data", False)
    args.setdefault("dirichlet-alpha", None)
    args.setdefault("nb-datapoints", None)
    args.setdefault("numb-labels", None)
    args.setdefault("batch-size-test", 100)
    args.setdefault("loss", "NLLLoss")
    args.setdefault("weight-decay", 0)
    args.setdefault("momentum-worker", 0.99)
    args.setdefault("bucket-size", 1)
    args.setdefault("pre-aggregator", None)
    args.setdefault("aggregator", "average")
    args.setdefault("learning-rate", 0.5)
    args.setdefault("learning-rate-decay", 5000)
    args.setdefault("learning-rate-decay-delta", 1)
    args.setdefault("mimic-learning-phase", None)
    args.setdefault("gradient-clip", None)
    args.setdefault("server-clip", False)
    args.setdefault("rag", False)
    args.setdefault("method", "cs+")
    args.setdefault("attack", None)
    args.setdefault("neighbor-sampler", "uniform")
    args.setdefault("sampler-params", {})
    args.setdefault("sampler-reward", "parameter_distance")
    args.setdefault("bandit-epsilon", 0.1)
    args.setdefault("bandit-initial-value", 0.0)
    args.setdefault("bandit-reward", "parameter_distance")
    args.setdefault("global-test-ratio", 0.1)
    args.setdefault("local-test-ratio", 0.2)
    args.setdefault("eval-split-seed", 0)
    args.setdefault("evaluate-test", False)
    args.setdefault("partition-method", "dirichlet")
    args.setdefault("partition-style", None)
    args.setdefault("classes-per-worker", None)
    args.setdefault("nb-shards", None)
    args.setdefault("shards-per-worker", None)
    args.setdefault("nb-groups", None)
    args.setdefault("classes-per-group", None)
    args.setdefault("group-overlap", 0)
    args.setdefault("dataset-mode", None)
    args.setdefault("nb-writers-limit", None)
    args["result-directory"] = str(result_dir)
    args["seed"] = seed
    args["device"] = device
    # normalize dashed keys for existing code style
    normalized = {k.replace("-", "_"): v for k, v in args.items()}
    if "rounds" not in normalized and "nb_steps" in normalized:
        normalized["rounds"] = normalized["nb_steps"]
    normalized["nb_honests"] = normalized["nb_workers"] - normalized["nb_real_byz"]
    return SimpleNamespace(**normalized)


def _build_worker_config(args) -> WorkerConfig:
    """Map the args namespace/dict to a structured WorkerConfig."""
    # Handle both SimpleNamespace and dict for transition
    def get_val(key, default=None):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    return WorkerConfig(
        model=get_val("model"),
        learning_rate=get_val("learning_rate"),
        learning_rate_decay=get_val("learning_rate_decay"),
        learning_rate_decay_delta=get_val("learning_rate_decay_delta"),
        weight_decay=get_val("weight_decay"),
        loss=get_val("loss"),
        momentum=get_val("momentum_worker"),
        device=get_val("device"),
        nb_local_steps=get_val("nb_local_steps"),
        nb_workers=get_val("nb_workers"),
        nb_byz=get_val("nb_decl_byz"),
        nb_real_byz=get_val("nb_real_byz"),
        b_hat=get_val("b_hat"),
        rag=get_val("rag"),
        numb_labels=get_val("numb_labels"),
        labelflipping=get_val("attack") == "LF",
        gradient_clip=get_val("gradient_clip"),
        server_clip=get_val("server_clip"),
        bucket_size=get_val("bucket_size"),
        aggregator=get_val("aggregator"),
        pre_aggregator=get_val("pre_aggregator"),
        nb_neighbors=get_val("nb_neighbors"),
        sampling_ratio=get_val("sampling_ratio"),
        mimic_learning_phase=get_val("mimic_learning_phase"),
        method=get_val("method"),
        epsilon=get_val("epsilon", 1.0),
    )


def _init_workers(args, train_loader_dict, local_test_loader_dict, comm_graph=None, dissensus=False):
    workers = []
    base_config = _build_worker_config(args)

    if comm_graph is not None:
        # Fixed graph mode
        for worker_id in range(args.nb_honests):
            config = replace(base_config, comm_graph=comm_graph, dissensus=dissensus)
            w = FixedGraphWorker(worker_id, train_loader_dict[worker_id], local_test_loader_dict[worker_id], config)
            if worker_id > 0:
                w.model.load_state_dict(workers[0].model.state_dict())
            workers.append(w)
    else:
        # Dynamic sampling mode
        for worker_id in range(args.nb_honests):
            sampler_params = dict(args.sampler_params or {})
            if args.neighbor_sampler in {"bandit", "epsilon_greedy"}:
                sampler_params.setdefault("epsilon", args.bandit_epsilon)
                sampler_params.setdefault("initial_value", args.bandit_initial_value)

            sampler_context = SamplerContext(
                worker_id=worker_id,
                nodes=args.nb_workers,
                k=args.nb_neighbors,
                horizon=args.rounds + 1,
                seed=args.seed + worker_id,
            )
            neighbor_sampler = make_neighbor_sampler(args.neighbor_sampler, context=sampler_context, params=sampler_params)
            reward_strategy = make_reward_strategy(args.sampler_reward)

            config = replace(base_config, neighbor_sampler=neighbor_sampler, reward_strategy=reward_strategy)
            w = DynamicWorker(worker_id, train_loader_dict[worker_id], local_test_loader_dict[worker_id], config)
            if worker_id > 0:
                w.model.load_state_dict(workers[0].model.state_dict())
            workers.append(w)

    return workers


def _best_fixed_subset(scores, worker_id: int, k: int):
    scores = np.asarray(scores, dtype=float)
    candidates = [i for i in range(len(scores)) if i != worker_id]
    selected = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
    reward = 0.0 if not selected else float(scores[selected].sum() / len(selected))
    return np.array(selected, dtype=int), reward


def _mean_selected_reward(rewards) -> float:
    rewards = list(rewards)
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def _dynamic_candidate_weights(w, honest_weights, byz_workers, current_step):
    candidate_weights = {
        worker_id: weight
        for worker_id, weight in enumerate(honest_weights)
        if worker_id != w.worker_id
    }
    context = {"honest_weights": honest_weights, "step": current_step}
    for byz_worker in byz_workers:
        weight = copy.deepcopy(byz_worker).pull(context)
        if weight is not None:
            candidate_weights[byz_worker.worker_id] = weight
    return candidate_weights


def _full_sampler_probability_vector(worker, nb_total: int) -> np.ndarray:
    """Return the full per-arm sampler probability row for `worker`.

    Length is `nb_total`; the worker's own slot is 0 (it cannot pick itself).
    """
    population = [i for i in range(nb_total) if i != worker.worker_id]
    probabilities_by_arm = worker.neighbor_sampler.probabilities(
        population,
        worker.nb_neighbors,
    )
    row = np.zeros(nb_total, dtype=float)
    for arm in population:
        row[arm] = float(probabilities_by_arm[arm])
    return row


def _sampler_probability_stats(worker) -> tuple[float, float, float]:
    population = list(range(worker.nb_honest + worker.nb_byz))
    population.remove(worker.worker_id)
    probabilities_by_arm = worker.neighbor_sampler.probabilities(
        population,
        worker.nb_neighbors,
    )
    probabilities = np.array(
        [probabilities_by_arm[arm] for arm in population],
        dtype=float,
    )
    uniform_probability = 1.0 / len(population)
    kl_to_uniform = float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12) / uniform_probability))
    )
    return kl_to_uniform, float(probabilities.min()), float(probabilities.max())


def _run_experiment(params: dict, result_dir: pathlib.Path, seed: int, device: str, mode: str) -> None:
    args = _make_args(params, result_dir, seed, device)
    _setup_seed(args.seed)
    _log_start(mode, args, result_dir)

    train_loader_dict, local_test_loader_dict, test_loader = dataset.make_train_validation_test_datasets(
        args.dataset,
        heterogeneity=args.hetero,
        numb_labels=args.numb_labels,
        alpha_dirichlet=args.dirichlet_alpha,
        distinct_datasets=args.distinct_data,
        nb_datapoints=args.nb_datapoints,
        honest_workers=args.nb_honests,
        train_batch=args.batch_size,
        test_batch=args.batch_size_test,
        global_test_ratio=args.global_test_ratio,
        local_test_ratio=args.local_test_ratio,
        split_seed=args.eval_split_seed,
        partition_method=args.partition_method,
        partition_style=args.partition_style,
        classes_per_worker=args.classes_per_worker,
        nb_shards=args.nb_shards,
        shards_per_worker=args.shards_per_worker,
        nb_groups=args.nb_groups,
        classes_per_group=args.classes_per_group,
        group_overlap=args.group_overlap,
        dataset_mode=args.dataset_mode,
        nb_writers_limit=args.nb_writers_limit,
    )

    comm_graph = None
    adjacency_honest = None
    if mode == "fixed":
        nb_edges = args.nb_workers * args.nb_neighbors // 2
        g = generate_connected_graph(args.nb_workers, nb_edges, seed=args.seed)
        comm_graph = CommunicationNetwork(g, weights_method="metropolis", device=args.device if args.device != "auto" else "cpu")
        adjacency_honest = torch.as_tensor(np.asarray(comm_graph.adjacency_matrix[: args.nb_honests, : args.nb_honests]), dtype=torch.float32, device=args.device)

    workers = _init_workers(args, train_loader_dict, local_test_loader_dict, comm_graph=comm_graph, dissensus=(args.attack == "dissensus"))
    base_config = _build_worker_config(args)

    byz_workers = [ByzantineWorker(i, workers[0].model_size, base_config) for i in range(args.nb_honests, args.nb_workers)]
    byz_workers_by_id = {byz.worker_id: byz for byz in byz_workers}
    dec_byz_workers = {i: DecByzantineWorker(i, args.nb_honests, base_config) for i in range(args.nb_honests, args.nb_workers)} if mode == "fixed" else {}

    tracker = ResultTracker(args, result_dir, test_loader)

    # Dynamic-specific tracking
    cumulative_arm_rewards = np.zeros((args.nb_honests, args.nb_workers))
    cumulative_algorithm_rewards = np.zeros(args.nb_honests)
    algorithm_reward_history, oracle_reward_history, selected_neighbor_history, oracle_neighbor_history = [], [], [], []
    sampler_kl_history, sampler_min_prob_history, sampler_max_prob_history = [], [], []
    reward_min_history, reward_max_history = [], []

    for current_step in range(args.rounds + 1):
        tracker.evaluate_step(current_step, workers, mode)

        for w in workers:
            w.train()
        tracker.record_gradient_norms(workers)
        honest_weights = [w.pull(None) for w in workers]

        if mode == "dynamic":
            selected_round = np.full((args.nb_honests, workers[0].nb_neighbors), -1, dtype=int)
            kl_round, min_prob_round, max_prob_round = np.zeros(args.nb_honests), np.zeros(args.nb_honests), np.zeros(args.nb_honests)
            reward_min_round, reward_max_round = np.full(args.nb_honests, np.nan), np.full(args.nb_honests, np.nan)

            for w in workers:
                kl_round[w.worker_id], min_prob_round[w.worker_id], max_prob_round[w.worker_id] = _sampler_probability_stats(w)
                neighbor_indices = w._sample_neighbors()
                candidate_weights = _dynamic_candidate_weights(w, honest_weights, byz_workers, current_step)
                selected_ids = [i for i in neighbor_indices if i in candidate_weights]

                for nid in selected_ids:
                    if nid >= args.nb_honests:
                        weight = byz_workers_by_id[nid].pull({"honest_weights": honest_weights, "step": current_step})
                        if weight is not None:
                            candidate_weights[nid] = weight

                candidate_ids = list(candidate_weights)
                candidate_rewards = w.reward_strategy.score(w.pull(None), [candidate_weights[i] for i in candidate_ids])
                rewards_by_id = dict(zip(candidate_ids, candidate_rewards, strict=True))
                cumulative_arm_rewards[w.worker_id, candidate_ids] += candidate_rewards

                neighbor_weights = [candidate_weights[i] for i in selected_ids]
                selected_round[w.worker_id, :len(selected_ids)] = selected_ids
                selected_rewards = [rewards_by_id[i] for i in selected_ids]
                if selected_rewards:
                    reward_min_round[w.worker_id], reward_max_round[w.worker_id] = min(selected_rewards), max(selected_rewards)
                    cumulative_algorithm_rewards[w.worker_id] += _mean_selected_reward(selected_rewards)

                w.num_selected_byz.append(len([i for i in selected_ids if i >= args.nb_honests]))
                w.observe_neighbors(selected_ids, neighbor_weights)
                w.aggregate(neighbor_weights)

            sampler_kl_history.append(kl_round)
            sampler_min_prob_history.append(min_prob_round)
            sampler_max_prob_history.append(max_prob_round)
            reward_min_history.append(reward_min_round)
            reward_max_history.append(reward_max_round)

            oracle_neighbors_round, oracle_rewards_round = [], []
            for w in workers:
                oids, oreward = _best_fixed_subset(cumulative_arm_rewards[w.worker_id], worker_id=w.worker_id, k=w.nb_neighbors)
                oracle_neighbors_round.append(oids)
                oracle_rewards_round.append(oreward)
            algorithm_reward_history.append(cumulative_algorithm_rewards.copy())
            oracle_reward_history.append(np.array(oracle_rewards_round))
            selected_neighbor_history.append(selected_round)
            oracle_neighbor_history.append(np.stack(oracle_neighbors_round))

            with torch.no_grad():
                updated = [w.pull(None) for w in workers]
                neighbor_matrix = selected_round.copy()
                neighbor_matrix[neighbor_matrix >= args.nb_honests] = -1
                tracker.record_drift(neighbor_disagreement(updated, neighbor_indices=neighbor_matrix.tolist()), consensus_drift(updated))
        else:
            # Fixed mode
            for w in workers:
                neighbors = [*list(w.comm_graph.neighbors(w.worker_id)), w.worker_id]
                honest_nids = [i for i in neighbors if i < args.nb_honests]
                byz_nids = [i for i in neighbors if i >= args.nb_honests]
                w.num_selected_byz.append(len(byz_nids))
                h_weights = [honest_weights[i] for i in honest_nids]
                if args.attack == "dissensus":
                    b_weights = [dec_byz_workers[i].pull({"target": w.worker_id, "honest_neighbors": honest_nids, "pivot_params": w.pull(None), "honest_local_params": h_weights}) for i in byz_nids]
                else:
                    b_weights = [byz_workers_by_id[i].pull({"honest_weights": honest_weights, "step": current_step}) for i in byz_nids]
                w.aggregate(h_weights + b_weights)

            with torch.no_grad():
                updated = [w.pull(None) for w in workers]
                tracker.record_drift(neighbor_disagreement(updated, adjacency=adjacency_honest), consensus_drift(updated))

        _raise_if_nonfinite_weights(workers, current_step, mode)

    tracker.finalize(workers, mode)
    if mode == "dynamic":
        np.save(result_dir / "reward_algorithm.npy", np.array(algorithm_reward_history))
        np.save(result_dir / "reward_oracle.npy", np.array(oracle_reward_history))
        np.save(result_dir / "regret.npy", np.array(oracle_reward_history) - np.array(algorithm_reward_history))
        np.save(result_dir / "reward_selected_min.npy", np.array(reward_min_history))
        np.save(result_dir / "reward_selected_max.npy", np.array(reward_max_history))
        np.save(result_dir / "selected_neighbors.npy", np.array(selected_neighbor_history, dtype=int))
        np.save(result_dir / "oracle_neighbors.npy", np.array(oracle_neighbor_history, dtype=int))
        np.save(result_dir / "sampler_kl_to_uniform.npy", np.array(sampler_kl_history))
        np.save(result_dir / "sampler_min_probability.npy", np.array(sampler_min_prob_history))
        np.save(result_dir / "sampler_max_probability.npy", np.array(sampler_max_prob_history))
        sampler_probs_final = np.stack([_full_sampler_probability_vector(w, args.nb_workers) for w in workers])
        np.save(result_dir / "sampler_probabilities_final.npy", sampler_probs_final)


def run_dynamic(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> None:
    _run_experiment(params, result_dir, seed, device, "dynamic")


def run_fixed(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> None:
    _run_experiment(params, result_dir, seed, device, "fixed")
