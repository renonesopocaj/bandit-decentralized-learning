"""Byzantine worker implementations for distributed training."""

import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.robustness.attacks import ByzantineAttack
from banditdl.core.worker.base import BaseWorker
from banditdl.core.worker.config import WorkerConfig


class ByzantineWorker(BaseWorker):
    """Byzantine participant implementing the worker API."""

    def __init__(
        self,
        worker_id,
        model_size,
        config: WorkerConfig,
    ):
        super().__init__(worker_id=worker_id, is_byzantine=True)
        robust_aggregator = RobustAggregator(
            config.aggregator,
            config.pre_aggregator,
            config.server_clip,
            config.nb_workers,
            config.nb_byz,
            config.bucket_size,
            model_size,
            config.device,
        )
        self.byzantine_attack = ByzantineAttack(
            config.attack if config.attack else "SF",
            config.nb_real_byz,
            model_size,
            config.device,
            config.mimic_learning_phase,
            config.gradient_clip,
            robust_aggregator,
        )

    def train(self) -> None:
        return None

    def aggregate(self, weights) -> None:
        return None

    def pull(self, context):
        if context is None:
            return None
        honest_vectors = context.get("honest_weights", [])
        current_step = context.get("step", 0)
        vectors = self.byzantine_attack.generate_byzantine_vectors(honest_vectors, None, current_step)
        return vectors[0] if len(vectors) > 0 else None

    def compute_validation_accuracy(self):
        return None

    def compute_validation_loss(self):
        return None

    def compute_train_loss(self):
        return None


class DecByzantineWorker(BaseWorker):
    """Decentralized Byzantine participant for fixed-graph dissensus."""

    def __init__(self, worker_id: int, nb_honest: int, config: WorkerConfig):
        super().__init__(worker_id=worker_id, is_byzantine=True)
        self.nb_honest = nb_honest
        self.network = config.comm_graph
        self.device = config.device
        self.epsilon = config.epsilon

    def train(self) -> None:
        return None

    def aggregate(self, weights) -> None:
        return None

    def pull(self, context):
        target = context["target"]
        honest_neighbors = context["honest_neighbors"]
        pivot_params = context["pivot_params"]
        honest_local_params = context["honest_local_params"]
        W_i = self.network.weights(target)
        byz_neighbors = [k for k in self.network.neighbors(target) if k >= self.nb_honest]
        total_byz_weights = W_i[byz_neighbors].sum()
        honest_local_params = torch.stack(honest_local_params)
        differences = honest_local_params - pivot_params
        byzantine_vector = pivot_params - self.epsilon / total_byz_weights * torch.matmul(
            (W_i[honest_neighbors]).unsqueeze(0), differences
        )
        return byzantine_vector.squeeze(0)

    def compute_validation_accuracy(self):
        return None

    def compute_validation_loss(self):
        return None

    def compute_train_loss(self):
        return None
