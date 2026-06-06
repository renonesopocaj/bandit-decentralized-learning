import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.robustness.summations import cs_he, cs_plus, gts
from banditdl.core.topology.gossip import LaplacianGossipMatrix
from banditdl.core.worker.base import HonestWorker
from banditdl.core.worker.config import WorkerConfig

_METHODS = {"cs+": cs_plus, "cs_he": cs_he, "gts": gts}


class FixedGraphWorker(HonestWorker):
    def __init__(
        self,
        worker_id,
        data_loader,
        data_loader_validation,
        config: WorkerConfig,
    ):
        super().__init__(
            worker_id,
            data_loader,
            data_loader_validation,
            config,
        )
        self.comm_graph = config.comm_graph
        self.method = config.method
        self.dissensus = config.dissensus
        self.rho = 1.0
        self.W = torch.tensor(LaplacianGossipMatrix(self.comm_graph)).to(self.device)
        self.nb_neighbors = len(list(self.comm_graph.neighbors(self.worker_id)))

        self.robust_aggregator = RobustAggregator(
            config.aggregator,
            config.pre_aggregator,
            config.server_clip,
            self.nb_neighbors + 1 - self.b_hat,
            self.b_hat,
            config.bucket_size,
            self.model_size,
            self.device,
        )

        metropolis = True
        if metropolis:
            self.byz_weights = 0
            neighbors = list(self.comm_graph.neighbors(self.worker_id))
            neighbors_degrees = sorted([self.comm_graph.degree(i) for i in neighbors])
            pivot_degree = self.comm_graph.degree(self.worker_id)
            for i in range(self.b_hat):
                try:
                    self.byz_weights += 1 / (neighbors_degrees[i] + pivot_degree + 1)
                except Exception:
                    print(
                        f"Warning: b_hat = {self.b_hat} is too large compared to the number of neighbors of worker {self.worker_id}, "
                        f"which is {len(neighbors)}"
                    )
        else:
            self.byz_weights = self.b_hat

        self.num_clipped = []

    def aggregate(self, weights) -> None:
        if len(weights) == 0:
            return None
        pivot_params = self.pull(None)
        neighbor_indices = list(self.comm_graph.neighbors(self.worker_id))
        neighbor_indices.append(self.worker_id)
        with torch.no_grad():
            worker_params = torch.stack(weights)
            differences = worker_params - pivot_params

            robust_aggregate, num_clipped = _METHODS[self.method](
                weights=self.comm_graph.weights(self.worker_id)[neighbor_indices].clone().detach().requires_grad_(False),
                gradients=differences,
                byz_weights=self.byz_weights,
            )
            self.num_clipped.append(num_clipped)
            aggregate_params = pivot_params + self.rho * robust_aggregate
            self.set_model_parameters(aggregate_params)
        return None


FixedGraphP2PWorker = FixedGraphWorker
