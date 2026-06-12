###
 # @file   models.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Neural network architectures used by the experiments. One model per dataset,
 # selected by name from the dataset config (`conf/dataset/*.yaml`) and built in
 # `banditdl.core.worker.base` via `getattr(models, config.model)()`.
###

import torch
import torch.nn.functional as F
from torch import nn


# MNIST: 2-conv / 2-FC, ~431K parameters (Appendix A, Table II).
class cnn_mnist(nn.Module):
    """Small convolutional network for MNIST."""

    def __init__(self):
        super().__init__()
        self._c1 = nn.Conv2d(1, 20, 5, 1)
        self._c2 = nn.Conv2d(20, 50, 5, 1)
        self._f1 = nn.Linear(800, 500)
        self._f2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self._c1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._c2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._f1(x.view(-1, 800)))
        return F.log_softmax(self._f2(x), dim=1)


# FEMNIST: 2-conv / 2-FC, 62 classes, ~2.37M parameters (Appendix A, Table II).
class cnn_femnist(nn.Module):
    """Small convolutional network for FEMNIST."""

    def __init__(self):
        super().__init__()
        self._c1 = nn.Conv2d(1, 64, 5, 1)
        self._c2 = nn.Conv2d(64, 128, 5, 1)
        self._f1 = nn.Linear(128 * 4 * 4, 1024)
        self._f2 = nn.Linear(1024, 62)

    def forward(self, x):
        x = F.relu(self._c1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._c2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._f1(x.view(-1, 128 * 4 * 4)))
        return F.log_softmax(self._f2(x), dim=1)


# CIFAR-10: 3-conv / 3-FC, same-padding + max-pool per block, ~2.32M parameters
# (Appendix A, Table II). Outputs logits (use with CrossEntropyLoss).
class cnn_cifar(nn.Module):
    """Convolutional network for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 100, 5, padding=2)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 200, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.conv3.out_channels * 4 * 4, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.fc3 = nn.Linear(self.fc2.out_features, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
