"""Baseline schedulers."""

import random

import torch
from torch import nn

from .model import DDQNAgent


def _fitting(cluster, job):
    return [index for index, vm in enumerate(cluster.vms) if vm.can_fit(job)]


class RoundRobin:
    def __init__(self):
        self.position = 0

    def select(self, cluster, job):
        fitting = _fitting(cluster, job)
        if not fitting:
            return None
        for offset in range(len(cluster.vms)):
            candidate = (self.position + offset) % len(cluster.vms)
            if candidate in fitting:
                self.position = (candidate + 1) % len(cluster.vms)
                return candidate
        return None


class RandomScheduler:
    def select(self, cluster, job):
        fitting = _fitting(cluster, job)
        return random.choice(fitting) if fitting else None


class SJF:
    def select(self, cluster, job):
        fitting = _fitting(cluster, job)
        return min(fitting, key=lambda index: cluster.vms[index].total_usage) if fitting else None


class MLPAgent(DDQNAgent):
    def __init__(self, num_vms=30, lr=1e-3):
        super().__init__(num_vms=num_vms, encoder=nn.Identity(), lr=lr)
        self.q_network = nn.Sequential(
            nn.Linear(num_vms * 6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_vms),
        )
        self.target_network = nn.Sequential(
            nn.Linear(num_vms * 6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_vms),
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def _q_values(self, state, adjacency=None):
        return self.q_network(state)


class GNNDQNAgent(DDQNAgent):
    def learn(self, batch, adjacency=None):
        self.target_network = self.q_network
        return super().learn(batch, adjacency)


MLPDDQN = MLPAgent
GNNDQN = GNNDQNAgent
