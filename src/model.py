"""Graph encoders and deep Q-learning agents."""

from collections import deque
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn


def build_adjacency_batch(
    state: torch.Tensor | np.ndarray,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> torch.Tensor:
    """Compute normalized adjacency matrices for a batch of states (batch, N, 6) -> (batch, N, N)."""
    if isinstance(state, np.ndarray):
        state = torch.as_tensor(state, dtype=torch.float32)
    else:
        state = state.float()

    utilization = state[..., :3]
    distance = torch.abs(utilization.unsqueeze(2) - utilization.unsqueeze(1))
    denom = 1.0 + alpha * distance[..., 0] + beta * distance[..., 1] + gamma * distance[..., 2]
    adjacency = 1.0 / denom

    batch_size, n, _ = state.shape
    eye = torch.eye(n, dtype=adjacency.dtype, device=adjacency.device).unsqueeze(0)
    adjacency = adjacency * (1.0 - eye) + eye

    degree = adjacency.sum(dim=-1)
    degree_outer = degree.unsqueeze(-1) * degree.unsqueeze(-2)
    normalized = adjacency / torch.sqrt(degree_outer.clamp(min=1e-12))
    return normalized


def build_adjacency(
    state: np.ndarray | torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> torch.Tensor:
    if (isinstance(state, torch.Tensor) and state.ndim == 3) or (isinstance(state, np.ndarray) and state.ndim == 3):
        return build_adjacency_batch(state, alpha, beta, gamma)
    state = np.asarray(state, dtype=np.float32)
    utilization = state[:, :3]
    distance = np.abs(utilization[:, None, :] - utilization[None, :, :])
    adjacency = 1.0 / (1.0 + alpha * distance[:, :, 0] + beta * distance[:, :, 1] + gamma * distance[:, :, 2])
    np.fill_diagonal(adjacency, 0.0)
    adjacency += np.eye(len(state), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    normalized = adjacency / np.sqrt(degree[:, None] * degree[None, :])
    return torch.as_tensor(normalized, dtype=torch.float32)


build_adjacency_matrix = build_adjacency


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and adjacency.ndim == 3:
            return torch.bmm(adjacency, x) @ self.weight
        return adjacency @ x @ self.weight


class GNNEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.relu(self.conv1(x, adjacency))
        x = torch.relu(self.conv2(x, adjacency))
        return self.output(x)


class QNetwork(nn.Module):
    def __init__(self, num_vms: int = 30, embedding_dim: int = 64) -> None:
        super().__init__()
        self.num_vms = num_vms
        self.network = nn.Sequential(
            nn.Linear(num_vms * embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_vms),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.flatten().unsqueeze(0)
        elif z.ndim > 2:
            z = z.flatten(start_dim=1)
        else:
            z = z.flatten().unsqueeze(0)
        return self.network(z)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 256):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DDQNAgent:
    def __init__(self, num_vms: int = 30, encoder: nn.Module | None = None, lr: float = 1e-3) -> None:
        self.num_vms = num_vms
        self.encoder = encoder or GNNEncoder()
        self.q_network = QNetwork(num_vms)
        self.target_network = deepcopy(self.q_network)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.q_network.parameters()), lr=lr)
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.99, 0.9, 0.05, 0.995
        self.previous_distribution: np.ndarray | None = None

    def _encode(self, state: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if state.ndim in (2, 3) and state.shape[-1] == 6:
            if state.ndim == 2:
                if adjacency is None:
                    adjacency = build_adjacency(state.detach().cpu().numpy())
                state = self.encoder(state, adjacency)
            else:
                if adjacency is None:
                    adjacency = build_adjacency_batch(state)
                elif adjacency.ndim == 2:
                    adjacency = adjacency.unsqueeze(0).expand(state.shape[0], -1, -1)
                state = self.encoder(state, adjacency)
        return state

    def _network_input(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def _q_values(self, state: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.q_network(self._network_input(self._encode(state, adjacency)))

    def select_action(self, state, valid_mask, adjacency: torch.Tensor | None = None, explore: bool = True) -> int | None:
        mask = np.asarray(valid_mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if not len(valid):
            return None
        if explore and random.random() < self.epsilon:
            action = int(random.choice(valid))
        else:
            with torch.no_grad():
                values = self._q_values(torch.as_tensor(state, dtype=torch.float32), adjacency)[0].cpu().numpy()
            values[~mask] = -np.inf
            action = int(np.argmax(values))
        if explore:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def learn(self, batch, adjacency: torch.Tensor | None = None) -> float:
        if not batch:
            return 0.0
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in states])
        next_states = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in next_states])
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        current = self._q_values(states, adjacency).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            encoded_next = self._network_input(self._encode(next_states, adjacency))
            next_actions = self.q_network(encoded_next).argmax(dim=1)
            target = self.target_network(encoded_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * (1 - dones) * target
        loss = nn.functional.mse_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def maybe_update_target(self, state: np.ndarray, threshold: float = 0.01) -> bool:
        distribution = np.asarray(state, dtype=np.float64)[:, :3].mean(axis=1)
        distribution = distribution + 1e-8
        distribution /= distribution.sum()
        if self.previous_distribution is None:
            self.previous_distribution = distribution
            return False
        previous = self.previous_distribution + 1e-8
        previous /= previous.sum()
        kl = float(np.sum(distribution * np.log(distribution / previous)))
        self.previous_distribution = distribution
        if kl > threshold:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return True
        return False
