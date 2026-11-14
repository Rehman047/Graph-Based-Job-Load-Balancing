"""Training and evaluation entry points."""

from pathlib import Path
import pandas as pd
import torch

from .baselines import GNNDQNAgent, MLPAgent, RandomScheduler, RoundRobin, SJF
from .env import GPUCluster, Job
from .metrics import avg_waiting_time, bsld, s_dolb
from .model import DDQNAgent, ReplayBuffer, build_adjacency


def _job(row) -> Job:
    return Job(float(row.gpu), float(row.cpu), float(row.mem), float(row.duration), float(row.get("arrival_time", 0)))


def train(jobs: pd.DataFrame, cluster: GPUCluster, encoder, agent: DDQNAgent, replay_buffer: ReplayBuffer, episodes: int = 1000):
    rewards, waiting = [], []
    for _ in range(episodes):
        cluster.reset()
        episode_reward = 0.0
        for position, (_, row) in enumerate(jobs.iterrows()):
            job = _job(row)
            state = cluster.get_state()
            adjacency = build_adjacency(state)
            action = agent.select_action(state, cluster.valid_mask(job), adjacency)
            cluster.step(action, job)
            next_state = cluster.get_state()
            reward = 0.6 * (1 - s_dolb(next_state)) + 0.4 * (1 - avg_waiting_time(cluster.jobs) / 10)
            replay_buffer.push(state, action or 0, reward, next_state, position == len(jobs) - 1)
            if len(replay_buffer) >= 256:
                agent.learn(replay_buffer.sample())
            agent.maybe_update_target(next_state)
            episode_reward += reward
        cluster.run()
        rewards.append(episode_reward)
        waiting.append(avg_waiting_time(cluster.jobs))
    return rewards, waiting


def evaluate(scheduler, jobs: pd.DataFrame, cluster: GPUCluster):
    cluster.reset()
    for _, row in jobs.iterrows():
        job = _job(row)
        state = cluster.get_state()
        if hasattr(scheduler, "select_action"):
            action = scheduler.select_action(state, cluster.valid_mask(job), explore=False)
        else:
            action = scheduler.select(cluster, job)
        cluster.step(action, job)
    cluster.run()
    state = cluster.get_state()
    scores = [bsld(job.waiting_time, job.duration) for job in cluster.jobs]
    return s_dolb(state), sum(scores) / len(scores) if scores else 0.0, avg_waiting_time(cluster.jobs)


def main():
    from .data_loader import load_traces
    from .plots import print_results_table

    normal, _ = load_traces()
    cluster = GPUCluster()
    agent = DDQNAgent(num_vms=len(cluster.vms))
    rewards, waits = train(normal, cluster, agent.encoder, agent, ReplayBuffer(), episodes=1000)
    methods = {
        "GNN-DDQN": agent,
        "MLP-DDQN": MLPAgent(len(cluster.vms)),
        "GNN-DQN": GNNDQNAgent(len(cluster.vms)),
        "RR": RoundRobin(),
        "Random": RandomScheduler(),
        "SJF": SJF(),
    }
    results = {name: dict(zip(("s_dolb", "bsld", "avg_waiting_time"), evaluate(scheduler, normal, cluster))) for name, scheduler in methods.items()}
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv("experiments/results/evaluation.csv")
    print_results_table(results)
    return rewards, waits


if __name__ == "__main__":
    main()
