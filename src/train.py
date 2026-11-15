"""Training and evaluation entry points."""

from pathlib import Path
import argparse
import pandas as pd

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
            waiting_component = 1.0 / (1.0 + avg_waiting_time(cluster.jobs))
            reward = 0.6 * (1 - s_dolb(next_state)) + 0.4 * waiting_component
            if action is not None:
                replay_buffer.push(state, action, reward, next_state, position == len(jobs) - 1)
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


def _methods(num_vms):
    return {
        "GNN-DDQN": DDQNAgent(num_vms=num_vms),
        "MLP-DDQN": MLPAgent(num_vms=num_vms),
        "GNN-DQN": GNNDQNAgent(num_vms=num_vms),
    }


def _evaluate_methods(jobs, cluster, agents):
    methods = {
        **agents,
        "RR": RoundRobin(),
        "Random": RandomScheduler(),
        "SJF": SJF(),
    }
    return {
        name: dict(zip(("s_dolb", "bsld", "avg_waiting_time"), evaluate(scheduler, jobs, cluster)))
        for name, scheduler in methods.items()
    }


def _train_methods(jobs, cluster, episodes):
    agents = _methods(len(cluster.vms))
    histories = {}
    for name, agent in agents.items():
        rewards, waits = train(jobs, cluster, agent.encoder, agent, ReplayBuffer(), episodes)
        histories[name] = (rewards, waits)
    return agents, histories


def _priority_jobs(jobs):
    return jobs.sort_values("gpu", ascending=False, kind="stable").reset_index(drop=True)


def run_experiments(normal, intensive, episodes=1000, vm_counts=(10, 20, 30), output_dir="experiments/results"):
    from .plots import plot_reward_convergence, plot_waiting_time_convergence, print_results_table

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    all_histories = {}
    for workload_name, jobs in (("normal", normal), ("intensive", intensive)):
        cluster = GPUCluster()
        agents, histories = _train_methods(jobs, cluster, episodes)
        results = _evaluate_methods(jobs, cluster, agents)
        all_results[workload_name] = results
        all_histories[workload_name] = histories
        pd.DataFrame(results).T.to_csv(output / f"{workload_name}_evaluation.csv")
        print(f"\n{workload_name.title()} workload")
        print_results_table(results)
        plot_reward_convergence({name: values[0] for name, values in histories.items()},
                                output / f"{workload_name}_reward_convergence.png")
        plot_waiting_time_convergence({name: values[1] for name, values in histories.items()},
                                      output / f"{workload_name}_waiting_time_convergence.png")

        priority_results = _evaluate_methods(_priority_jobs(jobs), cluster, agents)
        all_results[f"{workload_name}_gpu_priority"] = priority_results
        pd.DataFrame(priority_results).T.to_csv(output / f"{workload_name}_gpu_priority.csv")

    generalization = {}
    for count in vm_counts:
        cluster = GPUCluster(n_per_type=count)
        agents, _ = _train_methods(normal, cluster, episodes)
        generalization[count] = _evaluate_methods(normal, cluster, agents)
    pd.Series(generalization).to_json(output / "generalization.json")
    return all_results, all_histories, generalization


def main(episodes=1000):
    from .data_loader import load_traces

    return run_experiments(*load_traces(), episodes=episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate all scheduling experiments.")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    main(episodes=args.episodes)
