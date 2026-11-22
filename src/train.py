"""Training and evaluation entry points."""

from collections import deque
from pathlib import Path
import argparse
import hashlib
import random
import sys
import time
import numpy as np
import pandas as pd
import torch

from .baselines import GNNDQNAgent, MLPAgent, RandomScheduler, RoundRobin, SJF
from .env import GPUCluster, Job
from .metrics import avg_waiting_time, bsld, s_dolb
from .model import DDQNAgent, ReplayBuffer, build_adjacency


def _job(row) -> Job:
    return Job(float(row.gpu), float(row.cpu), float(row.mem), float(row.duration), float(row.get("arrival_time", 0)))


def get_checkpoint_dir(checkpoint_root: str | Path, workload: str, n_per_type: int, agent_name: str) -> Path:
    return Path(checkpoint_root) / f"{workload}_{n_per_type}x3_{agent_name}"


def compute_trace_fingerprint(jobs: pd.DataFrame) -> str:
    """Compute a deterministic hash of the job trace."""
    cols = ["gpu", "cpu", "mem", "duration", "arrival_time"]
    rounded = jobs[cols].round(4).to_numpy()
    hasher = hashlib.sha256()
    hasher.update(str(len(jobs)).encode("utf-8"))
    hasher.update(rounded.tobytes())
    return hasher.hexdigest()[:16]


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def get_rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def set_rng_states(states: dict) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])


def validate_config(saved_config: dict, current_config: dict) -> None:
    mismatches = []
    fields = [
        ("workload", "workload"),
        ("num_vms", "number of VMs"),
        ("n_per_type", "VMs per tier"),
        ("agent_name", "agent name"),
        ("lr", "learning rate"),
        ("gamma", "discount factor (gamma)"),
        ("train_freq", "train frequency"),
        ("batch_size", "batch size"),
        ("buffer_capacity", "buffer capacity"),
        ("trace_fingerprint", "trace fingerprint"),
        ("num_jobs", "number of jobs"),
    ]
    for key, label in fields:
        if key in saved_config and saved_config[key] != current_config.get(key):
            mismatches.append(
                f"{label} ({key}): checkpoint has '{saved_config[key]}', current run has '{current_config.get(key)}'"
            )
    if mismatches:
        raise ValueError(
            "Checkpoint configuration mismatch! Cannot resume:\n  - "
            + "\n  - ".join(mismatches)
            + "\n\nPlease use matching settings, a different --checkpoint-dir, or use --overwrite to start fresh."
        )


def log_train_step(csv_path: Path, row: dict) -> None:
    header = ["episode", "cumulative_reward", "avg_waiting_time", "epsilon", "mean_loss", "target_updates", "elapsed_seconds"]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(
            f"{row['episode']},{row['cumulative_reward']:.6f},{row['avg_waiting_time']:.6f},"
            f"{row['epsilon']:.6f},{row['mean_loss']:.6f},{row['target_updates']},{row['elapsed_seconds']:.4f}\n"
        )
        f.flush()


def log_eval_step(csv_path: Path, row: dict) -> None:
    header = ["episode", "s_dolb", "bsld", "avg_waiting_time", "score", "is_best"]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(
            f"{row['episode']},{row['s_dolb']:.6f},{row['bsld']:.6f},"
            f"{row['avg_waiting_time']:.6f},{row['score']:.6f},{row['is_best']}\n"
        )
        f.flush()


def clean_csv_logs(train_csv: Path, eval_csv: Path, max_episode: int) -> None:
    if train_csv.exists() and train_csv.stat().st_size > 0:
        df = pd.read_csv(train_csv)
        if "episode" in df.columns:
            df = df[df["episode"] <= max_episode]
            df.to_csv(train_csv, index=False)
    if eval_csv.exists() and eval_csv.stat().st_size > 0:
        df = pd.read_csv(eval_csv)
        if "episode" in df.columns:
            df = df[df["episode"] <= max_episode]
            df.to_csv(eval_csv, index=False)


def compute_eval_score(s_dolb_val: float, avg_wait: float) -> float:
    return 0.6 * (1.0 - s_dolb_val) + 0.4 * (1.0 / (1.0 + avg_wait))


def train(
    jobs: pd.DataFrame,
    cluster: GPUCluster,
    encoder,
    agent: DDQNAgent,
    replay_buffer: ReplayBuffer,
    episodes: int = 1000,
    train_freq: int = 8,
    agent_name: str = "GNN-DDQN",
    workload_name: str = "normal",
    checkpoint_dir: str | Path = "checkpoints",
    resume: bool = False,
    overwrite: bool = False,
    save_every: int = 10,
    eval_every: int = 25,
):
    ckpt_dir = get_checkpoint_dir(checkpoint_dir, workload_name, cluster.n_per_type, agent_name)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    train_csv = ckpt_dir / "train_log.csv"
    eval_csv = ckpt_dir / "eval_log.csv"

    current_config = {
        "workload": workload_name,
        "num_vms": len(cluster.vms),
        "n_per_type": cluster.n_per_type,
        "agent_name": agent_name,
        "lr": agent.optimizer.param_groups[0]["lr"],
        "gamma": agent.gamma,
        "train_freq": train_freq,
        "batch_size": 256,
        "buffer_capacity": replay_buffer.buffer.maxlen if hasattr(replay_buffer, "buffer") else 10000,
        "trace_fingerprint": compute_trace_fingerprint(jobs),
        "num_jobs": len(jobs),
    }

    start_episode = 0
    best_score = -float("inf")

    if latest_path.exists():
        if not resume and not overwrite:
            raise RuntimeError(
                f"Checkpoint already exists at '{ckpt_dir}'. Use --resume to continue or --overwrite to start fresh."
            )
        if resume:
            payload = torch.load(latest_path, map_location="cpu", weights_only=False)
            validate_config(payload.get("config", {}), current_config)
            agent.load_state(payload["agent_state"])
            replay_buffer.buffer = deque(payload["replay_buffer"], maxlen=replay_buffer.buffer.maxlen)
            set_rng_states(payload["rng_states"])
            start_episode = payload["completed_episodes"]
            best_score = payload.get("best_score", payload.get("score", -float("inf")))
            epsilon_decay_episodes = payload.get(
                "epsilon_decay_episodes", max(1, int(0.4 * episodes) if episodes <= 4 else round(0.4 * episodes))
            )
            clean_csv_logs(train_csv, eval_csv, start_episode)
            print(
                f"Resumed from {latest_path}: {start_episode} episodes completed, "
                f"current epsilon={agent.epsilon:.4f}, best score so far={best_score:.4f}, "
                f"epsilon decay window: {epsilon_decay_episodes} episodes (fixed at first start)"
            )
        elif overwrite:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for p in (latest_path, best_path, train_csv, eval_csv):
                if p.exists():
                    p.unlink()
            start_episode = 0
            best_score = -float("inf")
            epsilon_decay_episodes = max(1, int(0.4 * episodes) if episodes <= 4 else round(0.4 * episodes))
    else:
        if resume:
            print(f"No checkpoint found at {latest_path}, starting from scratch.")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for p in (train_csv, eval_csv):
            if p.exists():
                p.unlink()
        start_episode = 0
        best_score = -float("inf")
        epsilon_decay_episodes = max(1, int(0.4 * episodes) if episodes <= 4 else round(0.4 * episodes))

    def make_payload(completed_eps: int, b_score: float) -> dict:
        return {
            "config": current_config,
            "completed_episodes": completed_eps,
            "epsilon_decay_episodes": epsilon_decay_episodes,
            "agent_state": agent.get_state(),
            "replay_buffer": list(replay_buffer.buffer),
            "rng_states": get_rng_states(),
            "best_score": b_score,
        }

    def make_best_payload(completed_eps: int, b_score: float) -> dict:
        return {
            "config": current_config,
            "completed_episodes": completed_eps,
            "epsilon_decay_episodes": epsilon_decay_episodes,
            "agent_state": {
                "encoder": agent.encoder.state_dict(),
                "q_network": agent.q_network.state_dict(),
                "target_network": agent.target_network.state_dict(),
            },
            "score": b_score,
            "best_score": b_score,
        }

    if start_episode >= episodes:
        print(
            f"Agent '{agent_name}' has already completed {start_episode}/{episodes} episodes. "
            f"Skipping further training."
        )
        if best_path.exists():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            agent.load_state(best_payload["agent_state"])
            print(f"Loaded best checkpoint from {best_path} (score: {best_payload.get('best_score', best_payload.get('score', 0.0)):.4f}) for evaluation.")
        rewards, waiting = [], []
        if train_csv.exists() and train_csv.stat().st_size > 0:
            df = pd.read_csv(train_csv)
            rewards = df["cumulative_reward"].tolist()
            waiting = df["avg_waiting_time"].tolist()
        return rewards, waiting

    ep = start_episode
    try:
        for ep in range(start_episode, episodes):
            ep_start = time.perf_counter()
            agent.update_epsilon(ep, epsilon_decay_episodes)
            cluster.reset()
            episode_reward = 0.0
            episode_losses = []
            target_updates = 0

            for position, (_, row) in enumerate(jobs.iterrows()):
                job = _job(row)
                if job.arrival_time > cluster.env.now:
                    cluster.env.run(until=job.arrival_time)
                state = cluster.get_state()
                adjacency = build_adjacency(state)
                action = agent.select_action(state, cluster.valid_mask(job), adjacency)
                cluster.step(action, job)
                cluster.env.run(until=cluster.env.now)
                next_state = cluster.get_state()
                waiting_component = 1.0 / (1.0 + avg_waiting_time(cluster.jobs))
                reward = 0.6 * (1 - s_dolb(next_state)) + 0.4 * waiting_component
                if action is not None:
                    replay_buffer.push(state, action, reward, next_state, position == len(jobs) - 1)
                # Standard DRL practice: update every K steps (train_freq) rather than every step.
                if len(replay_buffer) >= 256 and position % train_freq == 0:
                    loss = agent.learn(replay_buffer.sample())
                    episode_losses.append(loss)
                if agent.maybe_update_target(next_state):
                    target_updates += 1
                episode_reward += reward

            cluster.run()
            ep_elapsed = time.perf_counter() - ep_start
            mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_wait = avg_waiting_time(cluster.jobs)
            completed_ep = ep + 1

            log_train_step(train_csv, {
                "episode": completed_ep,
                "cumulative_reward": episode_reward,
                "avg_waiting_time": avg_wait,
                "epsilon": agent.epsilon,
                "mean_loss": mean_loss,
                "target_updates": target_updates,
                "elapsed_seconds": ep_elapsed,
            })

            # Periodic evaluation
            if completed_ep % eval_every == 0 or (completed_ep == episodes and best_score == -float("inf")):
                eval_metrics = evaluate(agent, jobs, cluster)
                score = compute_eval_score(eval_metrics[0], eval_metrics[2])
                is_best = score > best_score
                if is_best:
                    best_score = score
                    save_checkpoint(best_path, make_best_payload(completed_ep, best_score))
                    print(f"[*] New best model saved at episode {completed_ep} with score: {best_score:.4f}")
                log_eval_step(eval_csv, {
                    "episode": completed_ep,
                    "s_dolb": eval_metrics[0],
                    "bsld": eval_metrics[1],
                    "avg_waiting_time": eval_metrics[2],
                    "score": score,
                    "is_best": is_best,
                })

            # Periodic latest checkpoint save
            if completed_ep % save_every == 0 or completed_ep == episodes:
                save_checkpoint(latest_path, make_payload(completed_ep, best_score))

            if completed_ep % 10 == 0 or episodes <= 10:
                print(
                    f"Episode {completed_ep}/{episodes} took {ep_elapsed:.2f}s | "
                    f"Reward: {episode_reward:.2f} | Avg Wait: {avg_wait:.4f} | Eps: {agent.epsilon:.4f}",
                    flush=True,
                )

    except KeyboardInterrupt:
        completed_ep = ep
        print(f"\n[Interrupted] Ctrl+C received during episode {ep + 1}. Saving checkpoint for {agent_name} at completed episode {completed_ep}...")
        clean_csv_logs(train_csv, eval_csv, completed_ep)
        save_checkpoint(latest_path, make_payload(completed_ep, best_score))
        print(f"Successfully saved {latest_path}. Exiting.")
        sys.exit(130)

    # Load best model if exists for final evaluation
    if best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        agent.load_state(best_payload["agent_state"])
        print(f"Loaded best checkpoint from {best_path} (eval score: {best_payload.get('best_score', best_payload.get('score', 0.0)):.4f}) for evaluation.")

    rewards, waiting = [], []
    if train_csv.exists() and train_csv.stat().st_size > 0:
        df = pd.read_csv(train_csv)
        rewards = df["cumulative_reward"].tolist()
        waiting = df["avg_waiting_time"].tolist()
    return rewards, waiting


def evaluate(scheduler, jobs: pd.DataFrame, cluster: GPUCluster, collect_utilization=False):
    cluster.reset()
    utilization_samples = []
    state_snapshots = []
    for _, row in jobs.iterrows():
        job = _job(row)
        if job.arrival_time > cluster.env.now:
            cluster.env.run(until=job.arrival_time)
        state = cluster.get_state()
        if hasattr(scheduler, "select_action"):
            action = scheduler.select_action(state, cluster.valid_mask(job), explore=False)
        else:
            action = scheduler.select(cluster, job)
        cluster.step(action, job)
        cluster.env.run(until=cluster.env.now)
        post_state = cluster.get_state()
        state_snapshots.append(post_state)
        utilization_samples.append(post_state[:, :3].mean(axis=0))
    cluster.run()
    scores = [bsld(job.waiting_time, job.duration) for job in cluster.jobs]
    mean_s_dolb = (
        float(sum(s_dolb(snapshot) for snapshot in state_snapshots) / len(state_snapshots))
        if state_snapshots
        else 0.0
    )
    metrics = (mean_s_dolb, sum(scores) / len(scores) if scores else 0.0, avg_waiting_time(cluster.jobs))
    if not collect_utilization:
        return metrics
    utilization = (
        sum(utilization_samples) / len(utilization_samples)
        if utilization_samples
        else [0.0, 0.0, 0.0]
    )
    return metrics, utilization


def _methods(num_vms, only_agent: str | None = None):
    all_methods = {
        "GNN-DDQN": DDQNAgent(num_vms=num_vms),
        "MLP-DDQN": MLPAgent(num_vms=num_vms),
        "GNN-DQN": GNNDQNAgent(num_vms=num_vms),
    }
    if only_agent:
        if only_agent not in all_methods:
            raise ValueError(f"Unknown agent '{only_agent}'. Allowed: {list(all_methods.keys())}")
        return {only_agent: all_methods[only_agent]}
    return all_methods


def _evaluate_methods(jobs, cluster, agents, collect_utilization=False):
    methods = {
        **agents,
        "RR": RoundRobin(),
        "Random": RandomScheduler(),
        "SJF": SJF(),
    }
    results = {}
    utilization = {}
    for name, scheduler in methods.items():
        evaluation = evaluate(scheduler, jobs, cluster, collect_utilization)
        metrics = evaluation[0] if collect_utilization else evaluation
        results[name] = dict(zip(("s_dolb", "bsld", "avg_waiting_time"), metrics))
        if collect_utilization:
            utilization[name] = evaluation[1]
    return (results, utilization) if collect_utilization else results


def _train_methods(
    jobs,
    cluster,
    episodes,
    train_freq: int = 8,
    only_agent: str | None = None,
    workload_name: str = "normal",
    checkpoint_dir: str | Path = "checkpoints",
    resume: bool = False,
    overwrite: bool = False,
    save_every: int = 10,
    eval_every: int = 25,
):
    agents = _methods(len(cluster.vms), only_agent=only_agent)
    histories = {}
    for name, agent in agents.items():
        print(f"\n--- Training {name} for {episodes} episodes (train_freq={train_freq}) ---")
        rewards, waits = train(
            jobs,
            cluster,
            agent.encoder,
            agent,
            ReplayBuffer(),
            episodes=episodes,
            train_freq=train_freq,
            agent_name=name,
            workload_name=workload_name,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            overwrite=overwrite,
            save_every=save_every,
            eval_every=eval_every,
        )
        histories[name] = (rewards, waits)
    return agents, histories


def _priority_jobs(jobs):
    return jobs.sort_values("gpu", ascending=False, kind="stable").reset_index(drop=True)


def get_histories_from_logs(checkpoint_dir, workload_name, n_per_type, agent_names):
    reward_history = {}
    wait_history = {}
    for name in agent_names:
        ckpt_dir = get_checkpoint_dir(checkpoint_dir, workload_name, n_per_type, name)
        train_csv = ckpt_dir / "train_log.csv"
        if train_csv.exists() and train_csv.stat().st_size > 0:
            df = pd.read_csv(train_csv)
            if "cumulative_reward" in df.columns and "avg_waiting_time" in df.columns:
                reward_history[name] = df["cumulative_reward"].tolist()
                wait_history[name] = df["avg_waiting_time"].tolist()
    return reward_history, wait_history


def run_experiments(
    normal,
    intensive,
    episodes: int = 1000,
    vm_counts=(10, 20, 30),
    output_dir: str = "experiments/results",
    workload: str = "all",
    only_agent: str | None = None,
    train_freq: int = 8,
    resume: bool = False,
    overwrite: bool = False,
    save_every: int = 10,
    eval_every: int = 25,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
):
    from .plots import (
        plot_reward_convergence,
        plot_utilization_bars,
        plot_waiting_time_convergence,
        print_results_table,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    all_histories = {}

    workload_configs = []
    if workload in ("normal", "all"):
        workload_configs.append(("normal", normal))
    if workload in ("intensive", "all"):
        workload_configs.append(("intensive", intensive))

    for workload_name, jobs in workload_configs:
        cluster = GPUCluster()
        agents, histories = _train_methods(
            jobs,
            cluster,
            episodes,
            train_freq=train_freq,
            only_agent=only_agent,
            workload_name=workload_name,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            overwrite=overwrite,
            save_every=save_every,
            eval_every=eval_every,
        )
        results, utilization = _evaluate_methods(jobs, cluster, agents, collect_utilization=True)
        all_results[workload_name] = results
        all_histories[workload_name] = histories
        pd.DataFrame(results).T.to_csv(output / f"{workload_name}_evaluation.csv")
        print(f"\n{workload_name.title()} workload")
        print_results_table(results)

        reward_hist, wait_hist = get_histories_from_logs(
            checkpoint_dir, workload_name, cluster.n_per_type, list(agents.keys())
        )
        if not reward_hist:
            reward_hist = {name: values[0] for name, values in histories.items()}
        if not wait_hist:
            wait_hist = {name: values[1] for name, values in histories.items()}

        plot_reward_convergence(reward_hist, output / f"{workload_name}_reward_convergence.png")
        plot_waiting_time_convergence(wait_hist, output / f"{workload_name}_waiting_time_convergence.png")
        plot_utilization_bars(utilization, output / f"{workload_name}_utilization.png")

        priority_results = _evaluate_methods(_priority_jobs(jobs), cluster, agents)
        all_results[f"{workload_name}_gpu_priority"] = priority_results
        pd.DataFrame(priority_results).T.to_csv(output / f"{workload_name}_gpu_priority.csv")

    generalization = {}
    if workload in ("generalization", "all"):
        for count in vm_counts:
            print(f"\n--- Generalization experiment with {count} VMs per tier ---")
            cluster = GPUCluster(n_per_type=count)
            agents, _ = _train_methods(
                normal,
                cluster,
                episodes,
                train_freq=train_freq,
                only_agent=only_agent,
                workload_name="generalization",
                checkpoint_dir=checkpoint_dir,
                resume=resume,
                overwrite=overwrite,
                save_every=save_every,
                eval_every=eval_every,
            )
            generalization[count] = _evaluate_methods(normal, cluster, agents)
        pd.Series(generalization).to_json(output / "generalization.json")
    return all_results, all_histories, generalization


def main(
    episodes: int = 1000,
    workload: str = "all",
    only_agent: str | None = None,
    train_freq: int = 8,
    resume: bool = False,
    overwrite: bool = False,
    save_every: int = 10,
    eval_every: int = 25,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from .data_loader import load_traces

    return run_experiments(
        *load_traces(),
        episodes=episodes,
        workload=workload,
        only_agent=only_agent,
        train_freq=train_freq,
        resume=resume,
        overwrite=overwrite,
        save_every=save_every,
        eval_every=eval_every,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate all scheduling experiments.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--workload",
        type=str,
        choices=["normal", "intensive", "generalization", "all"],
        default="all",
        help="Workload to run: normal, intensive, generalization, or all (default: all)",
    )
    parser.add_argument(
        "--only-agent",
        type=str,
        choices=["GNN-DDQN", "MLP-DDQN", "GNN-DQN"],
        default=None,
        help="Run only the specified agent (default: run all three)",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=8,
        help="Update policy network every K jobs (default: 8)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from latest checkpoint if available",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing checkpoints instead of erroring",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save latest checkpoint every N episodes (default: 10)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Run periodic evaluation every M episodes (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    args = parser.parse_args()
    main(
        episodes=args.episodes,
        workload=args.workload,
        only_agent=args.only_agent,
        train_freq=args.train_freq,
        resume=args.resume,
        overwrite=args.overwrite,
        save_every=args.save_every,
        eval_every=args.eval_every,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
