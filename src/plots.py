"""Result plots and tabular reporting."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("experiments/results")


def _save(name):
    requested = Path(name)
    path = requested if requested.is_absolute() or requested.parent != Path(".") else RESULTS / requested
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_reward_convergence(rewards, filename="reward_convergence.png"):
    for name, values in rewards.items():
        plt.plot(values, label=name)
    plt.xlabel("Episode"); plt.ylabel("Cumulative reward"); plt.legend()
    return _save(filename)


def plot_waiting_time_convergence(waiting_times, filename="waiting_time_convergence.png"):
    for name, values in waiting_times.items():
        plt.plot(values, label=name)
    plt.xlabel("Episode"); plt.ylabel("Average waiting time"); plt.legend()
    return _save(filename)


def plot_utilization_bars(utilization, filename="utilization.png"):
    names = list(utilization)
    values = np.asarray([utilization[name] for name in names])
    x = np.arange(len(names)); width = 0.25
    for index, label in enumerate(("GPU", "CPU", "Memory")):
        plt.bar(x + (index - 1) * width, values[:, index], width, label=label)
    plt.xticks(x, names); plt.ylabel("Utilization"); plt.legend()
    return _save(filename)


def print_results_table(results):
    headers = ("Method", "s_dolb", "bsld", "avg_waiting_time")
    print(f"{headers[0]:<14}{headers[1]:>12}{headers[2]:>12}{headers[3]:>20}")
    for name, metrics in results.items():
        print(f"{name:<14}{metrics['s_dolb']:>12.4f}{metrics['bsld']:>12.4f}{metrics['avg_waiting_time']:>20.4f}")


results_table = print_results_table
