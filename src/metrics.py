"""Scheduling metrics."""

import numpy as np


def s_dolb(state: np.ndarray) -> float:
    values = np.asarray(state)
    return float(sum(np.std(values[:, column]) for column in range(3)))


def bsld(waiting_time: float, execution_time: float, tau: float | None = None) -> float:
    threshold = 10.0 if tau is None else tau
    scale = max(execution_time, threshold)
    if scale <= 0:
        return 1.0
    return max((waiting_time + execution_time) / scale, 1.0)


def avg_waiting_time(jobs) -> float:
    started = [job.waiting_time for job in jobs if job.start_time is not None]
    return float(np.mean(started)) if started else 0.0
