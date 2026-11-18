"""Alibaba trace loading and synthetic fallback data generation."""

from pathlib import Path

import numpy as np
import pandas as pd


def _duration(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    values = values.clip(lower=0)
    maximum = float(values.max())
    if maximum <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values / maximum * 10).clip(0, 10)


def _synthetic_jobs(count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "gpu": rng.integers(1, 5, count).astype(float),
            "cpu": rng.uniform(1, 8, count),
            "mem": rng.uniform(2, 16, count),
            "duration": rng.uniform(0.5, 10, count),
            "arrival_time": np.arange(count, dtype=float),
        }
    )


def _load_v2020(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["plan_gpu", "plan_cpu", "plan_mem", "start_time", "end_time"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path} is missing one of {required}")
    result = pd.DataFrame(
        {
            "gpu": pd.to_numeric(frame["plan_gpu"], errors="coerce"),
            "cpu": pd.to_numeric(frame["plan_cpu"], errors="coerce"),
            "mem": pd.to_numeric(frame["plan_mem"], errors="coerce"),
            "duration_raw": pd.to_numeric(frame["end_time"], errors="coerce")
            - pd.to_numeric(frame["start_time"], errors="coerce"),
            "arrival_time": pd.to_numeric(frame["start_time"], errors="coerce"),
        }
    )
    result["duration"] = _duration(result.pop("duration_raw"))
    return result.dropna().reset_index(drop=True)


def _load_v2023(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["num_gpu", "cpu_milli", "memory_mib", "scheduled_time", "deletion_time"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path} is missing one of {required}")
    result = pd.DataFrame(
        {
            "gpu": pd.to_numeric(frame["num_gpu"], errors="coerce"),
            "cpu": pd.to_numeric(frame["cpu_milli"], errors="coerce") / 1000,
            "mem": pd.to_numeric(frame["memory_mib"], errors="coerce") / 1024,
            "duration_raw": pd.to_numeric(frame["deletion_time"], errors="coerce")
            - pd.to_numeric(frame["scheduled_time"], errors="coerce"),
            "arrival_time": pd.to_numeric(frame["scheduled_time"], errors="coerce"),
        }
    )
    result["duration"] = _duration(result.pop("duration_raw"))
    return result.dropna().reset_index(drop=True)


def load_traces(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normal (500 jobs) and intensive (1000 jobs) scheduling workloads."""
    root = Path(data_dir)
    try:
        first = _load_v2020(root / "v2020" / "pai_task_table.csv")
        second = _load_v2023(root / "v2023" / "openb_pod_list_default.csv")
        combined = pd.concat([first, second], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError, ValueError):
        combined = _synthetic_jobs(1000, seed=7)
    if combined.empty:
        combined = _synthetic_jobs(1000, seed=7)
    normal = (
        combined.sample(n=500, replace=len(combined) < 500, random_state=7)
        .sort_values("arrival_time", kind="stable")
        .reset_index(drop=True)
    )
    intensive = (
        combined.sample(n=1000, replace=len(combined) < 1000, random_state=8)
        .sort_values("arrival_time", kind="stable")
        .reset_index(drop=True)
    )
    return normal, intensive


def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible alias for :func:`load_traces`."""
    return load_traces(data_dir)
