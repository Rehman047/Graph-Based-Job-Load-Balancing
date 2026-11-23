"""Alibaba trace loading, preprocessing, and arrival time calibration.

Preprocessing decisions and rationale:
1. Unit Conversions:
   - PAI v2020 (pai_task_table.csv):
     * plan_gpu is in percentage of 1 GPU card (100 = 1 full GPU). Divided by 100 to yield device counts.
     * plan_cpu is in percentage of 1 CPU core (100 = 1 full core). Divided by 100 to yield core counts.
     * plan_mem is in GB (e.g. 29.296875 GB = 30000 MB / 1024). Retained as-is matching VM memory capacities.
   - OpenB v2023 (openb_pod_list_default.csv):
     * num_gpu represents GPU device count directly.
     * cpu_milli represents milli-cores. Divided by 1000 to yield core counts.
     * memory_mib represents MiB. Divided by 1024 to yield GB.
2. Filtering Unschedulable Jobs:
   - In the heterogeneous cluster, VM capacities are (4, 8, 16), (8, 16, 32), and (16, 32, 64) for
     (GPU, CPU, Memory in GB). The maximum capacity of any single VM is 16 GPU, 32 CPU, 64 GB memory.
   - Jobs demanding > 16 GPU, > 32 CPU, or > 64 GB memory can never fit on any VM in the cluster and are
     dropped before scheduling. Non-positive durations are also removed.
3. Duration Scaling:
   - Instead of dividing by the trace maximum duration (which compresses typical jobs into near-zero durations),
     raw durations are clipped at the 99th percentile within each trace and normalized by that trace's median.
   - This ensures a typical job lasts ~1.0 time unit, while long-running jobs are capped at the 99th percentile
     with a floor of 0.01 time units.
4. Arrival Time Calibration (TARGET_LOAD = 0.6):
   - 500 normal jobs are sampled (seed 7) and sorted by original arrival time.
   - The time span is chosen such that the offered load on the bottleneck resource equals TARGET_LOAD (0.6):
       Offered Load = sum(demand * duration) / (total_cluster_capacity * span)
       span = max_r(sum(demand_r * duration) / total_capacity_r) / TARGET_LOAD
   - Normal arrivals are mapped linearly onto [0, span].
   - The intensive workload samples 1000 jobs (seed 8) and maps them onto the EXACT same span [0, span],
     carrying roughly double the offered load (~1.23 on the memory bottleneck) to induce realistic queuing.
5. Visibility and Validation:
   - On every load, data source, filtering statistics, offered loads, median durations, and simulated Round Robin
     queueing fractions are displayed. Missing CSV files raise a clear FileNotFoundError instead of silent fallback.
"""

from pathlib import Path
import numpy as np
import pandas as pd

TARGET_LOAD = 0.6


def _scale_duration(raw_durations: pd.Series) -> pd.Series:
    """Clip raw durations at the 99th percentile and normalize by median."""
    vals = pd.to_numeric(raw_durations, errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return pd.Series(dtype=float)
    p99 = float(vals.quantile(0.99))
    med = float(vals.median())
    if med <= 0:
        med = 1.0
    return (vals.clip(lower=0.01, upper=p99) / med).clip(lower=0.01)


def _load_v2020(
    path: Path, max_gpu: float = 16.0, max_cpu: float = 32.0, max_mem: float = 64.0
) -> tuple[pd.DataFrame, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"PAI v2020 trace file not found at: {path}")
    frame = pd.read_csv(path)
    required = ["plan_gpu", "plan_cpu", "plan_mem", "start_time", "end_time"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path} is missing required columns: {required}")

    raw_dur = pd.to_numeric(frame["end_time"], errors="coerce") - pd.to_numeric(
        frame["start_time"], errors="coerce"
    )

    df = pd.DataFrame(
        {
            "gpu": pd.to_numeric(frame["plan_gpu"], errors="coerce") / 100.0,
            "cpu": pd.to_numeric(frame["plan_cpu"], errors="coerce") / 100.0,
            "mem": pd.to_numeric(frame["plan_mem"], errors="coerce"),
            "duration_raw": raw_dur,
            "arrival_time": pd.to_numeric(frame["start_time"], errors="coerce"),
        }
    ).dropna()

    df = df[df["duration_raw"] > 0]
    total_valid = len(df)

    schedulable_mask = (df["gpu"] <= max_gpu) & (df["cpu"] <= max_cpu) & (df["mem"] <= max_mem)
    dropped = int((~schedulable_mask).sum())
    df = df[schedulable_mask].copy()

    p99 = float(df["duration_raw"].quantile(0.99))
    med = float(df["duration_raw"].median())
    if med <= 0:
        med = 1.0
    df["duration"] = (df["duration_raw"].clip(lower=0.01, upper=p99) / med).clip(lower=0.01)
    df = df.drop(columns=["duration_raw"]).reset_index(drop=True)

    df["arrival_time"] = df["arrival_time"] - df["arrival_time"].min()
    return df, total_valid - dropped, dropped


def _load_v2023(
    path: Path, max_gpu: float = 16.0, max_cpu: float = 32.0, max_mem: float = 64.0
) -> tuple[pd.DataFrame, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"OpenB v2023 trace file not found at: {path}")
    frame = pd.read_csv(path)
    required = ["num_gpu", "cpu_milli", "memory_mib", "scheduled_time", "deletion_time"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path} is missing required columns: {required}")

    raw_dur = pd.to_numeric(frame["deletion_time"], errors="coerce") - pd.to_numeric(
        frame["scheduled_time"], errors="coerce"
    )

    df = pd.DataFrame(
        {
            "gpu": pd.to_numeric(frame["num_gpu"], errors="coerce"),
            "cpu": pd.to_numeric(frame["cpu_milli"], errors="coerce") / 1000.0,
            "mem": pd.to_numeric(frame["memory_mib"], errors="coerce") / 1024.0,
            "duration_raw": raw_dur,
            "arrival_time": pd.to_numeric(frame["scheduled_time"], errors="coerce"),
        }
    ).dropna()

    df = df[df["duration_raw"] > 0]
    total_valid = len(df)

    schedulable_mask = (df["gpu"] <= max_gpu) & (df["cpu"] <= max_cpu) & (df["mem"] <= max_mem)
    dropped = int((~schedulable_mask).sum())
    df = df[schedulable_mask].copy()

    p99 = float(df["duration_raw"].quantile(0.99))
    med = float(df["duration_raw"].median())
    if med <= 0:
        med = 1.0
    df["duration"] = (df["duration_raw"].clip(lower=0.01, upper=p99) / med).clip(lower=0.01)
    df = df.drop(columns=["duration_raw"]).reset_index(drop=True)

    df["arrival_time"] = df["arrival_time"] - df["arrival_time"].min()
    return df, total_valid - dropped, dropped


def _synthetic_jobs(count: int, seed: int = 7) -> pd.DataFrame:
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


def estimate_queueing_fraction(jobs: pd.DataFrame) -> tuple[int, float]:
    """Simulate RoundRobin once to estimate the number and fraction of jobs that queue."""
    from .env import GPUCluster, Job
    from .baselines import RoundRobin

    cluster = GPUCluster()
    rr = RoundRobin()
    for _, row in jobs.iterrows():
        job = Job(
            float(row.gpu),
            float(row.cpu),
            float(row.mem),
            float(row.duration),
            float(row.arrival_time),
        )
        if job.arrival_time > cluster.env.now:
            cluster.env.run(until=job.arrival_time)
        action = rr.select(cluster, job)
        cluster.step(action, job)
        cluster.env.run(until=cluster.env.now)
    cluster.run()
    queued = sum(1 for j in cluster.jobs if j.waiting_time > 0)
    fraction = queued / len(jobs) if len(jobs) > 0 else 0.0
    return queued, fraction


def load_traces(
    data_dir: str | Path = "data", synthetic: bool = False, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normal (500 jobs) and intensive (1000 jobs) scheduling workloads."""
    from .env import GPUCluster

    cluster = GPUCluster()
    max_gpu = max(vm.gpu_capacity for vm in cluster.vms)
    max_cpu = max(vm.cpu_capacity for vm in cluster.vms)
    max_mem = max(vm.mem_capacity for vm in cluster.vms)

    total_c_gpu = sum(vm.gpu_capacity for vm in cluster.vms)
    total_c_cpu = sum(vm.cpu_capacity for vm in cluster.vms)
    total_c_mem = sum(vm.mem_capacity for vm in cluster.vms)

    root = Path(data_dir)
    p2020 = root / "v2020" / "pai_task_table.csv"
    p2023 = root / "v2023" / "openb_pod_list_default.csv"

    if synthetic:
        source_name = "Synthetic fallback generator"
        combined = _synthetic_jobs(2000, seed=7)
        v20_kept, v20_dropped = len(combined), 0
        v23_kept, v23_dropped = 0, 0
    else:
        source_name = f"Real Alibaba cluster traces ({p2020}, {p2023})"
        first, v20_kept, v20_dropped = _load_v2020(p2020, max_gpu, max_cpu, max_mem)
        second, v23_kept, v23_dropped = _load_v2023(p2023, max_gpu, max_cpu, max_mem)
        combined = pd.concat([first, second], ignore_index=True)

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

    # Calibrate span so offered load on bottleneck resource equals TARGET_LOAD (0.6)
    w_gpu = float((normal["gpu"] * normal["duration"]).sum())
    w_cpu = float((normal["cpu"] * normal["duration"]).sum())
    w_mem = float((normal["mem"] * normal["duration"]).sum())

    load_gpu_ratio = w_gpu / total_c_gpu
    load_cpu_ratio = w_cpu / total_c_cpu
    load_mem_ratio = w_mem / total_c_mem

    bottleneck_val = max(load_gpu_ratio, load_cpu_ratio, load_mem_ratio)
    bottleneck_res = (
        "Memory"
        if bottleneck_val == load_mem_ratio
        else ("CPU" if bottleneck_val == load_cpu_ratio else "GPU")
    )

    span = bottleneck_val / TARGET_LOAD

    # Linearly project normal arrivals onto [0, span]
    t0_norm, t1_norm = float(normal["arrival_time"].min()), float(normal["arrival_time"].max())
    if t1_norm > t0_norm:
        normal["arrival_time"] = (normal["arrival_time"] - t0_norm) / (t1_norm - t0_norm) * span
    else:
        normal["arrival_time"] = np.linspace(0, span, len(normal))

    # Linearly project intensive arrivals onto the SAME [0, span] window
    t0_int, t1_int = float(intensive["arrival_time"].min()), float(intensive["arrival_time"].max())
    if t1_int > t0_int:
        intensive["arrival_time"] = (intensive["arrival_time"] - t0_int) / (t1_int - t0_int) * span
    else:
        intensive["arrival_time"] = np.linspace(0, span, len(intensive))

    # Offered loads
    norm_load_gpu = w_gpu / (total_c_gpu * span)
    norm_load_cpu = w_cpu / (total_c_cpu * span)
    norm_load_mem = w_mem / (total_c_mem * span)

    w_gpu_int = float((intensive["gpu"] * intensive["duration"]).sum())
    w_cpu_int = float((intensive["cpu"] * intensive["duration"]).sum())
    w_mem_int = float((intensive["mem"] * intensive["duration"]).sum())

    int_load_gpu = w_gpu_int / (total_c_gpu * span)
    int_load_cpu = w_cpu_int / (total_c_cpu * span)
    int_load_mem = w_mem_int / (total_c_mem * span)

    int_bottleneck_val = max(int_load_gpu, int_load_cpu, int_load_mem)
    int_bottleneck_res = (
        "Memory"
        if int_bottleneck_val == int_load_mem
        else ("CPU" if int_bottleneck_val == int_load_cpu else "GPU")
    )

    norm_queued_count, norm_queued_frac = estimate_queueing_fraction(normal)
    int_queued_count, int_queued_frac = estimate_queueing_fraction(intensive)

    if verbose:
        print("=" * 80)
        print("Workload Preprocessing & Cluster Saturation")
        print("=" * 80)
        print(f"Data Source: {source_name}")
        if not synthetic:
            v20_pct = v20_dropped / (v20_kept + v20_dropped) * 100 if (v20_kept + v20_dropped) else 0
            v23_pct = v23_dropped / (v23_kept + v23_dropped) * 100 if (v23_kept + v23_dropped) else 0
            print("Trace filtering (dropped if GPU > 16, CPU > 32, or MEM > 64 GB):")
            print(f"  - PAI v2020 : {v20_kept:,} kept, {v20_dropped:,} dropped ({v20_pct:.2f}%)")
            print(f"  - OpenB v2023: {v23_kept:,} kept, {v23_dropped:,} dropped ({v23_pct:.2f}%)")
        print(
            f"Duration scaling: clipped at 99th percentile and normalized by trace median (median = 1.0)"
        )
        print(f"\nNormal workload ({len(normal)} jobs):")
        print(f"  - Bottleneck resource: {bottleneck_res}")
        print(
            f"  - Offered load: GPU={norm_load_gpu:.3f}, CPU={norm_load_cpu:.3f}, MEM={norm_load_mem:.3f} (TARGET_LOAD={TARGET_LOAD:.2f})"
        )
        print(f"  - Time span: {span:.2f} time units")
        print(f"  - Median duration: {normal['duration'].median():.4f} time units")
        print(
            f"  - Estimated queueing fraction (Round Robin): {norm_queued_frac:.1%} ({norm_queued_count}/{len(normal)} jobs)"
        )

        print(f"\nIntensive workload ({len(intensive)} jobs):")
        print(f"  - Bottleneck resource: {int_bottleneck_res}")
        print(
            f"  - Offered load: GPU={int_load_gpu:.3f}, CPU={int_load_cpu:.3f}, MEM={int_load_mem:.3f}"
        )
        print(f"  - Time span: {span:.2f} time units (same span as normal)")
        print(f"  - Median duration: {intensive['duration'].median():.4f} time units")
        print(
            f"  - Estimated queueing fraction (Round Robin): {int_queued_frac:.1%} ({int_queued_count}/{len(intensive)} jobs)"
        )
        print("=" * 80)

    return normal, intensive


def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible alias for :func:`load_traces`."""
    return load_traces(data_dir)
