# Graph-Based Job Load Balancing

This project implements a GNN-DDQN batch scheduler for heterogeneous GPU
clusters, with RR, Random, SJF, MLP-DDQN, and GNN-DQN baselines.

## Setup

For a CPU-only machine, create and activate a virtual environment, then install
the CPU PyTorch wheel explicitly:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install simpy numpy pandas matplotlib scipy
```

The explicit CPU index avoids downloading the much larger CUDA distribution.
If you already have a compatible PyTorch installation, install the remaining
dependencies listed in `requirements.txt`:

```bash
python -m pip install simpy numpy pandas matplotlib scipy
```

Download Alibaba traces from
[`alibaba/clusterdata`](https://github.com/alibaba/clusterdata) and place
`pai_task_table.csv` in `data/v2020/` and `openb_pod_list_default.csv` in
`data/v2023/`. If either trace is unavailable or malformed, the loader
automatically uses reproducible synthetic jobs.

## Run

```bash
python -m src.train
```

Evaluation CSVs and plots are written to `experiments/results/`.
