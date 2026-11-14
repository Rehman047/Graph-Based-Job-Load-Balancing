"""Discrete-event GPU cluster environment."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import simpy


@dataclass
class Job:
    gpu: float
    cpu: float
    mem: float
    duration: float
    arrival_time: float = 0.0
    start_time: float | None = None
    finish_time: float | None = None

    @property
    def waiting_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return max(0.0, self.start_time - self.arrival_time)


class VM:
    def __init__(self, gpu: float, cpu: float, mem: float, name: str | None = None) -> None:
        self.gpu_capacity, self.cpu_capacity, self.mem_capacity = float(gpu), float(cpu), float(mem)
        self.gpu_used = self.cpu_used = self.mem_used = 0.0
        self.name = name

    def can_fit(self, job: Job) -> bool:
        return (
            job.gpu <= self.gpu_capacity - self.gpu_used
            and job.cpu <= self.cpu_capacity - self.cpu_used
            and job.mem <= self.mem_capacity - self.mem_used
        )

    def allocate(self, job: Job) -> None:
        if not self.can_fit(job):
            raise ValueError("job does not fit on VM")
        self.gpu_used += job.gpu
        self.cpu_used += job.cpu
        self.mem_used += job.mem

    def release(self, job: Job) -> None:
        self.gpu_used = max(0.0, self.gpu_used - job.gpu)
        self.cpu_used = max(0.0, self.cpu_used - job.cpu)
        self.mem_used = max(0.0, self.mem_used - job.mem)

    @property
    def total_usage(self) -> float:
        return self.gpu_used / self.gpu_capacity + self.cpu_used / self.cpu_capacity + self.mem_used / self.mem_capacity


class GPUCluster:
    NODE_TYPES = ((4, 8, 16), (8, 16, 32), (16, 32, 64))

    def __init__(self, n_per_type: int = 10, env: Any | None = None) -> None:
        self.n_per_type = n_per_type
        self.env = env or simpy.Environment()
        self.vms: list[VM] = []
        self.jobs: list[Job] = []
        self.reset()

    @property
    def machines(self) -> list[VM]:
        return self.vms

    def reset(self) -> None:
        self.env = simpy.Environment()
        self.vms = [
            VM(*node_type, name=f"T{type_index + 1}-{instance}")
            for type_index, node_type in enumerate(self.NODE_TYPES)
            for instance in range(self.n_per_type)
        ]
        self.jobs = []

    def get_state(self) -> np.ndarray:
        rows = []
        for vm in self.vms:
            rows.append(
                [
                    vm.gpu_used / vm.gpu_capacity,
                    vm.cpu_used / vm.cpu_capacity,
                    vm.mem_used / vm.mem_capacity,
                    (vm.gpu_capacity - vm.gpu_used) / vm.gpu_capacity,
                    (vm.cpu_capacity - vm.cpu_used) / vm.cpu_capacity,
                    (vm.mem_capacity - vm.mem_used) / vm.mem_capacity,
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def valid_mask(self, job: Job) -> np.ndarray:
        return np.asarray([vm.can_fit(job) for vm in self.vms], dtype=bool)

    def _run_job(self, vm: VM, job: Job):
        yield self.env.timeout(max(0.0, job.duration))
        vm.release(job)
        job.finish_time = self.env.now

    def step(self, vm_index: int | None, job: Job) -> None:
        if vm_index is None:
            self.jobs.append(job)
            return
        if vm_index < 0 or vm_index >= len(self.vms) or not self.vms[vm_index].can_fit(job):
            raise ValueError("invalid VM index or job does not fit")
        job.start_time = max(float(self.env.now), job.arrival_time)
        self.vms[vm_index].allocate(job)
        self.jobs.append(job)
        self.env.process(self._run_job(self.vms[vm_index], job))

    def run(self) -> None:
        self.env.run()
