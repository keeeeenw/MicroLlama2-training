"""CPU budgets that respect both affinity and container quotas."""
import os
from pathlib import Path


def available_cpus(cgroup=Path('/sys/fs/cgroup')):
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    directories = {cgroup}
    membership = Path('/proc/self/cgroup')
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if not controllers:
                current = cgroup / relative.lstrip('/')
                while current.is_relative_to(cgroup):
                    directories.add(current)
                    if current == cgroup:
                        break
                    current = current.parent
    for directory in directories:
        quota = directory / 'cpu.max'
        if quota.exists():
            limit, period = quota.read_text().split()
            if limit != 'max':
                count = min(count, max(1, int(limit) // int(period)))
    for directory in (cgroup, cgroup / 'cpu'):
        quota, period = directory / 'cpu.cfs_quota_us', directory / 'cpu.cfs_period_us'
        if quota.exists() and period.exists() and int(quota.read_text()) > 0:
            count = min(count, max(1, int(quota.read_text()) // int(period.read_text())))
    return count


def tokenization_budget(sources, workers=None, threads=None):
    cpus = available_cpus()
    if sources < 1 or any(value is not None and value < 1 for value in (workers, threads)):
        raise ValueError('Source, worker and thread counts must be positive')
    workers = min(sources, workers or max(1, cpus // (threads or 4)))
    threads = threads or max(1, cpus // workers)
    if workers * threads > cpus:
        raise ValueError(f'{workers} workers × {threads} threads exceeds the {cpus}-CPU budget')
    return {'cpus': cpus, 'workers': workers, 'threads': threads}
