"""Gunicorn configuration.

child_exit: cleans up the dead worker's prometheus mmap files so stale
metrics are not included in the next /metrics scrape.
Only active when PROMETHEUS_MULTIPROC_DIR is set.
"""
import os


def child_exit(server, worker):
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess
        multiprocess.mark_process_dead(worker.pid)
