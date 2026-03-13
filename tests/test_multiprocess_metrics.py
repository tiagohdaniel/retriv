"""
Verifies that Prometheus multiprocess mode aggregates metrics across workers.

Strategy: spawn two independent processes (simulating Gunicorn workers), each
writing to shared mmap files in a temp dir. Then use MultiProcessCollector to
aggregate and assert the sum matches both workers combined.

Uses multiprocessing.get_context("spawn") so each subprocess imports
prometheus_client fresh, after PROMETHEUS_MULTIPROC_DIR is set.
"""
import multiprocessing
import tempfile

import pytest


def _worker(multiproc_dir: str, increment: float) -> None:
    """Simulates a single Gunicorn worker process."""
    import os
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir

    # Import AFTER setting the env var so prometheus_client uses multiprocess mode
    from prometheus_client import Counter
    c = Counter("retriv_multiprocess_test_total", "Multiprocess aggregation test counter")
    c.inc(increment)
    # mmap file is flushed automatically on process exit


def test_multiprocess_metrics_are_aggregated():
    """
    Two workers increment the same counter independently.
    MultiProcessCollector must return their sum.
    """
    worker_1_value = 10.0
    worker_2_value = 20.0
    expected_total = worker_1_value + worker_2_value

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = multiprocessing.get_context("spawn")

        p1 = ctx.Process(target=_worker, args=(tmpdir, worker_1_value))
        p2 = ctx.Process(target=_worker, args=(tmpdir, worker_2_value))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        assert p1.exitcode == 0, "Worker 1 exited with an error"
        assert p2.exitcode == 0, "Worker 2 exited with an error"

        from prometheus_client import CollectorRegistry, generate_latest
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=tmpdir)
        output = generate_latest(registry).decode()

        assert "retriv_multiprocess_test_total" in output, (
            "Counter not found in aggregated metrics"
        )
        assert f"{expected_total}" in output, (
            f"Expected aggregated value {expected_total} not found in:\n{output}"
        )
