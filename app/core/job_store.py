"""File-based job store for async indexing.

Each job is a JSON file under /tmp/retriv_jobs/. Files are readable by all
Gunicorn workers without any shared state mechanism — a GET /index/status
request can hit any worker and still find the job.

Jobs are never explicitly deleted; they are left on disk and naturally
cleaned by the OS on reboot or tmpfs rotation. TTL is not enforced here
because in practice the job directory stays small.
"""

import json
import uuid
from pathlib import Path

_JOB_DIR = Path("/tmp/retriv_jobs")


def create(source_id: str) -> str:
    _JOB_DIR.mkdir(exist_ok=True)
    job_id = str(uuid.uuid4())
    _write(job_id, {"status": "processing", "source_id": source_id, "chunks_indexed": 0, "error": None})
    return job_id


def complete(job_id: str, chunks_indexed: int) -> None:
    _update(job_id, {"status": "done", "chunks_indexed": chunks_indexed, "error": None})


def fail(job_id: str, error: str) -> None:
    _update(job_id, {"status": "failed", "error": error})


def get(job_id: str) -> dict | None:
    path = _path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── private ──────────────────────────────────────────────────────────────────

def _path(job_id: str) -> Path:
    return _JOB_DIR / f"{job_id}.json"


def _write(job_id: str, data: dict) -> None:
    _path(job_id).write_text(json.dumps(data))


def _update(job_id: str, updates: dict) -> None:
    current = get(job_id) or {}
    current.update(updates)
    _write(job_id, current)
