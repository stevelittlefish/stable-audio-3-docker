"""In-memory job queue with a single worker thread.

Generation is serialized on one worker because the model is not thread-safe and
the GPU runs one job at a time. This mirrors ACE-Step's design (an in-memory
queue/task store that requires a single worker) and keeps the door open for a
cross-process GPU lease later — the worker is the one place that touches CUDA.

The store is in-memory, so it must run under a single process (uvicorn
--workers 1). Rendered files live on disk under the job's directory.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .generation import load_audio_upload, run_generation
from .schemas import GenerateRequest, JobState


@dataclasses.dataclass
class Job:
    job_id: str
    request: GenerateRequest
    job_dir: Path
    init_audio_path: Optional[str] = None
    inpaint_audio_path: Optional[str] = None
    state: JobState = JobState.queued
    error: Optional[str] = None
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    outputs: List[dict] = dataclasses.field(default_factory=list)


class JobManager:
    """Owns the job store, the work queue, and the worker thread."""

    def __init__(self, model, output_root: Path):
        self._model = model
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []  # submission order, for queue_position
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="sao-worker", daemon=True)
        self._thread.start()

    def submit(self, request: GenerateRequest, init_audio_path=None, inpaint_audio_path=None) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self._output_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            job_id=job_id,
            request=request,
            job_dir=job_dir,
            init_audio_path=init_audio_path,
            inpaint_audio_path=inpaint_audio_path,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in self._order]

    def remove(self, job_id: str) -> None:
        """Drop a job from the store. Its on-disk files are removed by the caller."""
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)

    def queue_position(self, job_id: str) -> Optional[int]:
        """Number of queued/running jobs ahead of this one (0 = next/running)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state not in (JobState.queued, JobState.running):
                return None
            ahead = 0
            for jid in self._order:
                if jid == job_id:
                    break
                if self._jobs[jid].state in (JobState.queued, JobState.running):
                    ahead += 1
            return ahead

    def _run(self):
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            self._process(job)

    def _process(self, job: Job):
        with self._lock:
            job.state = JobState.running
            job.started_at = time.time()
        try:
            init_audio = load_audio_upload(job.init_audio_path) if job.init_audio_path else None
            inpaint_audio = load_audio_upload(job.inpaint_audio_path) if job.inpaint_audio_path else None
            outputs = run_generation(self._model, job.request, init_audio, inpaint_audio, job.job_dir)
            with self._lock:
                job.outputs = outputs
                job.state = JobState.succeeded
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            with self._lock:
                job.state = JobState.failed
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.finished_at = time.time()
            # Uploaded inputs are only needed during processing.
            for p in (job.init_audio_path, job.inpaint_audio_path):
                if p:
                    Path(p).unlink(missing_ok=True)
