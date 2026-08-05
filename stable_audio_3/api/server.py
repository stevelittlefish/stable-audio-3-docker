"""FastAPI application factory for the Stable Audio 3 service.

Workflows (all POST /v1/generate): text-to-audio, init-audio variation
(upload `init_audio`), inpainting (upload `inpaint_audio` + mask regions),
per-LoRA control, batch, output transcoding, and spectrograms.

Not included, by design: the prompt assistant. Not yet wired: RF-inversion
(unimplemented in the inference path) and step-preview streaming.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .generation import resolve_defaults
from .schemas import (
    FILE_FORMATS,
    GenerateRequest,
    JobCreated,
    JobStatus,
    ModelInfo,
    OutputItem,
)
from .worker import Job, JobManager


def _to_status(job: Job, manager: JobManager) -> JobStatus:
    outputs = [
        OutputItem(
            index=o["index"],
            seed=o["seed"],
            format=o["format"],
            duration_seconds=o["duration_seconds"],
            audio_url=f"/v1/jobs/{job.job_id}/audio?index={o['index']}",
            spectrogram_url=(
                f"/v1/jobs/{job.job_id}/spectrogram?index={o['index']}"
                if o.get("spectrogram_path")
                else None
            ),
        )
        for o in job.outputs
    ]
    return JobStatus(
        job_id=job.job_id,
        state=job.state,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        queue_position=manager.queue_position(job.job_id),
        request=job.request,
        outputs=outputs,
    )


def create_app(model, model_name: str, output_root: Path, api_key: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app around an already-loaded model."""
    app = FastAPI(title="Stable Audio 3 API", version="1.0")
    manager = JobManager(model, output_root)

    def require_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
        """Optional bearer/X-API-Key auth. No-op when SAO_API_KEY is unset."""
        if not api_key:
            return
        supplied = x_api_key or (authorization[7:] if authorization and authorization.startswith("Bearer ") else None)
        if supplied != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/model", response_model=ModelInfo, dependencies=[Depends(require_key)])
    def model_info():
        d = resolve_defaults(model)
        sr = model.model_config["sample_rate"]
        size = model.model_config["sample_size"]
        return ModelInfo(
            model=model_name,
            sample_rate=sr,
            sample_size=size,
            max_duration_seconds=size / sr,
            diffusion_objective=model.model.diffusion_objective,
            default_steps=d["steps"],
            default_cfg_scale=d["cfg_scale"],
            default_sampler_type=d["sampler_type"],
            sampler_types=d["sampler_types"],
            loras=list(getattr(model.model, "lora_names", []) or []),
        )

    @app.post("/v1/generate", response_model=JobCreated, dependencies=[Depends(require_key)])
    async def generate(
        params: str = Form(..., description="JSON-encoded GenerateRequest"),
        init_audio: Optional[UploadFile] = File(None),
        inpaint_audio: Optional[UploadFile] = File(None),
    ):
        try:
            request = GenerateRequest.model_validate_json(params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid params: {exc}")

        if request.file_format not in FILE_FORMATS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown file_format {request.file_format!r}. Valid: {list(FILE_FORMATS)}",
            )

        init_path = _save_upload(init_audio)
        inpaint_path = _save_upload(inpaint_audio)
        job = manager.submit(request, init_audio_path=init_path, inpaint_audio_path=inpaint_path)
        return JobCreated(job_id=job.job_id, state=job.state)

    @app.get("/v1/jobs", response_model=List[JobStatus], dependencies=[Depends(require_key)])
    def list_jobs():
        return [_to_status(j, manager) for j in manager.list()]

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_key)])
    def get_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _to_status(job, manager)

    @app.get("/v1/jobs/{job_id}/audio", dependencies=[Depends(require_key)])
    def get_audio(job_id: str, index: int = 0):
        return _output_file(manager, job_id, index, "audio_path")

    @app.get("/v1/jobs/{job_id}/spectrogram", dependencies=[Depends(require_key)])
    def get_spectrogram(job_id: str, index: int = 0):
        return _output_file(manager, job_id, index, "spectrogram_path")

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def delete_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        shutil.rmtree(job.job_dir, ignore_errors=True)
        manager.remove(job_id)
        return {"deleted": job_id}

    return app


def _save_upload(upload: Optional[UploadFile]) -> Optional[str]:
    """Persist an uploaded file to a temp path, or None if no file was sent."""
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


def _output_file(manager: JobManager, job_id: str, index: int, key: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if index < 0 or index >= len(job.outputs):
        raise HTTPException(status_code=404, detail="Output index out of range.")
    path = job.outputs[index].get(key)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Requested file not available.")
    return FileResponse(path, filename=Path(path).name)
