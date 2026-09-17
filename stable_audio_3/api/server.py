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
    Artifact,
    GenerateRequest,
    JobCreated,
    JobStatus,
    ModelInfo,
)
from .worker import Job, JobManager

# Map an output file extension to its real MIME type, so the contract's
# `content_type` isn't a lie. Clips are audio, spectrograms are PNGs — ASS treats
# content_type as authoritative and doesn't assume everything is audio/*.
_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".png": "image/png",
}


def _content_type(path: str) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def _artifact_paths(job: Job) -> "dict[str, str]":
    """Flatten a job's outputs into {artifact_name -> on-disk path}.

    One audio clip per batch element, plus its spectrogram PNG when present. The
    artifact name is the file's basename, which is unique per job and is exactly
    what /v1/jobs/{id}/result/{name} looks up.
    """
    paths: "dict[str, str]" = {}
    for o in job.outputs:
        for key in ("audio_path", "spectrogram_path"):
            p = o.get(key)
            if p:
                paths[Path(p).name] = p
    return paths


def _to_status(job: Job, manager: JobManager) -> JobStatus:
    artifacts = [
        Artifact(
            name=name,
            kind="metadata" if path.lower().endswith(".png") else "audio",
            content_type=_content_type(path),
            bytes=Path(path).stat().st_size if Path(path).exists() else 0,
        )
        for name, path in _artifact_paths(job).items()
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
        artifacts=artifacts,
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
        # A parked backend is still "up" for readiness; it just has no weights on
        # the GPU. ASS unparks it before sending work.
        return {"status": "ok", "parked": manager.is_parked()}

    # ASS's park/unpark: our fork's addition so ASS can free the GPU for another
    # model without a full container restart (evict = "park"). No auth — same as
    # /health, these are orchestrator-plane, not user-plane.
    @app.post("/park")
    def park():
        manager.park()
        return {"parked": True}

    @app.post("/unpark")
    def unpark():
        manager.unpark()
        return {"parked": False}

    @app.get("/v1/info", response_model=ModelInfo, dependencies=[Depends(require_key)])
    def model_info():
        d = resolve_defaults(model)
        sr = model.model_config["sample_rate"]
        size = model.model_config["sample_size"]
        return ModelInfo(
            model=model_name,
            parked=manager.is_parked(),
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

    @app.get("/v1/jobs/{job_id}/result/{name}", dependencies=[Depends(require_key)])
    def get_result(job_id: str, name: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        path = _artifact_paths(job).get(name)
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Artifact {name!r} not available.")
        return FileResponse(path, filename=Path(path).name)

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
