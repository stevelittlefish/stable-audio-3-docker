"""Pydantic request/response models for the Stable Audio 3 API.

These mirror the controls of the Gradio interface. Objective-dependent knobs
(steps, cfg_scale, sampler_type, sigma_max) default to ``None`` and are resolved
at generation time against the loaded model — see ``generation.resolve_defaults``.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


# Output container formats, matching the Gradio dropdown. The value is passed
# verbatim to the ffmpeg transcode step (see generation.encode_audio).
FILE_FORMATS = (
    "wav",
    "flac",
    "mp3 320k",
    "mp3 v0",
    "mp3 128k",
    "m4a aac_he_v2 64k",
    "m4a aac_he_v2 32k",
)


class DistShift(BaseModel):
    """Sampling-schedule distribution shift.

    ``type`` selects the shift; ``params`` carries its parameters. Only relevant
    for rectified-flow / rf-denoiser models. Omit the whole object to use the
    model's built-in ``sampling_dist_shift``.

    Expected ``params`` keys per type:
      * LogSNR: anchor_length, anchor_logsnr, rate, logsnr_end
      * Flux:   min_length, max_length, alpha_min, alpha_max
      * Full:   base_shift, max_shift, min_length, max_length
      * None:   (identity — no params)
    """

    type: str = Field(description="One of: LogSNR, Flux, Full, None")
    params: Dict[str, float] = Field(default_factory=dict)


class LoraConfig(BaseModel):
    """Per-adapter LoRA control. One entry per loaded LoRA, in load order."""

    strength: float = 1.0
    interval_min: float = 0.0
    interval_max: float = 1.0
    layer_filter: str = ""


class GenerateRequest(BaseModel):
    """All generation parameters. Only ``prompt`` is required."""

    prompt: str
    negative_prompt: Optional[str] = None

    # Duration in seconds. None => the model's maximum length.
    seconds_total: Optional[float] = None

    # Objective-dependent; None => resolved from the model at generation time.
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    sampler_type: Optional[str] = None
    sigma_max: Optional[float] = None

    seed: int = -1
    batch_size: int = Field(default=1, ge=1)

    # CFG / guidance knobs
    cfg_interval_min: float = 0.0
    cfg_interval_max: float = 1.0
    cfg_rescale: float = 0.0
    cfg_norm_threshold: float = 0.0
    apg_scale: float = 1.0

    duration_padding_sec: float = 6.0
    cut_to_seconds_total: bool = True

    # Output
    file_format: str = "wav"
    return_spectrogram: bool = True

    # Init-audio variation (audio uploaded as the `init_audio` file part).
    # 0.01 = tiny change, 1.0 = ignore the init audio.
    init_noise_level: float = 0.9

    # Inpainting (audio uploaded as the `inpaint_audio` file part). Regions are
    # given as matching lists of start/end seconds.
    inpaint_mask_starts: Optional[List[float]] = None
    inpaint_mask_ends: Optional[List[float]] = None

    dist_shift: Optional[DistShift] = None
    loras: Optional[List[LoraConfig]] = None


class OutputItem(BaseModel):
    """One rendered clip. A batch produces one item per batch element."""

    index: int
    seed: int
    format: str
    duration_seconds: float
    audio_url: str
    spectrogram_url: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    error: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    queue_position: Optional[int] = None
    request: Optional[GenerateRequest] = None
    outputs: List[OutputItem] = Field(default_factory=list)


class JobCreated(BaseModel):
    job_id: str
    state: JobState


class ModelInfo(BaseModel):
    model: str
    sample_rate: int
    sample_size: int
    max_duration_seconds: float
    diffusion_objective: str
    default_steps: int
    default_cfg_scale: float
    default_sampler_type: str
    sampler_types: List[str]
    file_formats: List[str] = Field(default_factory=lambda: list(FILE_FORMATS))
    loras: List[str] = Field(default_factory=list)
