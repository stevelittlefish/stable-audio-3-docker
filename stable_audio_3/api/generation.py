"""Core generation logic, free of any Gradio dependency.

This is the distilled form of ``interface.diffusion_cond.generate_cond``: it
builds the ``model.generate`` arguments, runs inference, and encodes each batch
element to a file (plus an optional spectrogram). It imports neither Gradio nor
the prompt assistant.
"""

from __future__ import annotations

import gc
import os
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from einops import rearrange

from stable_audio_3.inference.distribution_shift import (
    DistributionShift,
    FluxDistributionShift,
    IdentityDistributionShift,
    LogSNRShift,
)
from .schemas import DistShift, GenerateRequest

# Audio input, as model.generate expects it: (sample_rate, waveform[C, N]).
AudioInput = Tuple[int, torch.Tensor]

# ffmpeg command templates keyed by the file_format string. wav is written
# directly by torchaudio and never hits ffmpeg.
_FFMPEG_TEMPLATES = {
    "m4a aac_he_v2 32k": '-c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k',
    "m4a aac_he_v2 64k": '-c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k',
    "flac": "",
    "mp3 320k": "-b:a 320k",
    "mp3 v0": "-q:a 0",
    "mp3 128k": "-b:a 128k",
}


def resolve_defaults(model) -> dict:
    """Objective-dependent defaults, mirroring create_sampling_ui.

    Returns steps, cfg_scale, sampler_type, sigma_max defaults plus the valid
    sampler list for the loaded model.
    """
    objective = model.model.diffusion_objective
    if objective == "rectified_flow":
        return {
            "steps": 50,
            "cfg_scale": 7.0,
            "sampler_type": "euler",
            "sampler_types": ["euler", "rk4", "dpmpp"],
            "sigma_max": 1.0,
        }
    if objective == "rf_denoiser":
        return {
            "steps": 8,
            "cfg_scale": 1.0,
            "sampler_type": "pingpong",
            "sampler_types": ["pingpong"],
            "sigma_max": 1.0,
        }
    # Plain v-diffusion.
    return {
        "steps": 100,
        "cfg_scale": 7.0,
        "sampler_type": "dpmpp-3m-sde",
        "sampler_types": [
            "dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms",
            "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast",
            "v-ddim", "v-ddim-cfgpp",
        ],
        "sigma_max": 100.0,
    }


def build_dist_shift(ds: Optional[DistShift]):
    """Construct a distribution-shift object from the request, or None."""
    if ds is None:
        return None
    p = ds.params
    if ds.type == "LogSNR":
        return LogSNRShift(
            anchor_length=int(p.get("anchor_length", 2000)),
            anchor_logsnr=p.get("anchor_logsnr", -6.2),
            rate=p.get("rate", 0.0),
            logsnr_end=p.get("logsnr_end", 2.0),
        )
    if ds.type == "Flux":
        return FluxDistributionShift(
            min_length=int(p.get("min_length", 256)),
            max_length=int(p.get("max_length", 4096)),
            alpha_min=p.get("alpha_min", 6.93),
            alpha_max=p.get("alpha_max", 6.93),
        )
    if ds.type == "Full":
        return DistributionShift(
            base_shift=p.get("base_shift", 0.5),
            max_shift=p.get("max_shift", 1.15),
            min_length=int(p.get("min_length", 256)),
            max_length=int(p.get("max_length", 4096)),
        )
    return IdentityDistributionShift()  # "None"


def _validate_inpaint_regions(starts, ends):
    """Return (start, end) sized for model.generate, or (None, None).

    A single region collapses to a float pair; multiple regions stay as lists.
    Raises ValueError on malformed input (the API turns this into a 400).
    """
    if not starts and not ends:
        return None, None
    if not starts or not ends:
        raise ValueError("Inpaint starts and ends must both be provided.")
    if len(starts) != len(ends):
        raise ValueError("Inpaint starts and ends must contain the same number of regions.")
    if any(s < 0 or e <= s for s, e in zip(starts, ends)):
        raise ValueError("Each inpaint region must have a non-negative start before its end.")
    if len(starts) == 1:
        return starts[0], ends[0]
    return starts, ends


def load_audio_upload(path: str) -> AudioInput:
    """Load an uploaded audio file into the (sample_rate, waveform) form."""
    waveform, sr = torchaudio.load(path)
    return sr, waveform


def encode_audio(audio_i16: torch.Tensor, sample_rate: int, out_path: Path, file_format: str) -> Path:
    """Write one clip to disk, transcoding via ffmpeg for non-wav formats.

    ``audio_i16`` is int16 [channels, samples]. Returns the final output path.
    """
    ext = file_format.split(" ")[0].lower() if file_format else "wav"
    wav_path = out_path.with_suffix(".wav")
    torchaudio.save(str(wav_path), audio_i16, sample_rate)

    if file_format == "wav" or not file_format:
        return wav_path

    final_path = out_path.with_suffix(f".{ext}")
    extra = _FFMPEG_TEMPLATES.get(file_format, "")
    cmd = f'ffmpeg -i "{wav_path}" {extra} -y "{final_path}" -loglevel error'
    subprocess.run(cmd, shell=True, check=True)
    wav_path.unlink(missing_ok=True)
    return final_path


def run_generation(
    model,
    req: GenerateRequest,
    init_audio: Optional[AudioInput],
    inpaint_audio: Optional[AudioInput],
    job_dir: Path,
) -> List[dict]:
    """Run one generation request and write its outputs into ``job_dir``.

    Returns a list of per-clip metadata dicts (index, seed, format,
    duration_seconds, audio_path, spectrogram_path).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    defaults = resolve_defaults(model)
    sample_rate = model.model_config["sample_rate"]
    sample_size = model.model_config["sample_size"]
    max_seconds = sample_size // sample_rate

    steps = req.steps if req.steps is not None else defaults["steps"]
    cfg_scale = req.cfg_scale if req.cfg_scale is not None else defaults["cfg_scale"]
    sampler_type = req.sampler_type or defaults["sampler_type"]
    sigma_max = req.sigma_max if req.sigma_max is not None else defaults["sigma_max"]
    seconds_total = req.seconds_total if req.seconds_total is not None else max_seconds

    # Resolve the seed up front so we can report exactly what was used.
    seed = req.seed if req.seed is not None and req.seed >= 0 else random.randint(0, 2**31 - 1)

    # Per-LoRA controls, matching the trailing-args handling in generate_cond.
    lora_configs = None
    if req.loras:
        lora_configs = []
        for i, lc in enumerate(req.loras):
            model.set_lora_strength(lc.strength, lora_index=i)
            lora_configs.append({
                "lora_index": i,
                "interval": (lc.interval_min, lc.interval_max),
                "layer_filter": lc.layer_filter,
            })

    start, end = _validate_inpaint_regions(req.inpaint_mask_starts, req.inpaint_mask_ends)
    if inpaint_audio is not None and start is None:
        raise ValueError("Inpainting requires at least one inpaint start/end region.")

    generate_args = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "duration": seconds_total,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": (req.cfg_interval_min, req.cfg_interval_max),
        "lora_configs": lora_configs,
        "batch_size": int(req.batch_size),
        "sample_size": sample_size,
        "seed": seed,
        "sampler_type": sampler_type,
        "sigma_max": sigma_max,
        "init_audio": init_audio,
        "init_noise_level": req.init_noise_level,
        "scale_phi": req.cfg_rescale,
        "cfg_norm_threshold": req.cfg_norm_threshold,
        "apg_scale": req.apg_scale,
        "duration_padding_sec": req.duration_padding_sec,
        "dist_shift": build_dist_shift(req.dist_shift),
    }
    if inpaint_audio is not None:
        generate_args.update({
            "inpaint_audio": inpaint_audio,
            "inpaint_mask_start_seconds": start,
            "inpaint_mask_end_seconds": end,
        })

    audio = model.generate(**generate_args)  # [batch, channels, samples]

    if req.cut_to_seconds_total:
        audio = audio[:, :, : int(seconds_total * sample_rate)]

    outputs = []
    for i in range(audio.shape[0]):
        clip = audio[i]  # [channels, samples]
        clip_i16 = clip.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        audio_path = encode_audio(clip_i16, sample_rate, job_dir / f"output_{i}", req.file_format)

        spectrogram_path = None
        if req.return_spectrogram:
            spectrogram_path = _write_spectrogram(clip_i16, sample_rate, job_dir / f"spectrogram_{i}.png")

        outputs.append({
            "index": i,
            "seed": seed,
            "format": req.file_format,
            "duration_seconds": clip.shape[-1] / sample_rate,
            "audio_path": str(audio_path),
            "spectrogram_path": str(spectrogram_path) if spectrogram_path else None,
        })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return outputs


def _write_spectrogram(audio_i16: torch.Tensor, sample_rate: int, out_path: Path) -> Path:
    """Render a mel-spectrogram PNG. Imported lazily to keep matplotlib optional."""
    from stable_audio_3.interface.aeiou import audio_spectrogram_image

    image = audio_spectrogram_image(audio_i16, sample_rate=sample_rate)
    image.save(str(out_path))
    return out_path
