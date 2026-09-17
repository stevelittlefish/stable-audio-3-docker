# stable-audio-3-docker

A dockerised, headless **fork of [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3)**
(Stability AI's open text-to-audio / music generation model), wrapped up as a
backend for **[ASS — the Audio Slop Server](https://github.com/stevelittlefish/AudioSlopServer)**.

Upstream ships a research repo and a Gradio UI. This fork keeps the model exactly
as it is and puts a job-queue HTTP API in front of it, so ASS can start it, run
generation jobs, harvest results, and shove the model off the GPU when another
backend needs the card — all over HTTP, no shared volumes.

**If you want the model itself** — its capabilities, the small/medium/large
variants, LoRA training, the CLI, the optimized CPU/MLX/TensorRT runtimes, the
prompting guide — go to [upstream](https://github.com/Stability-AI/stable-audio-3).
This README is only about the fork: what it adds, and how to run it.

---

## What this fork adds

Everything upstream does is untouched. On top of it:

- **A headless FastAPI service** (`run_api.py`) in place of the Gradio UI —
  text-to-audio, init-audio variation, inpainting, batch, LoRA control and
  transcoding, all over an async submit → poll → download job queue. (The Gradio
  prompt assistant is dropped.)
- **The ASS backend contract** — a finished job enumerates its outputs as typed
  **artifacts**, downloadable by name from `/v1/jobs/{id}/result/{name}`. See
  [The ASS contract](#the-ass-contract).
- **`/park` + `/unpark`** — move the whole model between GPU and CPU RAM so ASS
  can free the card for another backend without a cold container restart.
- **A GHCR image + release CI** — tag `v*` and GitHub Actions builds and pushes
  `ghcr.io/stevelittlefish/stable-audio-3-docker`.
- **The shared `/cache` convention** — `HF_HOME` and `TORCH_HOME` under `/cache`,
  so every ASS backend can share one weight cache and one Hugging Face token.

---

## Deploying it

The weights (`stabilityai/stable-audio-3-medium`) are gated. Before the first
run, accept the terms on the
[model page](https://huggingface.co/stabilityai/stable-audio-3-medium) with your
Hugging Face account and get a read-only token. The model download is ~10 GB; it
lands in the shared cache and is reused across rebuilds.

### As an ASS backend (the point of this repo)

ASS pulls and runs the image for you — you just point a service at it in
`ass.toml`:

```toml
[services.stableaudio]
image = "ghcr.io/stevelittlefish/stable-audio-3-docker:latest"
port  = 5335
verb  = "generate"
evict = "park"
shm_size_mb = 8192
volumes = ["/srv/ass/cache:/cache", "/srv/ass/outputs/stableaudio:/app/outputs"]
```

The gated weights authenticate via the HF token in the shared cache
(`/srv/ass/cache/huggingface/token`) — never in a committed config. Pull the
image with ASS's `./pull-services.sh`, then start ASS. See the ASS README for the
full picture.

### Standalone with Docker Compose

For running it on its own (dev, or a box without ASS):

```bash
export HF_TOKEN=hf_your_read_only_token
docker compose up -d --build
docker compose logs -f stable-audio-3
```

The REST API is published on port `5335`; interactive OpenAPI docs are at
`/docs`. To load LoRAs at startup, edit the `command:` in `compose.yaml`. Stop
with `docker compose down`.

### Building the image by hand

```bash
docker build -t stable-audio-3-docker:local .
```

The image is a chonker (CUDA devel + PyTorch + Flash Attention 2), so the first
build is slow; layer caching keeps rebuilds cheap. Weights are **not** baked in —
they're fetched at runtime into `/cache` — so no HF token is needed at build time.

### Cutting a release

```bash
./make_release.sh v0.1.0 "First release"
```

That tags and pushes, which fires `.github/workflows/release.yml` to build and
publish `ghcr.io/stevelittlefish/stable-audio-3-docker` (`vX.Y.Z`, `vX.Y`, `vX`,
`latest`).

---

## The API at a glance

Generation is asynchronous and serialized on a single worker (the model isn't
thread-safe and the GPU runs one job at a time).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness (no auth); reports `parked` |
| `GET` | `/v1/info` | Model, sample rate, defaults, LoRAs, `parked` |
| `POST` | `/v1/generate` | Submit a job (`multipart/form-data`) → `{job_id, state}` |
| `GET` | `/v1/jobs/{id}` | Job state and `artifacts[]` |
| `GET` | `/v1/jobs/{id}/result/{name}` | Download one named artifact |
| `DELETE` | `/v1/jobs/{id}` | Delete a job and its files |
| `POST` | `/park` / `/unpark` | (orchestrator) move the model off / onto the GPU |

`POST /v1/generate` takes a `params` form field holding a JSON-encoded request
(only `prompt` is required), plus optional `init_audio` / `inpaint_audio` file
parts. Everything upstream's generator supports — `negative_prompt`,
`seconds_total`, `steps`, `cfg_scale`, `seed`, `batch_size`, `file_format`,
`init_noise_level`, `inpaint_mask_starts`/`ends`, `loras`, … — is accepted; see
`stable_audio_3/api/schemas.py` for the full list and defaults, or the live
`/docs`.

```bash
# submit -> {"job_id":"a1b2c3...","state":"queued"}
curl -s -F 'params={"prompt":"deep dub techno, rolling bassline","seconds_total":30}' \
  http://localhost:5335/v1/generate

# poll -> queued -> running -> succeeded
curl -s http://localhost:5335/v1/jobs/a1b2c3...

# download (names come from the job's artifacts[] list)
curl -o out.wav http://localhost:5335/v1/jobs/a1b2c3.../result/output_0.wav
```

Auth is off by default; set `SAO_API_KEY` in the container to require a
`Authorization: Bearer <key>` (or `X-API-Key`) header on every request but
`/health`.

### The ASS contract

A finished job returns its outputs as **artifacts** — one entry per file, each
with the real MIME type, so ASS never has to assume everything is audio:

```jsonc
{
  "job_id": "a1b2c3...",
  "state": "succeeded",
  "artifacts": [
    { "name": "output_0.wav",      "kind": "audio",    "content_type": "audio/wav", "bytes": 5292044 },
    { "name": "spectrogram_0.png", "kind": "metadata", "content_type": "image/png", "bytes": 81234 }
  ]
}
```

Each artifact's `name` is its filename; download it from
`/v1/jobs/{id}/result/{name}`. A batch (`batch_size` > 1) yields `output_0.wav`,
`output_1.wav`, … each with its spectrogram.

`/park` moves the whole model — DiT, pretransform, conditioner — to CPU RAM and
calls `torch.cuda.empty_cache()` so the VRAM actually returns to the driver;
`/unpark` copies it back. Both are serialized against a running generation by a
GPU lock, so ASS can hand the card to another backend and take it back without a
cold reload of the multi-GB checkpoint.

---

## Credits & thanks

All the hard part — the model, the SAME autoencoder, the inference code, the
training and optimization work — is **Stability AI's**. Enormous thanks to them
and the Harmonai research team for releasing Stable Audio 3 with open weights and
a permissively-licensed codebase. This fork is a thin operational wrapper; the
brains are entirely theirs.

- **Upstream:** [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
- **Model weights:** [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
- **Foundational research:** [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)
- **Community:** [Harmonai Discord](https://discord.gg/7QM7mtY9uH)

If you build on this, credit and cite the upstream authors (below), not the fork.

## License

The **code** is MIT (© Stability AI — see [LICENSE](LICENSE)); this fork's
additions are released under the same terms. The **model weights** are covered by
the [Stability AI Community License](https://stability.ai/license) — read it
before deploying, especially for commercial use.

## Citation

For Stable Audio 3:

```bibtex
@misc{evans2026stableaudio3,
  title={Stable Audio 3},
  author={Zach Evans and Julian D. Parker and Matthew Rice and CJ Carr and Zack Zukowski and Josiah Taylor and Jordi Pons},
  year={2026},
  eprint={2605.17991},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2605.17991}
}
```

For SAME (the autoencoder):

```bibtex
@misc{parker2026SAME,
  title={SAME: A Semantically-Aligned Music Autoencoder},
  author={Julian D. Parker and Zach Evans and CJ Carr and Zack Zukowski and Josiah Taylor and Matthew Rice and Jordi Pons},
  year={2026},
  eprint={2605.18613},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2605.18613}
}
```
