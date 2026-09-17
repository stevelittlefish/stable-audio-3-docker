"""Entrypoint for the Stable Audio 3 FastAPI service.

Loads the model once, then serves the API with a single uvicorn worker (the job
store is in-memory and the GPU is used serially, so multiple workers are unsafe).
Replaces run_gradio.py.
"""

import os
import sys

# Silence library warnings unless --verbose, before any ML imports (they fire at
# import time). Mirrors run_gradio.py.
if "--verbose" not in sys.argv:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings

    warnings.filterwarnings("ignore")

import torch  # noqa: E402
import uvicorn  # noqa: E402

from stable_audio_3 import StableAudioModel  # noqa: E402
from stable_audio_3.api.server import create_app  # noqa: E402
from stable_audio_3.verbose import set_verbose  # noqa: E402


def main(args):
    set_verbose(args.verbose)
    torch.manual_seed(42)

    # The model loads HERE, before the server reports ready — so /health only goes
    # 200 once the weights are actually usable. On a cold cache from_pretrained
    # quietly pulls a multi-GB checkpoint from HuggingFace, which looks exactly like
    # a hung process if you don't say anything. So: say something.
    print(f"[startup] loading model '{args.model}' — on a cold cache this downloads "
          f"several GB from HuggingFace and can take a few minutes. It is NOT hung, "
          f"it is just slow. Grab a coffee.", flush=True)
    # Let HF's own progress bars through even in the default (non-verbose) run — a
    # silent download is the whole problem we're fixing here.
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    model = StableAudioModel.from_pretrained(args.model, model_half=args.model_half)
    print(f"[startup] weights ready — model '{args.model}' is loaded. Bringing the "
          f"API up now.", flush=True)
    if args.lora_ckpt_path:
        model.load_lora(args.lora_ckpt_path)

    api_key = os.environ.get("SAO_API_KEY") or None
    app = create_app(
        model=model,
        model_name=args.model,
        output_root=args.output_root,
        api_key=api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Stable Audio 3 API server")
    parser.add_argument("--model", type=str, required=True, help="Name of pretrained model")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5335)
    parser.add_argument("--output-root", type=str, default="/app/outputs/jobs",
                        help="Directory for rendered job outputs")
    parser.add_argument("--model-half", action="store_true", default=True,
                        help="Use half precision")
    parser.add_argument("--lora-ckpt-path", type=str, nargs="*",
                        help="Path(s) for LoRA(s) to apply. Can specify multiple.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed load/generation progress")
    args = parser.parse_args()
    main(args)
