"""FastAPI service for Stable Audio 3.

A headless replacement for the Gradio interface. Exposes every generation
workflow (text-to-audio, init-audio variation, inpainting, LoRA control,
batch, output transcoding, spectrograms) over an HTTP job queue. The Gradio
prompt assistant is intentionally not included.
"""
