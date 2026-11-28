"""Kandinsky ImageSR 2.0 core: configuration, native transformer, tiling and pipeline.

``config``, ``tiling``, ``model``, ``loader``, ``memory``, ``pipeline`` and ``vae``
depend only on PyTorch (and safetensors for weight reading). ``runtime`` integrates
with ComfyUI and imports ``comfy`` lazily.
"""
