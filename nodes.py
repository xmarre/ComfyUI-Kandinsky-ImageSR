"""ComfyUI nodes for Kandinsky ImageSR 2.0."""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .kandinsky_sr import config as cfg
from .kandinsky_sr.loader import discover_bundles
from .kandinsky_sr.runtime import (
    ACCUMULATION_MODES,
    VAE_DTYPES,
    WEIGHT_DTYPES,
    KandinskySRModel,
    load_model,
    upscale,
)

logger = logging.getLogger(__name__)

KandinskySRModelType = io.Custom("KANDINSKY_SR_MODEL")

NO_MODEL = "(no Kandinsky ImageSR bundle found)"
FACTOR_OPTIONS = {"2x": 2, "4x": 4}
CATEGORY = "image/upscaling/Kandinsky ImageSR"


def _search_roots() -> list[str]:
    import folder_paths

    return list(folder_paths.get_folder_paths("diffusion_models"))


def _bundle_names() -> list[str]:
    names = list(discover_bundles(_search_roots()))
    return names or [NO_MODEL]


class KandinskyImageSRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyImageSRLoader",
            display_name="Kandinsky ImageSR Loader",
            category=CATEGORY,
            description=(
                "Loads an unmodified Kandinsky-ImageSR-2.0 Hugging Face bundle (baked transformer and FLUX VAE) "
                "from a folder inside models/diffusion_models. The bundle is validated tensor by tensor before "
                "any weights are read."
            ),
            search_aliases=["kandinsky", "imagesr", "super resolution", "upscale", "restoration"],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_bundle_names(),
                    tooltip=(
                        "Bundle folder found in models/diffusion_models (or a configured extra path). The folder must "
                        "contain model_index.json, transformer/ and vae/ exactly as published on Hugging Face."
                    ),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=list(WEIGHT_DTYPES),
                    default="default",
                    tooltip=(
                        "Transformer precision. 'default' picks bf16 on hardware that supports it and fp32 otherwise. "
                        "bf16 reproduces the official mixed-precision inference (modulation and conditioning tables "
                        "stay fp32). fp32 doubles weight memory and is mainly useful on GPUs without bf16 support."
                    ),
                ),
                io.Combo.Input(
                    "vae_dtype",
                    options=list(VAE_DTYPES),
                    default="default",
                    tooltip="FLUX VAE precision. 'default' follows ComfyUI's VAE precision selection (bf16 or fp32).",
                ),
            ],
            outputs=[KandinskySRModelType.Output(display_name="sr_model")],
        )

    @classmethod
    def execute(cls, model_name: str, weight_dtype: str, vae_dtype: str) -> io.NodeOutput:
        bundles = discover_bundles(_search_roots())
        if model_name not in bundles:
            raise ValueError(
                f"Kandinsky ImageSR bundle {model_name!r} was not found. Download "
                "https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0 and place the whole repository folder "
                "(model_index.json, transformer/, vae/) in ComfyUI/models/diffusion_models/, then refresh."
            )
        model = load_model(bundles[model_name], model_name, weight_dtype=weight_dtype, vae_dtype=vae_dtype)
        return io.NodeOutput(model)


class KandinskyImageSRUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyImageSRUpscale",
            display_name="Kandinsky ImageSR Upscale",
            category=CATEGORY,
            description=(
                "One-step generative super-resolution and restoration with Kandinsky ImageSR 2.0. The image is "
                "processed in fixed 1024x1024 output tiles (one transformer evaluation per tile) blended with "
                "boundary-aware Hann windows; the output is exactly input size times the upscale factor."
            ),
            search_aliases=["kandinsky", "imagesr", "super resolution", "upscale", "restore"],
            inputs=[
                KandinskySRModelType.Input("sr_model"),
                io.Image.Input("image"),
                io.Combo.Input(
                    "upscale_factor",
                    options=list(FACTOR_OPTIONS),
                    default="2x",
                    tooltip="Output scale. Tiles are 512 px (2x) or 256 px (4x) of the input, 1024 px of the output.",
                ),
                io.Int.Input(
                    "creativity",
                    default=cfg.DEFAULT_CREATIVITY,
                    min=0,
                    max=cfg.MAX_CREATIVITY,
                    step=1,
                    tooltip=(
                        "Calibrated restoration strength (0-8; 4 is the official default). Higher values correct the "
                        "input more aggressively: more reconstructed texture and detail, but also more change to "
                        "existing content, including faces and fine structure. Higher is not automatically better; "
                        "use low values when fidelity to the input matters."
                    ),
                ),
                io.Float.Input(
                    "overlap",
                    default=cfg.DEFAULT_OVERLAP,
                    min=cfg.MIN_OVERLAP,
                    max=cfg.MAX_OVERLAP,
                    step=0.01,
                    tooltip=(
                        "Fraction of each input tile shared with its neighbour (official default 0.25). Larger "
                        "overlap blends seams over a wider band but increases the number of tiles."
                    ),
                ),
                io.Int.Input(
                    "tile_batch_size",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Tiles processed together. 0 = automatic (sized from free VRAM after the models are loaded, "
                        "at most 8). On out-of-memory the batch is halved and retried without duplicating work."
                    ),
                ),
                io.Combo.Input(
                    "accumulation_device",
                    options=list(ACCUMULATION_MODES),
                    default="auto",
                    tooltip=(
                        "Where the FP32 blend buffer (12 bytes per output pixel) lives. 'auto' uses the GPU only when "
                        "it fits comfortably in free VRAM, otherwise system RAM."
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        sr_model: KandinskySRModel,
        image,
        upscale_factor: str,
        creativity: int,
        overlap: float,
        tile_batch_size: int,
        accumulation_device: str,
    ) -> io.NodeOutput:
        if not isinstance(sr_model, KandinskySRModel):
            raise TypeError("sr_model must come from the Kandinsky ImageSR Loader node")
        if upscale_factor not in FACTOR_OPTIONS:
            raise ValueError(f"upscale_factor must be one of {list(FACTOR_OPTIONS)}, got {upscale_factor!r}")
        output, _stats = upscale(
            sr_model,
            image,
            FACTOR_OPTIONS[upscale_factor],
            int(creativity),
            overlap=float(overlap),
            tile_batch_size=int(tile_batch_size),
            accumulation_device=accumulation_device,
        )
        return io.NodeOutput(output)


class KandinskyImageSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [KandinskyImageSRLoader, KandinskyImageSRUpscale]


async def comfy_entrypoint() -> KandinskyImageSRExtension:
    return KandinskyImageSRExtension()
