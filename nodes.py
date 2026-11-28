"""ComfyUI nodes for Kandinsky ImageSR 2.0."""

from __future__ import annotations

import dataclasses
import logging

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .kandinsky_sr import config as cfg
from .kandinsky_sr.loader import discover_bundles, is_single_file_checkpoint
from .kandinsky_sr.runtime import (
    ACCUMULATION_MODES,
    RECONSTRUCTIONS,
    RESTORATION_MODES,
    VAE_DTYPES,
    WEIGHT_DTYPES,
    KandinskySRModel,
    load_model,
    upscale,
)

logger = logging.getLogger(__name__)

#: Connection type produced by the loader. A different node pack uses the generic name
#: "KANDINSKY_SR_MODEL" for an incompatible object, so the loader outputs a
#: package-specific type that cannot be wired into that pack's nodes.
MODEL_TYPE = "KANDINSKY_IMAGESR_MODEL"
#: Type name used by earlier versions of these nodes; still accepted by the upscale
#: input so saved workflows keep their links. Foreign objects arriving this way are
#: rejected with an explicit error.
LEGACY_MODEL_TYPE = "KANDINSKY_SR_MODEL"
KandinskySRModelType = io.Custom(MODEL_TYPE)
KandinskySRModelInputType = io.Custom(f"{MODEL_TYPE},{LEGACY_MODEL_TYPE}")

NO_MODEL = "(no Kandinsky ImageSR checkpoint found)"
FACTOR_OPTIONS = {"2x": 2, "4x": 4}
CATEGORY = "image/upscaling/Kandinsky ImageSR"


_SCHEMA_FIELDS = {f.name for f in dataclasses.fields(io.Schema)} if dataclasses.is_dataclass(io.Schema) else set()


def _schema(**kwargs) -> io.Schema:
    """Build a schema, dropping optional fields older ComfyUI releases do not know."""
    optional = ("search_aliases",)
    return io.Schema(**{k: v for k, v in kwargs.items() if k not in optional or k in _SCHEMA_FIELDS})


def _search_roots() -> list[str]:
    import folder_paths

    return list(folder_paths.get_folder_paths("diffusion_models"))


def discover_models() -> dict[str, str]:
    """Single-file checkpoints (listed first) and official bundle folders.

    Single files come from ComfyUI's ``diffusion_models`` file list (all configured
    paths, nested folders included); only their safetensors headers are read, and
    the result is cached per file. Bundles are folders found directly in a
    ``diffusion_models`` root or one level below it.
    """
    import folder_paths

    models: dict[str, str] = {}
    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if path and is_single_file_checkpoint(path):
            models[name] = path
    for name, path in discover_bundles(_search_roots()).items():
        models.setdefault(name, str(path))
    return models


def _model_names() -> list[str]:
    return list(discover_models()) or [NO_MODEL]


class KandinskyImageSRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            node_id="KandinskyImageSRLoader",
            display_name="Kandinsky ImageSR Loader",
            category=CATEGORY,
            description=(
                "Loads the Kandinsky ImageSR 2.0 transformer from models/diffusion_models: the single-file checkpoint "
                "kandinsky_imagesr_2.0.safetensors (use it with the FLUX.1 VAE from Load VAE) or the unmodified "
                "Hugging Face bundle folder. The checkpoint is validated tensor by tensor before any weights are read."
            ),
            search_aliases=["kandinsky", "imagesr", "super resolution", "upscale", "restoration"],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_model_names(),
                    tooltip=(
                        "Single-file checkpoint (kandinsky_imagesr_2.0.safetensors) or official bundle folder in "
                        "models/diffusion_models or a configured extra path. Only Kandinsky ImageSR checkpoints are "
                        "listed."
                    ),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=list(WEIGHT_DTYPES),
                    default="default",
                    tooltip=(
                        "Transformer precision. 'default' picks bf16 on hardware that supports it and fp32 otherwise. "
                        "bf16 reproduces the official mixed-precision inference (conditioning tables stay fp32). fp32 "
                        "doubles weight memory and is mainly useful on GPUs without bf16 support."
                    ),
                ),
                io.Combo.Input(
                    "vae_dtype",
                    options=list(VAE_DTYPES),
                    default="default",
                    tooltip=(
                        "Precision of the official bundle's own VAE, used only when no VAE is connected to the "
                        "upscale node. 'default' follows ComfyUI's VAE precision selection (bf16 or fp32)."
                    ),
                ),
            ],
            outputs=[KandinskySRModelType.Output(display_name="sr_model")],
        )

    @classmethod
    def execute(cls, model_name: str, weight_dtype: str, vae_dtype: str) -> io.NodeOutput:
        models = discover_models()
        if model_name not in models:
            raise ValueError(
                f"Kandinsky ImageSR checkpoint {model_name!r} was not found. Place kandinsky_imagesr_2.0.safetensors "
                "(https://huggingface.co/hmdlsk/Kandinsky-ImageSR-2.0-ComfyUI) or the official bundle folder "
                "(https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0) in ComfyUI/models/diffusion_models/, "
                "then refresh."
            )
        model = load_model(models[model_name], model_name, weight_dtype=weight_dtype, vae_dtype=vae_dtype)
        return io.NodeOutput(model)


class KandinskyImageSRUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            node_id="KandinskyImageSRUpscale",
            display_name="Kandinsky ImageSR Upscale",
            category=CATEGORY,
            description=(
                "One-step generative super-resolution and restoration with Kandinsky ImageSR 2.0. The image is "
                "restored at the native 2x or 4x scale in fixed 1024x1024 output tiles (one transformer evaluation "
                "per tile) blended with boundary-aware Hann windows. By default the output is exactly input size "
                "times the native scale; output_scale optionally resizes the finished restoration to another size."
            ),
            search_aliases=["kandinsky", "imagesr", "super resolution", "upscale", "restore"],
            inputs=[
                KandinskySRModelInputType.Input("sr_model"),
                io.Image.Input("image"),
                io.Combo.Input(
                    "upscale_factor",
                    options=list(FACTOR_OPTIONS),
                    default="2x",
                    display_name="native restoration scale",
                    tooltip=(
                        "Scale at which the model restores the image: one of its two calibrated factors. It sets the "
                        "tile geometry (512 px input tiles at 2x, 256 px at 4x, 1024 px output tiles), the cost, and "
                        "the meaning of creativity. The output has this scale unless output_scale is set."
                    ),
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
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "FLUX.1 VAE (ae.safetensors from Load VAE). Required for the single-file checkpoint; optional "
                        "for the official bundle, whose own VAE is used when nothing is connected. The connected VAE "
                        "is used as is and not modified."
                    ),
                ),
                io.Float.Input(
                    "output_scale",
                    default=0.0,
                    min=0.0,
                    max=4.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Final size relative to the input. 0 = native restoration size. Otherwise 1.0 up to the "
                        "native restoration scale (e.g. 1.0 = restore without enlarging, 1.2 = 1.2x): the image is "
                        "restored at the native scale and the finished result is resized with antialiased bicubic "
                        "resampling to round(input size x output_scale). Restoration cost and creativity behaviour "
                        "are those of the native scale. Values above 2 require the 4x native scale. In "
                        "direct_fractional mode it instead sets the size the model restores at directly (1.0-4.0)."
                    ),
                ),
                io.Combo.Input(
                    "restoration_mode",
                    options=list(RESTORATION_MODES),
                    default="calibrated",
                    optional=True,
                    tooltip=(
                        "calibrated: restore at the native 2x/4x scale the model was calibrated for, then resize to "
                        "output_scale (full native cost). direct_fractional (experimental, uncalibrated): restore "
                        "directly at output_scale, so the number of model evaluations follows the requested size "
                        "(e.g. 49 instead of 144 tiles for 4576 px at 1.2x); the native restoration scale only "
                        "selects which calibrated timesteps are reused."
                    ),
                ),
                io.Float.Input(
                    "correction_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Multiplier of the model's correction: z - strength * t * v. The model is still conditioned "
                        "on the creativity timestep t; only the size of the applied update changes. 1.0 = the "
                        "official full correction. 0 skips the transformer (VAE round trip only; with residual "
                        "reconstruction the bicubic input is returned without any model work). Lower values are "
                        "mainly useful with direct_fractional restoration."
                    ),
                ),
                io.Combo.Input(
                    "reconstruction",
                    options=list(RECONSTRUCTIONS),
                    default="latent",
                    optional=True,
                    tooltip=(
                        "latent: decode the corrected latent (official). residual: add only the decoded effect of "
                        "the correction to the bicubic input, decode(z_sr) - decode(z_lq), which cancels the VAE's "
                        "own reconstruction loss (most useful for clean, uncompressed inputs); costs one extra VAE "
                        "decode per tile."
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
        vae=None,
        output_scale: float = 0.0,
        restoration_mode: str = "calibrated",
        correction_strength: float = 1.0,
        reconstruction: str = "latent",
    ) -> io.NodeOutput:
        if not isinstance(sr_model, KandinskySRModel):
            origin = type(sr_model).__module__
            raise TypeError(
                f"sr_model must come from the 'Kandinsky ImageSR Loader' node of this package; received "
                f"{type(sr_model).__name__} from {origin!r}, which is another node pack's model object and cannot "
                "be used here."
            )
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
            vae=vae,
            output_scale=output_scale,
            restoration_mode=restoration_mode,
            correction_strength=correction_strength,
            reconstruction=reconstruction,
        )
        return io.NodeOutput(output)


class KandinskyImageSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [KandinskyImageSRLoader, KandinskyImageSRUpscale]


async def comfy_entrypoint() -> KandinskyImageSRExtension:
    return KandinskyImageSRExtension()
