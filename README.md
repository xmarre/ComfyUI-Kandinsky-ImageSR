# ComfyUI-Kandinsky-ImageSR

Native ComfyUI nodes for **Kandinsky ImageSR 2.0**, a one-step generative 2x/4x
super-resolution and image restoration model from Kandinsky Lab.

- Loads the single-file checkpoint `kandinsky_imagesr_2.0.safetensors` together
  with the standard FLUX.1 VAE from ComfyUI's Load VAE node, or the unmodified
  official Hugging Face bundle.
- Native ComfyUI implementation of the baked transformer: ComfyUI operations,
  optimized attention dispatch, `ModelPatcher` model management (loading,
  offloading, low-VRAM partial loading). Diffusers/Transformers are not required.
- Reproduces the official inference pipeline: tile geometry, bicubic
  preprocessing, FLUX VAE latent scaling, one-step flow correction and
  boundary-aware Hann blending.
- 2x and 4x, all nine calibrated creativity levels per factor, arbitrary input
  sizes, image batches, RGBA input.
- Automatic tile batching, GPU or CPU blend accumulation, and out-of-memory
  recovery that never duplicates or drops tiles.

## What the model does

Kandinsky ImageSR 2.0 is a Kandinsky 5 diffusion transformer fine-tuned for
restoration. Each 1024x1024 output tile is produced by one transformer
evaluation: the bicubic-upscaled tile is encoded with the FLUX.1 VAE, the
transformer predicts a flow velocity at a fixed timestep `t`, the latent is
corrected as `z = z_lq - t * v`, and the result is decoded. There is no prompt,
no sampler, no noise and no seed: the output is a deterministic function of the
input image and the settings.

The released checkpoint has its text conditioning "baked": the text encoder and
all cross-attention layers are replaced by per-timestep bias tables for 19
discrete timesteps. Only those timesteps can be evaluated, which is why the node
exposes calibrated creativity levels instead of a continuous strength.

## Installation

Install through ComfyUI-Manager or clone into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Kandinsky-ImageSR
```

There are no additional Python dependencies; the nodes use PyTorch and
safetensors from ComfyUI.

**Supported ComfyUI versions:** v0.3.50 or newer. CI runs the full test suite
against v0.3.50, v0.46.3 and the current ComfyUI master.

## Model download and placement

### Recommended: single-file checkpoint + FLUX.1 VAE

| File | Folder | Source |
|---|---|---|
| `kandinsky_imagesr_2.0.safetensors` (10.25 GB) | `ComfyUI/models/diffusion_models/` | [hmdlsk/Kandinsky-ImageSR-2.0-ComfyUI](https://huggingface.co/hmdlsk/Kandinsky-ImageSR-2.0-ComfyUI) |
| `ae.safetensors` (FLUX.1 VAE, 335 MB) | `ComfyUI/models/vae/` | the FLUX.1 release; most FLUX installations already have it |

The single-file checkpoint contains the transformer tensors of the official
release unchanged (same names, shapes, dtypes and bytes) plus the official
configuration files as safetensors metadata (`format =
kandinsky_image_sr_single_file`, `format_version = 1`). It does not contain a
VAE: connect `ae.safetensors` through a **Load VAE** node to the upscale node's
`vae` input. Subfolders and `diffusion_models`/`vae` paths from
`extra_model_paths.yaml` work as usual. Only files carrying this format's
metadata are listed by the loader; to decide that, it reads just the
safetensors header of each candidate.

The official release ships its VAE as `ae.safetensors` rounded to bfloat16.
With the VAE in bf16 (ComfyUI's default on GPUs that support it) the two are
bit-for-bit the same weights. Loading `ae.safetensors` in fp32 (`--fp32-vae`)
runs the VAE at higher precision than the official pipeline, which is not
bit-identical to it; its effect on results has not been evaluated.

To create the single-file checkpoint from an official bundle you already have
(streams the data; uses little memory):

```bash
python scripts/convert_to_single_file.py convert <bundle folder> \
  ComfyUI/models/diffusion_models/kandinsky_imagesr_2.0.safetensors --verify
```

Converting the unmodified official bundle reproduces the published
`kandinsky_imagesr_2.0.safetensors` byte for byte (SHA-256
`56e2a496735be188faf8750a21e5221281e0ddf639ae15bf9947d6942c84fc3c`).
`scripts/convert_to_single_file.py compare <checkpoint> <bundle folder>` checks
any single-file checkpoint against a bundle (metadata and every tensor's bytes).

### Alternative: official Hugging Face bundle

Download the repository (about 10.4 GB) from
<https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0> and place it, with
its folder structure unchanged, in a folder inside `models/diffusion_models`
(or any configured `diffusion_models` path):

```text
ComfyUI/models/diffusion_models/Kandinsky-ImageSR-2.0/
├── model_index.json
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   └── diffusion_pytorch_model-00003-of-00003.safetensors
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

```bash
huggingface-cli download kandinskylab/Kandinsky-ImageSR-2.0 \
  --include "model_index.json" "transformer/*" "vae/*" \
  --local-dir ComfyUI/models/diffusion_models/Kandinsky-ImageSR-2.0
```

Bundle folders are found directly in a `diffusion_models` root or one level
below it. When no VAE is connected, the bundle's own VAE is used.

### Validation

Nothing is downloaded automatically. Before any weights are read, the loader
checks every configuration value it depends on and every tensor name, shape and
storage dtype against the architecture, and rejects missing, duplicated,
unexpected, overlapping, truncated or mis-shaped tensors and unknown format
versions with a specific error message. A connected VAE must have the FLUX.1
architecture, 8x compression, 16 latent channels and the FLUX.1 encoder weights
(architecturally identical VAEs with a different latent space, such as the SD3
VAE, are rejected); it must run in bf16 or fp32. The connected VAE is only read,
never modified, so other branches of the workflow that use it are unaffected.

## Nodes

### Kandinsky ImageSR Loader (`KandinskyImageSRLoader`)

| Input | Default | Description |
|---|---|---|
| `model_name` | – | Single-file checkpoint or bundle folder (see above). |
| `weight_dtype` | `default` | Transformer precision: `default` (bf16 where supported, otherwise fp32), `bf16`, `fp32`. |
| `vae_dtype` | `default` | Precision of a bundle's own VAE (used only when no VAE is connected): `default` (ComfyUI's choice), `bf16`, `fp32`. |

Output: `sr_model` (connection type `KANDINSKY_IMAGESR_MODEL`), the transformer
under ComfyUI model management. Changing upscale settings or inputs reuses the
loaded model; it is reloaded only when the loader's own inputs change. The
connection type is specific to this package so it cannot be wired into
another Kandinsky ImageSR node pack; the upscale input also accepts the earlier
name `KANDINSKY_SR_MODEL` so saved workflows keep working, and it reports a
clear error if it receives another pack's model object.

### Kandinsky ImageSR Upscale (`KandinskyImageSRUpscale`)

| Input | Default | Range | Description |
|---|---|---|---|
| `sr_model` | – | | From the loader. |
| `image` | – | | ComfyUI `IMAGE` (RGB or RGBA, any size, any batch). |
| `upscale_factor` | `2x` | `2x`, `4x` | Output is exactly `width × factor` by `height × factor`. |
| `creativity` | 4 | 0–8 | Calibrated correction strength (see below). 4 is the official default. |
| `overlap` | 0.25 | 0.05–0.75 | Fraction of each input tile shared with its neighbour. 0.25 is the official default. |
| `tile_batch_size` | 0 | 0–64 | Tiles processed together; 0 = automatic. |
| `accumulation_device` | `auto` | `auto`, `cpu`, `gpu` | Where the FP32 blend buffer lives. |
| `vae` (optional) | – | | FLUX.1 VAE from Load VAE. Required for the single-file checkpoint; replaces a bundle's own VAE. |

Output: `IMAGE`.

## Creativity

Creativity selects the timestep `t` of the one-step correction. Higher values
apply a stronger correction:

| Creativity | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| `t` at 2x | 0.070 | 0.086 | 0.118 | 0.134 | **0.150** | 0.180 | 0.210 | 0.270 | 0.300 |
| `t` at 4x | 0.170 | 0.1885 | 0.207 | 0.2255 | **0.244** | 0.278 | 0.312 | 0.346 | 0.380 |

Higher creativity is not "higher quality". A stronger correction reconstructs
more texture and removes more degradation, and it also changes more of what is
already there: facial features, small text, repeated patterns and other fine
structure can be altered or invented. When fidelity to the source matters (for
example when preparing training images of a specific person), start low and
compare against the input; the `02_Creativity_Comparison` workflow renders all
levels side by side from one loaded model.

The checkpoint contains one additional baked timestep (0.1) that the official
calibration does not assign to a creativity level; it is not exposed.

## Tiling

The model works on fixed 1024x1024 output tiles (128x128 latents, 4096 tokens);
the tile size cannot be changed. Input tiles are 512 px at 2x and 256 px at 4x
with a stride of `int(tile × (1 − overlap))` (384 / 192 px at the default
overlap). A last tile is aligned to the far image edge when the stride does not
reach it, so it can overlap its neighbour by more than the nominal amount.
Inputs smaller than one tile are replicate-padded on the bottom/right and the
output is cropped back. Tiles are blended with periodic Hann windows whose
halves facing the image border are flat, so every output pixel receives a
positive weight; the output is the weighted average.

Tile counts at the default overlap, for planning: 1024² input → 9 tiles at 2x,
25 at 4x; 2048² → 25 at 2x, 121 at 4x; 4096² → 121 at 2x.

The overlap range is narrower than the reference's `[0, 1)`: with zero overlap
the reference blend gives interior tile borders zero weight (black lines), and
very large overlaps only multiply the work.

## Memory and performance behaviour

- **Model residency.** The transformer (9.06 GB of weights in bf16 mode, about
  18.1 GB in fp32 mode) and the VAE (0.17 GB in bf16) are loaded through
  ComfyUI's model manager together with an estimate of the working memory
  needed. With `--lowvram`/normal VRAM mode ComfyUI may keep part of the
  transformer in system RAM and stream it (slower); `--highvram` keeps it
  resident.
- **Automatic tile batch** (`tile_batch_size = 0`): after the models are
  loaded, the batch is sized from the free device memory using conservative
  per-tile estimates (ComfyUI's own VAE estimates plus a transformer activation
  estimate), at most 8 tiles. On CPU the batch is 1.
- **Out of memory.** If a batch fails with an out-of-memory error, it is
  discarded and retried at half the size. Tiles are added to the output only
  after every stage of their batch succeeded, so no tile is accumulated twice or
  lost, and blend buffers allocated by the failed attempt are released before
  the retry. If a single tile does not fit, the node stops with an explicit error
  (the tile size is fixed by the model). Errors that are not out-of-memory are
  never retried.
- **Accumulation.** The FP32 RGB blend buffer needs 12 bytes per output pixel
  (about 768 MiB for 8192x8192; the blend weights are separable and kept as two
  1-D vectors). One buffer exists per image that the current tile batch
  touches, and it is released as soon as the image is complete; a batch of `n`
  tiles over images of `t` tiles each keeps at most `1 + ceil((n − 1) / t)`
  buffers alive. `auto` keeps them on the GPU only if all of those fit in a
  quarter of the free VRAM after the tile batch, otherwise in system RAM; `cpu`
  always uses system RAM (also with `--gpu-only`). Normalising and copying to
  the output happens in row chunks without a second full-size buffer.
- **Batches of images** share one tile queue, so small images are batched
  across image boundaries. With an explicit `tile_batch_size`, working memory
  is reserved only for as many tiles as there are.
- Progress is reported per tile and the node checks for cancellation between
  stages.

No performance figures on consumer or workstation GPUs have been measured for
this package yet. For orientation only, the official repository reports median
times on an NVIDIA H100 80 GB (bf16, tile batch 1): 1024²→2048² 2.78 s,
1024²→4096² 7.80 s, 2048²→8192² 37.5 s. `scripts/benchmark.py` measures load
time, warm latency, per-stage time, tile throughput and peak memory on your own
hardware.

## Precision

| Mode | Transformer | Notes |
|---|---|---|
| `bf16` | bf16 weights and activations; baked tables, normalisation, modulation arithmetic, residual updates and the flow correction in FP32 | Reproduces the official CUDA inference (which runs under bf16 autocast). |
| `fp32` | everything FP32 | Higher precision than the reference; about 18.1 GB of weights. For GPUs without bf16 support. |

The checkpoint stores the modulation projections in FP32, but the official
CUDA inference computes them under bf16 autocast, which rounds weight and bias
to bf16 on every call. In bf16 mode these weights are therefore stored as that
same bf16 rounding (1.19 GB less memory, bit-identical results; covered by a
test against the reference computation). fp32 mode keeps them in FP32. The baked
time-embedding and cross-attention tables always stay FP32.

FP16 and FP8/INT8 weight formats are not offered: they have not been qualified
for this model. Storing the attention and feed-forward weights as FP8 (e4m3fn)
and upcasting them for compute changes the transformer output by about 9–10 %
relative RMS on a real tile (42 % of these weights fall below e4m3fn's smallest
normal value), and saves memory without speeding up inference; accelerated FP8
or INT8 execution would need dedicated kernels and a quality evaluation. ComfyUI's global `--fp16-unet`/`--fp8-*-unet` flags are ignored
for this model with a warning.

The VAE follows the reference latent convention: latents are multiplied by the
VAE `scaling_factor` (0.3611) only. The FLUX `shift_factor` that ComfyUI applies
to FLUX image models is **not** applied; the latents never pass through
ComfyUI's FLUX latent format.

## Example workflows

In `example_workflows/` (drag into ComfyUI):

- `01_Basic_Upscale.json` – load, upscale 2x at creativity 4, preview and save.
- `02_Creativity_Comparison.json` – one loader, five upscalers at creativity
  0/2/4/6/8, each saved with its setting in the file name.
- `03_Large_Image_Upscale.json` – 4x with automatic tile batching and
  memory-aware accumulation.
- `04_Qualification_Grid.json` – 2x and 4x at creativity 0/4/8 plus bicubic
  baselines, for judging results on your own images.
- `05_Official_Bundle.json` – the official bundle with its own VAE.

Workflows 01–04 use the single-file checkpoint with `ae.safetensors` from Load
VAE.

## Testing and reference qualification

```bash
pip install pytest
COMFYUI_PATH=/path/to/ComfyUI python -m pytest -q
```

Without `COMFYUI_PATH` the ComfyUI integration tests are skipped (and reported
as skipped). The suite includes:

- bit-exact comparison of the native transformer with the upstream reference
  code (copied into `tests/reference_impl.py`) on reduced-size random models,
  for all baked timesteps;
- tile positions, windows, blend weights and full tiled reconstruction against
  the upstream tiling code for many sizes, aspect ratios and overlaps;
- latent normalisation (scale once, no shift) against a real ComfyUI FLUX VAE;
- strict bundle validation, including a fixture of the published checkpoint's
  tensor layout;
- out-of-memory injection in every stage and during accumulator allocation,
  verifying identical output after retries;
- node registration, schema, example workflow validation with ComfyUI's
  prompt validator, and execution of a workflow through ComfyUI's
  `PromptExecutor` on a small synthetic bundle.

`scripts/verify_reference.py` compares this implementation with the official
pipeline on the real checkpoint, stage by stage (encoded latents, transformer
velocity for all 18 creativity presets, decoded tiles, final image). The
reference phase runs in a separate environment with the official package, so it
never touches the ComfyUI environment.

### Measured real-checkpoint parity (CPU)

Measured on CPU (PyTorch 2.15, bf16 weights and VAE) against the official
pipeline with its transformer under bf16 autocast, the mode the official CUDA
path uses. ComfyUI was run with `--use-pytorch-cross-attention` (ComfyUI's
default on current CUDA setups). Inputs were public-domain/CC0 photographs.

| Stage | Result |
|---|---|
| Bicubic tiles fed to the VAE | bitwise identical |
| Transformer velocity, identical latents, all 18 creativity presets | **bitwise identical** |
| VAE encoder, fp32, ComfyUI vs Diffusers implementation | relative RMS 8.7e-8 |
| VAE encoder, bf16, same batch size | relative RMS ≈ 1e-3 (different bf16 kernels) |
| Final image, tile batch 1 (3 cases: 2x c4, 4x c0, 4x c8) | PSNR 53.6 / 53.5 / 46.5 dB vs. official output |
| Final image, tile batch 2 here vs. 1 in the reference (2x c4) | PSNR 48.9 dB |

The remaining end-to-end difference is bf16 rounding in the VAE, amplified by
the transformer: a 1e-3 relative change of the latents changes the velocity by
1.5–6 %. The bf16 VAE is intrinsically sensitive (bf16 vs. fp32 encoding of the
same tile differs by about 5 % relative RMS). The tile batch size changes which
bf16 kernels run, so it changes results at this level too (latent difference
0.1 % at equal batch size, 2.3 % at batch 2 vs. 1); the official pipeline is
subject to the same mechanism. These figures are CPU measurements; CUDA parity
is checked with the script above.

Attention backend matters: ComfyUI's split/sub-quadratic/basic attention compute
scores in bf16 unless asked to upcast; this package requests FP32 score
precision, which brings those backends to within 0.25 % of SDPA (from 2–3 %).
Backends with their own quantization (e.g. SageAttention) are used as
configured and are not qualified.

## Troubleshooting

- **The loader shows "(no Kandinsky ImageSR checkpoint found)"**: put
  `kandinsky_imagesr_2.0.safetensors` in a `diffusion_models` folder (subfolders
  are fine), or a bundle folder containing `model_index.json` and
  `transformer/config.json` directly inside a `diffusion_models` folder or one
  level below it. Refresh the browser after adding it. A `.safetensors` file
  without the single-file format metadata is not listed.
- **"weights do not match the expected architecture" / "truncated"**: a file is
  incomplete or from a different model; re-download the listed file.
- **"no bundled VAE: connect the FLUX.1 VAE"**: the single-file checkpoint needs a
  Load VAE node with `ae.safetensors` connected to the upscale node.
- **"incompatible VAE"**: the connected VAE is not the FLUX.1 VAE (for example an
  SD1/SDXL, SD3 or FLUX.2 VAE). Select `ae.safetensors` in Load VAE.
- **Out of memory with a single tile**: free VRAM, use `--lowvram`, or set
  `accumulation_device` to `cpu`.
- **Colours differ from the official command line tool**: ComfyUI's Load Image
  node does not convert embedded ICC colour profiles to sRGB, the official tool
  does. Convert such images to sRGB beforehand.
- **Visible change of faces or text**: lower `creativity`.

## Known limitations

- Only the published 2x/4x tile geometry and the 18 calibrated timesteps are
  supported; arbitrary strengths would require timesteps that are not baked into
  the checkpoint.
- Quantized (FP8/INT8) and FP16 execution and `torch.compile` are not
  implemented.
- The transformer is managed by ComfyUI's standard `ModelPatcher`, also when
  ComfyUI's dynamic VRAM is enabled (dynamic VRAM still applies to other models,
  including the VAE). The transformer is built from ComfyUI's casting operations
  and should be compatible with `ModelPatcherDynamic`, but that path has not been
  tested on a GPU; the qualification and benchmark scripts can select it with
  `--patcher dynamic`. LoRA or other weight patches for this model are not
  supported.
- Combined single-file checkpoints with a bundled VAE (`transformer.`/`vae.`
  tensor prefixes) are not supported; use the transformer-only file.
- Visual quality has only been spot-checked on a few public-domain photographs;
  GPU performance has not been measured by this project. Use
  `04_Qualification_Grid.json` and `scripts/benchmark.py` on your own data and
  hardware.

## Attribution and licenses

- This package: GPL-3.0 (see `LICENSE`), consistent with ComfyUI.
- Model architecture and inference algorithm: Kandinsky Lab,
  [kandinsky-image-sr](https://github.com/kandinskylab/kandinsky-image-sr)
  (MIT). The Kandinsky 5 building blocks are ported from Diffusers
  (Apache-2.0). Notices are reproduced in `NOTICE`.
- The single-file checkpoint format and its tensor layout follow
  [ComfyUI-KSR-Unofficial](https://github.com/the-aSak/ComfyUI-KSR-Unofficial)
  (MIT); `scripts/convert_to_single_file.py` reproduces its converter's output.
- Model weights (not distributed here): Kandinsky ImageSR 2.0, MIT License. The
  VAE is the FLUX.1-schnell VAE by Black Forest Labs, Apache-2.0.
