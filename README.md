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
- Native 2x and 4x restoration with all nine calibrated creativity levels per
  factor, arbitrary input sizes, image batches, RGBA input.
- Optional final output scale between 1x and the native factor (for example
  restore at 2x and return 1.2x), produced by resizing the finished
  restoration.
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
versions with a specific error message. The VAE latent scale in the checkpoint
metadata must be the FLUX.1 value 0.3611.

A connected VAE must have the FLUX.1 architecture, 8x compression and 16 latent
channels, and run in bf16 or fp32. Its weights are identified in two steps:

- a sampled fingerprint of three encoder tensors must match FLUX.1. This
  rejects architecturally identical VAEs with a different latent space, such as
  the SD3 VAE;
- a SHA-256 digest of the complete encoder and of the complete decoder, with
  values rounded to bf16 (so the fp32 `ae.safetensors` and its bf16 copies hash
  identically), is compared with FLUX.1. A VAE that passes the fingerprint but
  differs here (modified weights, a copy stored in a lower precision, or a
  fine-tuned decoder) is used with a warning in the log that the results are not
  qualified.

The digests are computed once per loaded VAE (about 0.1–0.3 s on CPU) and
reused until its weights change. The connected VAE is only read, never
modified, so other branches of the workflow that use it are unaffected.

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
| `upscale_factor` (shown as *native restoration scale*) | `2x` | `2x`, `4x` | Scale at which the model restores; sets tile geometry, cost and the meaning of creativity. Without `output_scale` the output is exactly `width × factor` by `height × factor`. |
| `creativity` | 4 | 0–8 | Calibrated correction strength (see below). 4 is the official default. |
| `overlap` | 0.25 | 0.05–0.75 | Fraction of each input tile shared with its neighbour. 0.25 is the official default. |
| `tile_batch_size` | 0 | 0–64 | Tiles processed together; 0 = automatic. |
| `accumulation_device` | `auto` | `auto`, `cpu`, `gpu` | Where the FP32 blend buffer lives. |
| `vae` (optional) | – | | FLUX.1 VAE from Load VAE. Required for the single-file checkpoint; replaces a bundle's own VAE. |
| `output_scale` (optional) | 0 | 0, or 1.0 to the native factor (max 4.0) | Final size relative to the input; 0 = native size. See [Output scale](#output-scale). |

Output: `IMAGE`.

## Output scale

The model restores only at its two calibrated scales, 2x and 4x. `output_scale`
sets the final size independently:

- `0` (default): the native restoration size, exactly as without this input.
  Workflows saved before this input existed behave identically.
- `1.0` up to the native factor: the image is restored at the native scale, then
  the finished, blended image is resized to
  `round_half_up(input width × output_scale)` by
  `round_half_up(input height × output_scale)` with antialiased bicubic
  resampling (FP32, clamped to [0, 1]). `1.0` restores without changing the
  size; `1.2` turns 1000×1000 into 1200×1200.
- Values above the native factor are rejected (for example 2.5 with
  `2x`: select the `4x` native scale); values between 0 and 1, negative and
  non-finite values are rejected as well. The range is checked on the value
  itself; only floating point noise of up to 1e-9 at an endpoint is accepted
  (`2.0000000000000004` counts as 2, `2.0000001` is rejected).

Dimensions are computed from the original input size, with the scale taken as
a decimal rounded to six places (so widget float noise such as
`1.2000000000000002` cannot change a dimension), and halves rounded up. When
the result equals the native size (for example `output_scale` equal to the
native factor, or a scale that rounds to it for a tiny image), no resize is
performed. An alpha channel is resized directly from the original to the final
size (bilinear, as on the native path); it never passes through the native
resolution.

Cost and behaviour are those of the native scale: restoring at 2x and returning
1.2x costs exactly as much as a plain 2x restoration, and 4x restoration
returned at 2.5x costs a full 4x restoration. Creativity keeps its native-scale
meaning. Resizing down does not by itself improve restoration quality; it lets
the final resolution be chosen independently of the restoration strength. Low
creativity (0–2) at 2x followed by a modest output scale can be a good starting
point for photographic restoration where identity and natural texture matter,
but the best setting depends on the input.

The resize runs on the device that holds the restored image (ComfyUI's
intermediate device), one image at a time; alpha is processed on the same
device. If assembling the output runs out of GPU memory, the restored image is
moved to system memory and the assembly (alpha, resize, dtype conversion) is
repeated there without repeating the restoration; if system memory is
insufficient as well, the node reports how large the output was. On failure or
cancellation the restored image is released rather than kept alive by the
error.

Direct restoration at a non-native scale (feeding the model tiles enlarged by,
say, 1.2x instead of 2x) is not part of the node: the released model has no
calibration for it. A research prototype and its findings are described under
[Direct fractional restoration (research)](#direct-fractional-restoration-research).

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
- **Output.** The FP32 RGB output (12 bytes per output pixel and image) is
  allocated on ComfyUI's intermediate device. When that is the GPU (for example
  with `--gpu-only`) it is included in the memory reserved before the run and
  in the automatic tile batch. If it does not fit, it is kept in system RAM
  instead (with a warning); appending the alpha channel and converting to
  ComfyUI's output dtype fall back to system RAM the same way.
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
hardware. It reports PyTorch allocator peaks and device-wide memory in use
(sampled from the driver, including allocations outside PyTorch's allocator,
such as those of dynamic VRAM) separately; compare standard and dynamic model
patchers with the device-wide figures.

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
- `06_Fractional_Restoration.json` – native 2x restoration at creativity 1,
  returned at 1.2x the input size through `output_scale`.
- `07_Fractional_Comparison.json` – one 2x restoration (creativity 1) saved at
  its native size and reduced to 1.2x and 1.0x with ComfyUI's built-in Lanczos
  resize, next to a plain bicubic 1.2x enlargement and the original. The
  built-in resize avoids repeating the restoration; it is close to, but not
  bit-identical with, what `output_scale` produces.

Workflows 01–04, 06 and 07 use the single-file checkpoint with
`ae.safetensors` from Load VAE.

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
  prompt validator, and execution of workflows through ComfyUI's
  `PromptExecutor` on small synthetic checkpoints, including workflows saved
  before the latest inputs existed;
- output scale: validation, exact target dimensions and rounding, equality with
  the native restoration followed by the documented resize (RGB, RGBA,
  batches, tiny images), bit-identical native results when no resize applies,
  out-of-memory fallback without repeating the restoration, cancellation and
  memory budgeting;
- geometry of the direct fractional research prototype.

`scripts/verify_reference.py` compares this implementation with the official
pipeline on the real checkpoint, stage by stage (encoded latents, transformer
velocity for all 18 creativity presets, decoded tiles, final image). The
reference phase runs in a separate environment with the official package, so it
never touches the ComfyUI environment.

### Measured real-checkpoint parity (CPU)

Measured on CPU (PyTorch 2.15, bf16 weights and VAE) against the official
pipeline with its transformer under bf16 autocast, the mode the official CUDA
path uses, fed a PIL image as in the official command line and demo. ComfyUI was
run with `--use-pytorch-cross-attention` (ComfyUI's default on current CUDA
setups). Input: a 200x300 public-domain photograph (one tile at 2x, four tiles
at 4x).

| Stage | Result |
|---|---|
| Bicubic tiles fed to the VAE | bitwise identical |
| Transformer velocity, identical latents, all 18 creativity presets | **bitwise identical** (bundle and single-file checkpoint) |
| VAE encoder, fp32, ComfyUI vs Diffusers implementation | relative RMS 8.7e-8 |
| Encoded latents, bf16 VAE | relative RMS 1.1–1.3e-3 (different bf16 kernels) |
| Final image, 2x, creativity 0 / 4 / 8 | PSNR 55.3 / 51.7 / 45.1 dB vs. official output |
| Final image, 4x, creativity 0 / 4 / 8, tile batch 1 | PSNR 53.8 / 50.4 / 46.5 dB |
| Final image, 4x, creativity 0 / 4 / 8, tile batch 2 | PSNR 53.7 / 50.6 / 46.5 dB |
| Single-file checkpoint + `ae.safetensors` (bf16) vs. official bundle | identical results in every case |

The remaining end-to-end difference is bf16 rounding in the VAE, amplified by
the transformer: a latent difference of about 1e-3 relative RMS becomes a
velocity difference of 1.6–2.4 % relative RMS in these runs. The bf16 VAE is
intrinsically sensitive (bf16 vs. fp32 encoding of the same tile differs by
about 5 % relative RMS), and the tile batch size changes which bf16 kernels run,
which explains the small differences between tile batch 1 and 2; the official
pipeline is subject to the same mechanism. These figures are CPU measurements;
CUDA parity is checked with the script above.

Attention backend matters: ComfyUI's split/sub-quadratic/basic attention compute
scores in bf16 unless asked to upcast; this package requests FP32 score
precision, which brings those backends to within 0.25 % of SDPA (from 2–3 %).
Backends with their own quantization (e.g. SageAttention) are used as
configured and are not qualified.

## Direct fractional restoration (research)

`output_scale` always restores at a calibrated native scale. A faster
alternative would be to restore directly at the requested scale: enlarge the
input by, say, 1.2x instead of 2x, so each 1024-pixel tile covers more of the
input and fewer transformer evaluations are needed. This is **not** offered by
the node. `scripts/research_direct_fractional.py` is an isolated prototype for
studying it.

**What the model receives.** The transformer sees only the VAE latent of the
bicubic-enlarged tile and one of its baked timesteps. There is no scale or
degradation input; the two calibrated tables (2x: 0.07–0.30, 4x: 0.17–0.38) do
not share any timestep, so the scale is implied only by the choice of timestep
and by how blurred the enlarged input looks. A 1.2x enlargement is much sharper
than a 2x one, so its latent plausibly lies outside what the 2x calibration was
made for. The prototype reuses the 2x creativity timesteps for scales between
1x and 2x; no calibration exists for this.

**Geometry.** Tiles are placed in output pixels with the official rule
(1024-pixel tiles, stride `int(1024 × (1 − overlap))`, last tile aligned to the
edge, boundary-aware Hann windows); every output pixel centre `o` maps to the
input coordinate `(o + 0.5) × in / out − 0.5`, and each tile is sampled from
the whole image (bicubic, border taps clamped), so overlapping tiles share one
global mapping. Tests establish positive blend weight for every pixel,
equality with the native tile placement at 2x, seam-free reconstruction with an
identity model, and bit-identical delegation to the native pipeline at exact
2x/4x.

**Experiment** (CPU, bf16, single-file checkpoint and `ae.safetensors`): a
420×1260 reference (a public-domain portrait, a synthetic text and pattern
chart, a CC0 photograph) was reduced by each scale with antialiased bicubic
resampling and JPEG-compressed (quality 75); each method then produced the
reference size. PSNR (over RGB) / SSIM (on luma) against the reference,
creativity 1 (t = 0.086):

| Scale | Bicubic | Native 2x + resize | Direct | Tiles native / direct |
|---|---|---|---|---|
| 1.0 | 30.92 / 0.963 | 28.41 / 0.951 | 24.32 / 0.880 | 3 / 2 |
| 1.2 | 24.44 / 0.925 | 25.90 / 0.926 | 24.70 / 0.887 | 3 / 2 |
| 1.5 | 22.41 / 0.881 | 24.09 / 0.889 | 23.87 / 0.872 | 2 / 2 |
| 1.75 | 21.30 / 0.850 | 23.05 / 0.867 | 22.89 / 0.855 | 2 / 2 |
| 2.0 | 20.38 / 0.815 | 20.90 / 0.827 | (= native) | 2 / 2 |

At creativity 0, 2 and 4 the ordering is the same at every scale, and the
direct path loses more as creativity rises (at 1.2x: SSIM 0.896 / 0.865 / 0.847
direct versus 0.929 / 0.920 / 0.913 native). Visually, direct restoration
over-sharpens faces and raises contrast, and it invents glyph shapes in small
text ("Restoration" rendered as "Raslorallon", "13px" as "18px") where native
2x + resize stays faithful. At 1.0x neither restoration beats the JPEG input in
PSNR: the restoration changes pixels in ways PSNR penalises. The picture is not
uniformly negative for the direct path: at 1.2x it still exceeds bicubic in
PSNR while falling below it in SSIM, and its deficit shrinks to 0.2 dB at 1.5x
and 1.75x. No step discontinuities were found at the tile window boundaries.

These results compare two complete pipelines on one composite image with one
degradation; they do not isolate a cause. The pipelines differ in the
enlargement fed to the transformer, the source area per tile, tile-local versus
global bicubic sampling, and the antialiased reduction after native
restoration, which by itself suppresses generated high-frequency texture. A
timestep that is too strong for the milder enlargement is one plausible
explanation consistent with the results, not an established one. The exact
2x/4x control cases delegate to the native pipeline, so they do not show how
the direct path behaves as the scale approaches 2x; the identity-model tests
establish geometry and blending only, not equivalence of restoration
behaviour. The script's timings are exploratory (the native arm runs the full
node runtime, the direct arm only the tile loop, without warm-up or repeated
measurements) and are not a speed comparison.

**Cost.** With the implemented placement, a 2048² input needs 9 transformer
evaluations directly at 1.0x or 1.2x, 16 at 1.5x and 25 at 1.75x, against 25 for
native 2x (1024²: 1, 4, 4, 4 against 9). The saving disappears towards 2x
because tile counts are quantised. Per-tile cost is the same, so time scales
with the tile count; this has not been measured on a GPU.

**Conclusion.** On this evidence direct fractional restoration needs fewer
transformer evaluations but was less faithful on the tested image, so the node
keeps the native restoration followed by a resize. A broader benchmark (more
images and degradations) would be needed for a general verdict. The prototype remains available for experiments on other images or
with other timestep choices (`--timestep-table 4` reuses the 4x values):

```bash
python scripts/research_direct_fractional.py tiles --sizes 1024 2048
python scripts/research_direct_fractional.py experiment --comfyui /path/to/ComfyUI \
    --model models/diffusion_models/kandinsky_imagesr_2.0.safetensors --vae models/vae/ae.safetensors \
    --images reference.png --scales 1.0 1.2 1.5 1.75 2.0 --creativity 0 1 2 4 --jpeg 75 --out results/
```

The reference image's dimensions must be reachable from an integer size at
every scale (420 works for 1.0/1.2/1.5/1.75/2.0).

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
- **Warning "encoder/decoder ... not qualified"**: the VAE resembles FLUX.1 but
  its weights are not identical to it. Use the original `ae.safetensors`.
- **"output_scale ... exceeds the native restoration scale"**: the output scale
  is above the selected native factor; select `4x` for output scales above 2.
- **"output_scale must be 0 (native size) or at least 1.0"**: values between 0
  and 1 would shrink the image below its input size; resize afterwards with a
  separate node if that is intended.
- **"not enough system memory to assemble the ... output"**: the restoration
  finished but the final image could not be built; process fewer images at once
  or choose a smaller output scale.
- **Out of memory with a single tile**: free VRAM, use `--lowvram`, or set
  `accumulation_device` to `cpu`.
- **Colours differ from the official command line tool**: ComfyUI's Load Image
  node does not convert embedded ICC colour profiles to sRGB, the official tool
  does. Convert such images to sRGB beforehand.
- **Visible change of faces or text**: lower `creativity`.

## Known limitations

- Only the published 2x/4x tile geometry and the 18 calibrated timesteps are
  supported; arbitrary strengths would require timesteps that are not baked into
  the checkpoint. Other output sizes come from resizing a native restoration
  (`output_scale`) and cost a full native restoration.
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
