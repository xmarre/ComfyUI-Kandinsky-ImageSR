"""Integration with a real ComfyUI checkout (CPU mode): node discovery, schema,
model management lifecycle and end-to-end execution on a small synthetic bundle."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from pathlib import Path

import pytest
import torch

from tests.helpers import published_vae_config, small_transformer_config, write_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def env(comfy, tmp_path_factory):
    import sys

    import comfy.diffusers_convert
    import comfy.sd
    import comfy.utils
    import folder_paths
    from safetensors.torch import load_file, save_file

    import nodes
    from kandinsky_sr import vae as test_vae_module
    from tests.helpers import write_single_file

    models = tmp_path_factory.mktemp("models")
    extra = tmp_path_factory.mktemp("extra_models")
    bundle = write_bundle(models / "Kandinsky-ImageSR-2.0", small_transformer_config(latent=16), published_vae_config())
    alt = write_bundle(
        models / "variants" / "ImageSR-alt", small_transformer_config(latent=16), published_vae_config(), seed=5
    )
    write_single_file(models / "kandinsky_imagesr_2.0.safetensors", bundle)
    write_single_file(models / "nested" / "deeper" / "ksr_alt.safetensors", alt)
    write_single_file(extra / "extra_ksr.safetensors", bundle)
    save_file({"x": torch.zeros(2)}, str(models / "unrelated_model.safetensors"))
    folder_paths.add_model_folder_path("diffusion_models", str(models))
    folder_paths.add_model_folder_path("diffusion_models", str(extra))

    # A FLUX-format (LDM key) VAE file, loaded exactly like the core Load VAE node does.
    vae_file = tmp_path_factory.mktemp("vae") / "ae.safetensors"
    state = comfy.diffusers_convert.convert_vae_state_dict(load_file(str(bundle / "vae" / VAE_WEIGHTS)))
    save_file({k: v.float().contiguous() for k, v in state.items()}, str(vae_file))

    folder_paths.add_model_folder_path("vae", str(vae_file.parent))  # for the core Load VAE node

    def load_vae(dtype=None):
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae_file)), dtype=dtype)

    assert asyncio.run(nodes.load_custom_node(str(ROOT)))
    upscale = nodes.NODE_CLASS_MAPPINGS["KandinskyImageSRUpscale"]
    prefix = upscale.__module__.rsplit(".", 1)[0]  # ComfyUI imports the package under its path

    def module(name):
        return sys.modules[f"{prefix}.kandinsky_sr.{name}"]

    # The synthetic VAE stands in for FLUX.1: make it the identity reference for both
    # the node's copy of the package and the tests' import of it.
    reference = {"source": "synthetic"}
    sd = load_vae(torch.float32).first_stage_model.state_dict()
    for names in test_vae_module.FINGERPRINT_TENSORS.values():
        for n in names:
            reference[n] = test_vae_module.tensor_fingerprint(sd[n])
    reference["sha256_bf16"] = {part: test_vae_module.bf16_digest(sd, f"{part}.") for part in ("encoder", "decoder")}
    originals = []
    for mod in (module("vae"), test_vae_module):
        originals.append((mod, mod.load_reference_fingerprint))
        mod.load_reference_fingerprint = lambda reference=reference: reference
    yield {
        "loader": nodes.NODE_CLASS_MAPPINGS["KandinskyImageSRLoader"],
        "upscale": upscale,
        "models": models,
        "extra": extra,
        "module": module,
        "load_vae": load_vae,
        "vae_file": vae_file,
    }
    for mod, original in originals:
        mod.load_reference_fingerprint = original


VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"


def _options(spec):
    """Combo options in either the legacy ``[options]`` or the ``("COMBO", {"options"})`` format."""
    return spec[1]["options"] if spec[0] == "COMBO" else spec[0]


def _image(batch=1, h=40, w=56, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch, h, w, 3, generator=gen)


def _load(env, name="Kandinsky-ImageSR-2.0", weight_dtype="bf16"):
    """Load through the loader node; the default name is the official bundle layout."""
    return env["loader"].execute(model_name=name, weight_dtype=weight_dtype, vae_dtype="fp32").result[0]


def _run(env, model, image, **kw):
    params = dict(upscale_factor="2x", creativity=4, overlap=0.25, tile_batch_size=0, accumulation_device="auto")
    params.update(kw)
    return env["upscale"].execute(sr_model=model, image=image, **params).result[0]


def test_node_registration_and_schema(env):
    import nodes

    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["KandinskyImageSRLoader"] == "Kandinsky ImageSR Loader"
    loader = env["loader"].GET_NODE_INFO_V1()
    required = loader["input"]["required"]
    # single-file checkpoints (root, nested folder, extra model path) first, then bundles;
    # the unrelated safetensors file is not listed
    assert _options(required["model_name"]) == [
        "extra_ksr.safetensors",
        "kandinsky_imagesr_2.0.safetensors",
        "nested/deeper/ksr_alt.safetensors",
        "Kandinsky-ImageSR-2.0",
        "variants/ImageSR-alt",
    ]
    assert _options(required["weight_dtype"]) == ["default", "bf16", "fp32"]
    assert loader["output"] == ["KANDINSKY_IMAGESR_MODEL"]
    up = env["upscale"].GET_NODE_INFO_V1()
    req = up["input"]["required"]
    assert list(req) == [
        "sr_model",
        "image",
        "upscale_factor",
        "creativity",
        "overlap",
        "tile_batch_size",
        "accumulation_device",
    ]
    # accepts the legacy type name so saved links stay valid
    assert req["sr_model"][0] == "KANDINSKY_IMAGESR_MODEL,KANDINSKY_SR_MODEL" and req["image"][0] == "IMAGE"
    optional = up["input"]["optional"]
    assert list(optional) == ["vae", "output_scale"] and optional["vae"][0] == "VAE"
    # new widget: optional, appended after the existing widgets, default = native size
    assert optional["output_scale"][0] == "FLOAT"
    assert {k: optional["output_scale"][1][k] for k in ("default", "min", "max", "step")} == {
        "default": 0.0,
        "min": 0.0,
        "max": 4.0,
        "step": 0.01,
    }
    assert _options(req["upscale_factor"]) == ["2x", "4x"] and req["upscale_factor"][1]["default"] == "2x"
    assert req["creativity"][1] == {**req["creativity"][1], "default": 4, "min": 0, "max": 8}
    assert req["overlap"][1]["default"] == 0.25
    assert req["tile_batch_size"][1]["default"] == 0
    assert _options(req["accumulation_device"]) == ["auto", "cpu", "gpu"]
    assert up["output"] == ["IMAGE"]
    json.dumps(loader), json.dumps(up)  # serialisable for /object_info


def test_end_to_end_execution(env):
    model = _load(env)
    image = _image(batch=2)
    out = _run(env, model, image)
    assert out.shape == (2, 80, 112, 3) and out.dtype == torch.float32
    assert torch.isfinite(out).all() and float(out.min()) >= 0 and float(out.max()) <= 1
    out4 = _run(env, model, image, upscale_factor="4x", creativity=8)
    assert out4.shape == (2, 160, 224, 3)
    # deterministic for a fixed batch size
    single = _run(env, model, image, tile_batch_size=1)
    assert torch.equal(single, _run(env, model, image, tile_batch_size=1))
    # batching only changes BF16 GEMM accumulation order (kernel blocking depends on
    # the batch dimension), so results agree to BF16 rounding, not bitwise
    batched = _run(env, model, image, tile_batch_size=3)
    assert (batched - single).abs().mean() < 2e-3 and (batched - single).abs().max() < 5e-2


def test_rgba_alpha_is_resized_not_generated(env):
    model = _load(env)
    rgba = torch.cat([_image(), torch.linspace(0, 1, 56).view(1, 1, 56, 1).expand(1, 40, 56, 1)], dim=-1)
    out = _run(env, model, rgba)
    assert out.shape == (1, 80, 112, 4)
    rgb = _run(env, model, rgba[..., :3].contiguous())
    assert torch.equal(out[..., :3], rgb)
    assert torch.allclose(out[0, 0, :, 3], out[0, -1, :, 3]) and float(out[..., 3].max()) <= 1


def test_repeated_execution_does_not_reload(env, monkeypatch):
    runtime = env["module"]("runtime")  # the node's own copy of the package
    real = runtime.load_transformer_state_dict
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "load_transformer_state_dict", counting)
    model = _load(env)
    assert calls == [1]  # the patch is effective
    image = _image()
    a = _run(env, model, image, creativity=2)
    b = _run(env, model, image, creativity=6)
    c = _run(env, model, image, creativity=2)
    assert calls == [1] and torch.equal(a, c) and not torch.equal(a, b)


def test_model_switching_and_unloading(env):
    import comfy.model_management as mm

    a = _load(env)
    b = _load(env, "variants/ImageSR-alt")
    image = _image()
    out_a, out_b = _run(env, a, image), _run(env, b, image)
    assert not torch.equal(out_a, out_b)
    mm.unload_all_models()
    assert torch.equal(_run(env, a, image), out_a)
    ref = weakref.ref(b.transformer)
    del b
    gc.collect()
    mm.cleanup_models_gc() if hasattr(mm, "cleanup_models_gc") else None
    gc.collect()
    assert ref() is None, "transformer kept alive by a global reference"


def test_input_validation(env):
    model = _load(env)
    bad = _image()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        _run(env, model, bad)
    with pytest.raises(ValueError, match="channels"):
        _run(env, model, torch.rand(1, 8, 8, 2))
    with pytest.raises(ValueError, match="overlap"):
        _run(env, model, _image(), overlap=0.9)
    with pytest.raises(ValueError, match="creativity"):
        _run(env, model, _image(), creativity=9)
    with pytest.raises(ValueError, match="upscale_factor"):
        _run(env, model, _image(), upscale_factor="3x")
    with pytest.raises(ValueError, match="GPU compute device"):
        _run(env, model, _image(), accumulation_device="gpu")
    with pytest.raises(TypeError, match="Loader"):
        _run(env, object(), _image())


def test_foreign_model_objects_get_an_explicit_error(env):
    class KandinskySRModel:  # same class name as another node pack's object
        transformer = patcher = None

    with pytest.raises(TypeError, match="another node pack"):
        _run(env, KandinskySRModel(), _image())


def test_missing_checkpoint_error(env):
    with pytest.raises(ValueError, match="models/diffusion_models"):
        env["loader"].execute(
            model_name="(no Kandinsky ImageSR checkpoint found)", weight_dtype="default", vae_dtype="default"
        )


def test_weight_dtype_choices(env):
    model = _load(env, weight_dtype="fp32")
    assert model.weight_dtype == torch.float32
    assert model.transformer.time_embed_table.dtype == torch.float32
    assert model.transformer.visual_embeddings.in_layer.weight.dtype == torch.float32
    block = model.transformer.visual_transformer_blocks[0]
    assert block.visual_modulation.out_layer.weight.dtype == torch.float32
    model = _load(env, weight_dtype="bf16")
    block = model.transformer.visual_transformer_blocks[0]
    assert model.transformer.visual_embeddings.in_layer.weight.dtype == torch.bfloat16
    # modulation projections are stored as the BF16 values autocast would use; tables stay FP32
    assert block.visual_modulation.out_layer.weight.dtype == torch.bfloat16
    assert block.xattn_bias_table.dtype == torch.float32 and model.transformer.time_embed_table.dtype == torch.float32


def test_interrupt_is_honoured(env):
    import comfy.model_management as mm

    model = _load(env)
    mm.interrupt_current_processing(True)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            _run(env, model, _image())
    finally:
        mm.interrupt_current_processing(False)


@pytest.mark.parametrize("path", sorted((ROOT / "example_workflows").glob("*.json")), ids=lambda p: p.name)
def test_example_workflows_reference_registered_nodes(env, path):
    import nodes

    workflow = json.loads(path.read_text())
    widget_counts = {}
    for cls_name in ("KandinskyImageSRLoader", "KandinskyImageSRUpscale"):
        info = nodes.NODE_CLASS_MAPPINGS[cls_name].GET_NODE_INFO_V1()

        def count(section, info=info):
            return sum(
                1
                for spec in info["input"].get(section, {}).values()
                if isinstance(spec[0], list) or spec[0] in ("INT", "FLOAT", "COMBO")
            )

        # workflows saved before optional widgets existed carry only the required ones
        widget_counts[cls_name] = {count("required"), count("required") + count("optional")}
    ours = 0
    for node in workflow["nodes"]:
        if node["type"] in ("Note", "MarkdownNote"):
            continue
        assert node["type"] in nodes.NODE_CLASS_MAPPINGS, node["type"]
        if node["type"] in widget_counts:
            ours += 1
            assert len(node["widgets_values"]) in widget_counts[node["type"]], node
    assert ours >= 2
    ids = {n["id"] for n in workflow["nodes"]}
    for link in workflow["links"]:
        assert link[1] in ids and link[3] in ids


def _api_prompt(workflow):
    """Convert a UI workflow to an API prompt the way the ComfyUI frontend does."""
    import nodes

    links = {link[0]: link for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        if node["type"] in ("Note", "MarkdownNote"):
            continue
        cls = nodes.NODE_CLASS_MAPPINGS[node["type"]]
        spec = cls.INPUT_TYPES()
        ordered = list(spec.get("required", {}).items()) + list(spec.get("optional", {}).items())
        linked = {i["name"]: links[i["link"]] for i in node["inputs"] if i.get("link") is not None}
        values = iter(node["widgets_values"])
        inputs = {}
        for name, input_spec in ordered:
            if name in linked:
                link = linked[name]
                inputs[name] = [str(link[1]), link[2]]
            elif isinstance(input_spec[0], list) or input_spec[0] in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                # widgets added after a workflow was saved keep their default, as in the frontend
                default = input_spec[1].get("default") if len(input_spec) > 1 else None
                value = next(values, default)
                if value is None:
                    raise AssertionError(f"workflow lacks a value for widget {name!r} of {node['type']}")
                inputs[name] = value
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt


@pytest.fixture(scope="module")
def io_dirs(env, tmp_path_factory):
    import folder_paths
    from PIL import Image

    base = tmp_path_factory.mktemp("io")
    (base / "input").mkdir()
    (base / "output").mkdir()
    gen = torch.Generator().manual_seed(0)
    pixels = (torch.rand(24, 30, 3, generator=gen) * 255).to(torch.uint8).numpy()
    Image.fromarray(pixels).save(base / "input" / "example.png")
    folder_paths.set_input_directory(str(base / "input"))
    folder_paths.set_output_directory(str(base / "output"))
    return base


@pytest.mark.parametrize("path", sorted((ROOT / "example_workflows").glob("*.json")), ids=lambda p: p.name)
def test_example_workflows_pass_comfyui_validation(env, io_dirs, path):
    import execution

    prompt = _api_prompt(json.loads(path.read_text()))
    result = asyncio.run(execution.validate_prompt("test", prompt, None))
    assert result[0], (result[1], result[3])


def _execute_workflow(path):
    import inspect

    import execution

    class Server:
        client_id = None
        last_node_id = None

        def send_sync(self, *args, **kwargs):
            pass

        def queue_updated(self):
            pass

    prompt = _api_prompt(json.loads(path.read_text()))
    valid = asyncio.run(execution.validate_prompt("run", prompt, None))
    assert valid[0], valid[1]
    kwargs = {}
    if "cache_args" in inspect.signature(execution.PromptExecutor.__init__).parameters:
        kwargs["cache_args"] = {"lru": 0, "ram": 0.0, "ram_inactive": 0.0}
    executor = execution.PromptExecutor(Server(), **kwargs)
    executor.execute(prompt, "run", {}, valid[2])
    assert executor.success, executor.status_messages[-3:]


def test_basic_workflow_executes_through_prompt_executor(env, io_dirs):
    """Single-file checkpoint + core Load VAE, the primary workflow."""
    from PIL import Image

    _execute_workflow(ROOT / "example_workflows" / "01_Basic_Upscale.json")
    saved = sorted((io_dirs / "output" / "kandinsky_sr").glob("upscale_*.png"))
    assert saved and Image.open(saved[-1]).size == (60, 48)


def test_fractional_workflow_executes(env, io_dirs):
    """06: 2x restoration, 1.2x output of the 30x24 example image -> 36x29 (28.8 rounds up)."""
    from PIL import Image

    wf = json.loads((ROOT / "example_workflows" / "06_Fractional_Restoration.json").read_text())
    up = next(n for n in wf["nodes"] if n["type"] == "KandinskyImageSRUpscale")
    assert up["widgets_values"] == ["2x", 1, 0.25, 0, "auto", 1.2]
    _execute_workflow(ROOT / "example_workflows" / "06_Fractional_Restoration.json")
    saved = sorted((io_dirs / "output" / "kandinsky_sr").glob("restore2x_c1_out1.2x_*.png"))
    assert saved and Image.open(saved[-1]).size == (36, 29)


def test_fractional_comparison_workflow_restores_once(env, io_dirs, monkeypatch):
    from PIL import Image

    runtime = env["module"]("runtime")
    runs = []
    real_run = runtime.TiledRestorer.run
    monkeypatch.setattr(runtime.TiledRestorer, "run", lambda self, i, o: runs.append(1) or real_run(self, i, o))
    _execute_workflow(ROOT / "example_workflows" / "07_Fractional_Comparison.json")
    assert runs == [1]
    out = io_dirs / "output" / "kandinsky_sr"
    sizes = {
        prefix: Image.open(sorted(out.glob(f"{prefix}_*.png"))[-1]).size
        for prefix in (
            "cmp_original",
            "cmp_bicubic_out1.2x",
            "cmp_restore2x_c1_out2x",
            "cmp_restore2x_c1_out1.2x_lanczos",
            "cmp_restore2x_c1_out1.0x_lanczos",
        )
    }
    assert sizes == {
        "cmp_original": (30, 24),
        "cmp_bicubic_out1.2x": (36, 29),
        "cmp_restore2x_c1_out2x": (60, 48),
        "cmp_restore2x_c1_out1.2x_lanczos": (36, 29),
        "cmp_restore2x_c1_out1.0x_lanczos": (30, 24),
    }


def test_api_prompt_without_output_scale_validates_and_runs(env, io_dirs):
    """API-format prompts written before output_scale existed omit the input entirely."""
    import execution

    prompt = _api_prompt(json.loads((ROOT / "example_workflows" / "01_Basic_Upscale.json").read_text()))
    for node in prompt.values():
        if node["class_type"] == "KandinskyImageSRUpscale":
            node["inputs"].pop("output_scale", None)
            assert "output_scale" not in node["inputs"]
    valid = asyncio.run(execution.validate_prompt("run", prompt, None))
    assert valid[0], valid[1]


def test_workflows_saved_with_earlier_versions_still_run(env, io_dirs):
    """A bundle workflow saved before the VAE input and the type rename existed."""
    from PIL import Image

    legacy = ROOT / "tests" / "fixtures" / "legacy_bundle_workflow.json"
    assert "KANDINSKY_SR_MODEL" in legacy.read_text() and '"vae"' not in legacy.read_text()
    _execute_workflow(legacy)
    saved = sorted((io_dirs / "output" / "kandinsky_sr").glob("upscale_*.png"))
    assert saved and Image.open(saved[-1]).size == (60, 48)


@torch.no_grad()
def test_comfy_attention_dispatch_matches_sdpa_precision(env):
    """Whatever backend ComfyUI selected must keep FP32 score precision like SDPA."""
    import comfy.ldm.modules.attention as attention

    from kandinsky_sr.model import comfy_attention, sdpa_attention

    gen = torch.Generator().manual_seed(0)
    q, k = ((torch.randn(1, 4, 1024, 128, generator=gen) * 3).to(torch.bfloat16) for _ in range(2))
    v = torch.randn(1, 4, 1024, 128, generator=gen).to(torch.bfloat16)
    expected = sdpa_attention(q, k, v, 4).float()
    actual = comfy_attention(q, k, v, 4).float()
    rel = (actual - expected).norm() / expected.norm()
    assert rel < 6e-3, (attention.optimized_attention.__name__, rel)


def test_explicit_batch_reserves_memory_only_for_runnable_tiles(env, monkeypatch):
    import comfy.model_management as mm

    runtime = env["module"]("runtime")
    model = _load(env)
    requested = []
    real = mm.load_models_gpu

    def spy(models, memory_required=0, **kwargs):
        requested.append(memory_required)
        return real(models, memory_required=memory_required, **kwargs)

    monkeypatch.setattr(mm, "load_models_gpu", spy)
    per_tile = runtime.estimate_tile_memory(model, model.bundled_vae()).per_tile
    _run(env, model, _image(), tile_batch_size=64)  # a single-tile image
    assert requested == [per_tile]


@pytest.mark.parametrize("output_on_device", [True, False])
def test_device_resident_output_is_part_of_the_memory_budget(env, monkeypatch, output_on_device):
    """With --gpu-only the FP32 output batch shares the GPU with the tile working memory."""
    import comfy.model_management as mm

    runtime = env["module"]("runtime")
    model = _load(env)
    gpu = torch.device("cuda", 0)
    requested = []

    class Planned(Exception):
        pass

    def plan(models, memory_required=0, **kwargs):
        requested.append(memory_required)
        raise Planned  # stop before anything touches the (simulated) device

    monkeypatch.setattr(model.patcher, "load_device", gpu)
    monkeypatch.setattr(mm, "intermediate_device", lambda: gpu if output_on_device else torch.device("cpu"))
    monkeypatch.setattr(mm, "load_models_gpu", plan)
    image = _image(batch=3, h=40, w=56)
    with pytest.raises(Planned):
        runtime.upscale(model, image, 2, 4, tile_batch_size=1, accumulation_device="cpu")
    per_tile = runtime.estimate_tile_memory(model, model.bundled_vae()).per_tile
    output = 3 * 80 * 112 * 3 * 4 if output_on_device else 0
    assert requested == [per_tile + output]


# ----------------------------------------------------------------------------- single-file checkpoints and VAEs


def test_single_file_with_connected_vae_matches_bundle(env):
    """Same weights through both formats, bundled VAE vs. the same VAE from Load VAE: identical output."""
    image = _image(batch=2)
    bundle_out = _run(env, _load(env), image)  # bundled VAE, loaded in fp32 by _load
    single = _load(env, "kandinsky_imagesr_2.0.safetensors")
    assert single.bundle.kind == "single_file" and not single.has_bundled_vae
    single_out = _run(env, single, image, vae=env["load_vae"](torch.float32))
    assert torch.equal(single_out, bundle_out)
    # the bundle with a connected VAE uses the connected one
    assert torch.equal(_run(env, _load(env), image, vae=env["load_vae"](torch.float32)), bundle_out)
    # nested folders and extra model paths load as well
    assert _load(env, "extra_ksr.safetensors").bundle.root == env["extra"] / "extra_ksr.safetensors"
    assert _load(env, "nested/deeper/ksr_alt.safetensors").bundle.kind == "single_file"


def test_single_file_requires_a_vae(env):
    single = _load(env, "kandinsky_imagesr_2.0.safetensors")
    with pytest.raises(ValueError, match="connect the FLUX.1 VAE"):
        _run(env, single, _image())


def test_incompatible_vaes_are_rejected(env):
    import comfy.sd

    vae_module = env["module"]("vae")
    single = _load(env, "kandinsky_imagesr_2.0.safetensors")
    with pytest.raises(vae_module.IncompatibleVAEError, match="not a ComfyUI VAE"):
        _run(env, single, _image(), vae=object())
    # same architecture, different encoder weights (e.g. the SD3 VAE)
    good = env["load_vae"](torch.float32).first_stage_model.state_dict()
    gen = torch.Generator().manual_seed(123)
    other = {k: v + 0.05 * torch.randn(v.shape, generator=gen) * v.std().clamp_min(1e-3) for k, v in good.items()}
    with pytest.raises(vae_module.IncompatibleVAEError, match="encoder weights"):
        _run(env, single, _image(), vae=comfy.sd.VAE(sd=other, dtype=torch.float32))
    # an SD1-style 4-channel VAE
    sd1 = dict(good)
    sd1["decoder.conv_in.weight"] = torch.zeros(512, 4, 3, 3)
    sd1["encoder.conv_out.weight"] = torch.zeros(8, 512, 3, 3)
    sd1["encoder.conv_out.bias"] = torch.zeros(8)
    with pytest.raises(vae_module.IncompatibleVAEError, match="latent channels"):
        _run(env, single, _image(), vae=comfy.sd.VAE(sd=sd1, dtype=torch.float32))


def _vae_with(env, edit):
    import comfy.sd

    state = {k: v.clone() for k, v in env["load_vae"](torch.float32).first_stage_model.state_dict().items()}
    edit(state)
    return comfy.sd.VAE(sd=state, dtype=torch.float32)


def _flip_sign(name, index):
    def edit(state):
        state[name].view(-1)[index] *= -1

    return edit


@pytest.mark.parametrize(
    "edit,part",
    [
        # unsampled element of a fingerprinted tensor: same samples, same sum of squares
        (_flip_sign("encoder.conv_in.weight", 1), "encoder"),
        # tensor outside the sampled fingerprint
        (_flip_sign("encoder.down.1.block.0.conv1.weight", 0), "encoder"),
        (_flip_sign("decoder.up.0.block.0.conv2.weight", 0), "decoder"),
    ],
)
def test_vae_weight_changes_invisible_to_the_fingerprint_are_reported(env, caplog, edit, part):
    import logging

    vae_module = env["module"]("vae")
    vae = _vae_with(env, edit)
    reference = vae_module.load_reference_fingerprint()
    state = vae.first_stage_model.state_dict()
    assert all(
        vae_module._fingerprint_matches(vae_module.tensor_fingerprint(state[n]), reference[n])
        for n in vae_module.FINGERPRINT_TENSORS[part]
    )
    with caplog.at_level(logging.WARNING):
        vae_module.validate_imagesr_vae(vae)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and f"{part} " in warnings[0] and "not qualified" in warnings[0]


def test_canonical_vae_validates_silently_in_both_precisions(env, caplog):
    import logging

    vae_module = env["module"]("vae")
    with caplog.at_level(logging.WARNING):
        for dtype in (torch.float32, torch.bfloat16):
            vae = env["load_vae"](dtype)
            vae_module.validate_imagesr_vae(vae)
            vae_module.validate_imagesr_vae(vae)  # cached digest
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_vae_validation_with_inference_tensors(env, caplog):
    """ComfyUI loads and runs nodes under inference mode; its tensors have no version counter."""
    import logging

    vae_module = env["module"]("vae")
    with torch.inference_mode():
        vae = env["load_vae"](torch.float32)
        with caplog.at_level(logging.WARNING):
            vae_module.validate_imagesr_vae(vae)
    vae_module.validate_imagesr_vae(vae)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_vae_digest_cache_follows_in_place_weight_changes(env, caplog):
    import logging

    vae_module = env["module"]("vae")
    vae = env["load_vae"](torch.float32)
    vae_module.validate_imagesr_vae(vae)
    weight = vae.first_stage_model.encoder.down[1].block[0].conv1.weight
    with torch.no_grad():
        weight.view(-1)[0] *= -1
    with caplog.at_level(logging.WARNING):
        vae_module.validate_imagesr_vae(vae)
    assert any("encoder" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_connected_vae_is_not_mutated_and_not_retained(env):
    import comfy.model_management as mm

    vae = env["load_vae"](torch.float32)
    before = {k: v.detach().clone() for k, v in vae.first_stage_model.state_dict().items()}
    attrs = (vae.vae_dtype, vae.device, vae.patcher.load_device, vae.patcher.offload_device, vae.output_device)
    pixels = _image()
    decoded_before = vae.decode(vae.encode(pixels))
    single = _load(env, "kandinsky_imagesr_2.0.safetensors")
    _run(env, single, pixels, vae=vae)
    after = vae.first_stage_model.state_dict()
    assert all(torch.equal(before[k], after[k].to(before[k].device)) for k in before)
    assert (vae.vae_dtype, vae.device, vae.patcher.load_device, vae.patcher.offload_device, vae.output_device) == attrs
    assert torch.equal(vae.decode(vae.encode(pixels)), decoded_before)  # other workflow branches unaffected
    ref = weakref.ref(vae.first_stage_model)
    del vae, after, before
    mm.unload_all_models()
    gc.collect()
    assert ref() is None, "the connected VAE is kept alive by the ImageSR model"
    assert single._bundled_vae is None


def test_dynamic_patcher_is_opt_in_and_guarded(env):
    import importlib.util

    import comfy.model_patcher

    memory_management = None
    if importlib.util.find_spec("comfy.memory_management") is not None:
        import comfy.memory_management as memory_management
    runtime = env["module"]("runtime")
    path = env["models"] / "kandinsky_imagesr_2.0.safetensors"
    model = runtime.load_model(path, "x", weight_dtype="bf16")
    assert type(model.patcher) is comfy.model_patcher.ModelPatcher  # default, also with dynamic VRAM enabled
    if not getattr(memory_management, "aimdo_enabled", False):
        with pytest.raises(RuntimeError, match="dynamic VRAM is not available"):
            runtime.load_model(path, "x", weight_dtype="bf16", patcher="dynamic")
    with pytest.raises(ValueError, match="patcher"):
        runtime.load_model(path, "x", weight_dtype="bf16", patcher="other")


# ----------------------------------------------------------------------------- output scale


def _rgba(batch=2, h=27, w=41, seed=3):
    """RGBA batch with opaque, transparent, gradient and hard-edged alpha."""
    gen = torch.Generator().manual_seed(seed)
    image = torch.rand(batch, h, w, 4, generator=gen)
    image[0, ..., 3] = torch.linspace(0, 1, w)[None, :]  # smooth gradient
    image[0, : h // 3, :, 3] = 0.0  # fully transparent band
    image[0, -2:, :, 3] = 1.0  # fully opaque rows
    if batch > 1:
        image[1, ..., 3] = (torch.arange(w)[None, :] >= w // 2).float()  # sharp boundary
    return image


def _reference_resize(native_rgb, alpha, size):
    """The documented operation applied to the native restoration (and to the original alpha)."""
    import torch.nn.functional as F

    from kandinsky_sr.resize import resize_rgb

    rgb = resize_rgb(native_rgb, size)
    if alpha is None:
        return rgb
    resized = F.interpolate(alpha.movedim(-1, 1).float(), size=size, mode="bilinear", align_corners=False)
    return torch.cat([rgb, resized.clamp(0, 1).movedim(1, -1)], dim=-1)


def _forbid_resize(env, monkeypatch):
    runtime = env["module"]("runtime")

    def fail(*args, **kwargs):
        raise AssertionError("the native path must not resize")

    monkeypatch.setattr(runtime, "resize_rgb", fail)
    monkeypatch.setattr(runtime, "_finish_resized", fail)


@pytest.mark.parametrize("factor,native_scale", [("2x", 2.0), ("4x", 4.0)])
def test_native_output_scale_values_are_bitwise_unchanged_and_never_resize(env, monkeypatch, factor, native_scale):
    model = _load(env)
    for image in (_image(batch=2, h=33, w=47), _rgba()):
        omitted = _run(env, model, image, upscale_factor=factor)
        _forbid_resize(env, monkeypatch)
        for scale in (0.0, native_scale):
            assert torch.equal(_run(env, model, image, upscale_factor=factor, output_scale=scale), omitted)
        monkeypatch.undo()


def test_scale_rounding_to_the_native_size_bypasses_the_resize(env, monkeypatch):
    model = _load(env)
    image = _image(batch=1, h=2, w=3)  # 2 * 1.99 = 3.98 -> 4 and 3 * 1.99 = 5.97 -> 6: the native 2x size
    native = _run(env, model, image)
    _forbid_resize(env, monkeypatch)
    assert torch.equal(_run(env, model, image, output_scale=1.99), native)


@pytest.mark.parametrize(
    "factor,scale",
    [("2x", 1.0), ("2x", 1.2), ("2x", 1.5), ("2x", 1.75), ("4x", 2.25), ("4x", 2.5), ("4x", 3.2), ("4x", 1.0)],
)
def test_fractional_output_is_the_native_restoration_resized(env, factor, scale):
    from decimal import Decimal

    import comfy.model_management as mm

    from kandinsky_sr.resize import target_size

    model = _load(env)
    for image in (_image(batch=2, h=33, w=47), _rgba(batch=2, h=27, w=41)):
        native = _run(env, model, image, upscale_factor=factor).float()
        out = _run(env, model, image, upscale_factor=factor, output_scale=scale)
        h, w = image.shape[1:3]
        size = target_size(h, w, Decimal(str(scale)), int(factor[0]))
        alpha = image[..., 3:] if image.shape[-1] == 4 else None
        out_dtype = mm.intermediate_dtype() if hasattr(mm, "intermediate_dtype") else torch.float32
        expected = _reference_resize(native[..., :3], alpha, size).to(out_dtype)
        assert out.shape == (image.shape[0], *size, image.shape[-1])
        assert out.dtype == expected.dtype and out.device == native.device
        assert torch.equal(out, expected)
        assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_alpha_is_resized_from_the_original_not_from_the_native_resolution(env):
    import torch.nn.functional as F

    model = _load(env)
    image = _rgba(batch=2, h=27, w=41)
    out = _run(env, model, image, output_scale=1.5)
    direct = F.interpolate(image[..., 3:].movedim(-1, 1), size=out.shape[1:3], mode="bilinear", align_corners=False)
    assert torch.equal(out[..., 3:], direct.clamp(0, 1).movedim(1, -1))
    # fully transparent and fully opaque regions stay exactly so; a hard edge stays within [0, 1]
    assert (out[0, :6, :, 3] == 0).all()
    torch.testing.assert_close(out[0, -2:, :, 3], torch.ones_like(out[0, -2:, :, 3]), rtol=0, atol=1e-6)
    assert out[1, ..., 3].min() == 0 and out[1, ..., 3].max() == 1


@pytest.mark.parametrize("h,w", [(1, 1), (1, 9), (8, 1), (5, 7), (3, 64)])
def test_tiny_and_thin_images(env, h, w):
    from decimal import Decimal

    from kandinsky_sr.resize import target_size

    model = _load(env)
    for scale in (1.0, 1.5, 1.75):
        out = _run(env, model, _image(batch=1, h=h, w=w), output_scale=scale)
        assert out.shape[1:3] == target_size(h, w, Decimal(str(scale)), 2) and min(out.shape[1:3]) >= 1


@pytest.mark.parametrize("scale", [-1.0, 0.5, float("nan"), float("inf"), -float("inf"), 2.5, 4.5, "1.2", None])
def test_invalid_output_scale_fails_before_any_model_work(env, monkeypatch, scale):
    import comfy.model_management as mm

    runtime = env["module"]("runtime")
    model = _load(env)

    def no_load(*args, **kwargs):
        raise AssertionError("models were loaded before the output scale was validated")

    monkeypatch.setattr(mm, "load_models_gpu", no_load)
    monkeypatch.setattr(runtime, "resolve_vae", no_load)
    with pytest.raises(ValueError, match="output_scale"):
        _run(env, model, _image(), output_scale=scale)


def test_existing_factor_validation_is_unchanged(env):
    model = _load(env)
    with pytest.raises(ValueError, match="upscale_factor must be one of"):
        _run(env, model, _image(), upscale_factor="3x")
    with pytest.raises(ValueError, match="upscale_factor must be one of"):
        _run(env, model, _image(), upscale_factor="3x", output_scale=1.5)


def test_native_inference_is_identical_with_and_without_output_scale(env, monkeypatch):
    """Same tiles, batches, timestep index; the native restoration runs once per call."""
    runtime = env["module"]("runtime")
    model = _load(env)
    runs = []
    real_run = runtime.TiledRestorer.run

    def spy(self, images, opts):
        output, stats = real_run(self, images, opts)
        runs.append((self.grid, opts.timestep, opts.timestep_index, stats.tiles, list(stats.batch_sizes), output))
        return output, stats

    monkeypatch.setattr(runtime.TiledRestorer, "run", spy)
    image = _image(batch=2, h=150, w=170)
    _run(env, model, image, creativity=1)
    _run(env, model, image, creativity=1, output_scale=1.2)
    assert len(runs) == 2
    (g0, t0, i0, n0, b0, out0), (g1, t1, i1, n1, b1, out1) = runs
    assert (g0, t0, i0, n0, b0) == (g1, t1, i1, n1, b1) and t0 == 0.086 and n0 == 2 * g0.num_tiles
    assert torch.equal(out0, out1)


def test_runtime_signature_is_backward_compatible(env):
    import inspect

    runtime = env["module"]("runtime")
    params = list(inspect.signature(runtime.upscale).parameters)
    assert params == [
        "model",
        "image",
        "upscale_factor",
        "creativity",
        "overlap",
        "tile_batch_size",
        "accumulation_device",
        "vae",
        "output_scale",
    ]
    assert inspect.signature(runtime.upscale).parameters["output_scale"].default == 0.0
    model = _load(env)
    image = _image()
    positional, stats = runtime.upscale(model, image, 2, 4, 0.25, 0, "auto", None)
    assert positional.shape[1:3] == (80, 112) and stats.output_size == (80, 112)
    _, stats = runtime.upscale(model, image, 2, 4, 0.25, 0, "auto", None, 1.2)
    assert stats.output_size == (48, 67)  # 40 * 1.2 = 48, 56 * 1.2 = 67.2


def test_repeated_runs_with_different_scales_keep_no_state(env):
    import comfy.model_management as mm

    runtime = env["module"]("runtime")
    model = _load(env)
    image = _image(batch=1, h=31, w=45)
    first = _run(env, model, image)
    seen = []
    for scale in (1.2, 1.5, 0.0, 1.0, 2.0, 1.2):
        out = _run(env, model, image, output_scale=scale)
        seen.append(tuple(out.shape[1:3]))
        if scale in (0.0, 2.0):
            assert torch.equal(out, first)
    assert seen == [(37, 54), (47, 68), (62, 90), (31, 45), (62, 90), (37, 54)]
    assert not any(isinstance(v, torch.Tensor) and v.ndim == 4 for v in vars(model).values())
    del mm, runtime


def test_native_resolution_output_is_released_after_resizing(env, monkeypatch):
    runtime = env["module"]("runtime")
    model = _load(env)
    refs = []
    real_run = runtime.TiledRestorer.run

    def spy(self, images, opts):
        output, stats = real_run(self, images, opts)
        refs.append(weakref.ref(output))
        return output, stats

    monkeypatch.setattr(runtime.TiledRestorer, "run", spy)
    out = _run(env, model, _image(batch=2, h=60, w=70), output_scale=1.3)
    gc.collect()
    assert refs and refs[0]() is None and out.shape[1:3] == (78, 91)


def test_cancellation_before_the_final_resize(env, monkeypatch):
    import comfy.model_management as mm

    runtime = env["module"]("runtime")
    model = _load(env)
    real_run = runtime.TiledRestorer.run
    resized = []

    def run_then_cancel(self, images, opts):
        result = real_run(self, images, opts)
        mm.interrupt_current_processing(True)
        return result

    monkeypatch.setattr(runtime.TiledRestorer, "run", run_then_cancel)
    monkeypatch.setattr(runtime, "resize_rgb", lambda *a: resized.append(1))
    try:
        with pytest.raises(mm.InterruptProcessingException):
            _run(env, model, _image(), output_scale=1.5)
    finally:
        mm.interrupt_current_processing(False)
    assert not resized


class _OnDevice:
    """Stands in for a restored output on an accelerator: ``.cpu()`` yields the real tensor."""

    device = torch.device("cuda", 0)

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape

    def cpu(self):
        return self.tensor


def test_assembly_oom_on_the_device_finishes_in_system_memory_without_rerunning(env):
    runtime = env["module"]("runtime")
    native = torch.rand(2, 10, 12, 3)
    calls, freed = [], []

    def assemble(output):
        calls.append(output.device.type)
        if output.device.type != "cpu":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (resize)")
        return output * 2

    out = runtime.assemble_output(_OnDevice(native), assemble, (10, 12), on_oom=lambda: freed.append(1))
    assert calls == ["cuda", "cpu"] and freed == [1] and torch.equal(out, native * 2)


def test_assembly_errors_other_than_oom_are_not_retried(env):
    runtime = env["module"]("runtime")
    calls = []

    def assemble(output):
        calls.append(1)
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        runtime.assemble_output(_OnDevice(torch.zeros(1, 2, 2, 3)), assemble, (2, 2))
    assert calls == [1]


@pytest.mark.parametrize("on_device", [True, False])
def test_assembly_oom_in_system_memory_raises_an_actionable_error(env, on_device):
    runtime = env["module"]("runtime")
    native = torch.zeros(3, 4, 5, 3)

    def assemble(output):
        raise torch.cuda.OutOfMemoryError("out of memory")

    with pytest.raises(runtime.OutputAssemblyError, match=r"assemble the 7x6 output of 3 image\(s\)"):
        runtime.assemble_output(_OnDevice(native) if on_device else native, assemble, (6, 7))


def test_final_resize_out_of_memory_in_the_node_does_not_rerun_the_restoration(env, monkeypatch):
    runtime = env["module"]("runtime")
    model = _load(env)
    runs = []
    real_run = runtime.TiledRestorer.run
    monkeypatch.setattr(runtime.TiledRestorer, "run", lambda self, i, o: runs.append(1) or real_run(self, i, o))

    def oom(*args):
        raise torch.cuda.OutOfMemoryError("out of memory (resize)")

    monkeypatch.setattr(runtime, "resize_rgb", oom)
    with pytest.raises(runtime.OutputAssemblyError):
        _run(env, model, _image(), output_scale=1.5)  # CPU output: no further fallback exists
    assert runs == [1]


@pytest.mark.parametrize("scale,expect_resize", [(0.0, False), (1.2, True)])
def test_final_resize_memory_is_budgeted_for_device_resident_outputs(env, monkeypatch, scale, expect_resize):
    import comfy.model_management as mm

    runtime = env["module"]("runtime")
    model = _load(env)
    gpu = torch.device("cuda", 0)
    requested = []

    class Planned(Exception):
        pass

    def plan(models, memory_required=0, **kwargs):
        requested.append(memory_required)
        raise Planned

    monkeypatch.setattr(model.patcher, "load_device", gpu)
    monkeypatch.setattr(mm, "intermediate_device", lambda: gpu)
    monkeypatch.setattr(mm, "load_models_gpu", plan)
    image = _rgba(batch=3, h=200, w=300)
    with pytest.raises(Planned):
        runtime.upscale(model, image, 2, 4, tile_batch_size=1, accumulation_device="cpu", output_scale=scale)
    grid = runtime.make_grid(200, 300, 2, model.config.tile_size, 0.25)
    per_tile = runtime.estimate_tile_memory(model, model.bundled_vae()).per_tile
    output = runtime.output_buffer_bytes(3, grid)
    resize = runtime.resize_working_bytes(3, grid, (240, 360), 4) if expect_resize else 0
    assert resize == 0 or resize == 3 * 240 * 360 * 4 * 4 + 12 * (400 * 600 + 400 * 360 + 240 * 360)
    assert requested == [max(per_tile, resize) + output]
