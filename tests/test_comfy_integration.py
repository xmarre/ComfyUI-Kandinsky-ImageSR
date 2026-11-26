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
    import folder_paths

    import nodes

    models = tmp_path_factory.mktemp("models")
    write_bundle(models / "Kandinsky-ImageSR-2.0", small_transformer_config(latent=16), published_vae_config(), seed=0)
    write_bundle(
        models / "variants" / "ImageSR-alt", small_transformer_config(latent=16), published_vae_config(), seed=5
    )
    folder_paths.add_model_folder_path("diffusion_models", str(models))
    assert asyncio.run(nodes.load_custom_node(str(ROOT)))
    return {
        "loader": nodes.NODE_CLASS_MAPPINGS["KandinskyImageSRLoader"],
        "upscale": nodes.NODE_CLASS_MAPPINGS["KandinskyImageSRUpscale"],
        "models": models,
    }


def _options(spec):
    """Combo options in either the legacy ``[options]`` or the ``("COMBO", {"options"})`` format."""
    return spec[1]["options"] if spec[0] == "COMBO" else spec[0]


def _image(batch=1, h=40, w=56, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch, h, w, 3, generator=gen)


def _load(env, name="Kandinsky-ImageSR-2.0", weight_dtype="bf16"):
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
    assert _options(required["model_name"]) == ["Kandinsky-ImageSR-2.0", "variants/ImageSR-alt"]
    assert _options(required["weight_dtype"]) == ["default", "bf16", "fp32"]
    assert loader["output"] == ["KANDINSKY_SR_MODEL"]
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
    assert req["sr_model"][0] == "KANDINSKY_SR_MODEL" and req["image"][0] == "IMAGE"
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
    import kandinsky_sr.runtime as runtime

    model = _load(env)
    calls = []
    monkeypatch.setattr(runtime, "load_transformer_state_dict", lambda *a, **k: calls.append(1))
    image = _image()
    a = _run(env, model, image, creativity=2)
    b = _run(env, model, image, creativity=6)
    c = _run(env, model, image, creativity=2)
    assert calls == [] and torch.equal(a, c) and not torch.equal(a, b)


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


def test_missing_bundle_error(env):
    with pytest.raises(ValueError, match="models/diffusion_models"):
        env["loader"].execute(
            model_name="(no Kandinsky ImageSR bundle found)", weight_dtype="default", vae_dtype="default"
        )


def test_weight_dtype_choices(env):
    model = _load(env, weight_dtype="fp32")
    assert model.weight_dtype == torch.float32
    assert model.transformer.time_embed_table.dtype == torch.float32
    assert model.transformer.visual_embeddings.in_layer.weight.dtype == torch.float32
    model = _load(env, weight_dtype="bf16")
    assert model.transformer.visual_embeddings.in_layer.weight.dtype == torch.bfloat16
    assert model.transformer.visual_transformer_blocks[0].visual_modulation.out_layer.weight.dtype == torch.float32


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
        widget_counts[cls_name] = sum(
            1
            for spec in info["input"]["required"].values()
            if isinstance(spec[0], list) or spec[0] in ("INT", "FLOAT", "COMBO")
        )
    ours = 0
    for node in workflow["nodes"]:
        if node["type"] in ("Note", "MarkdownNote"):
            continue
        assert node["type"] in nodes.NODE_CLASS_MAPPINGS, node["type"]
        if node["type"] in widget_counts:
            ours += 1
            assert len(node["widgets_values"]) == widget_counts[node["type"]], node
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
                inputs[name] = next(values)
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


def test_basic_workflow_executes_through_prompt_executor(env, io_dirs):
    import execution

    class Server:
        client_id = None
        last_node_id = None

        def send_sync(self, *args, **kwargs):
            pass

        def queue_updated(self):
            pass

    prompt = _api_prompt(json.loads((ROOT / "example_workflows" / "01_Basic_Upscale.json").read_text()))
    valid = asyncio.run(execution.validate_prompt("run", prompt, None))
    assert valid[0], valid[1]
    import inspect

    kwargs = {}
    if "cache_args" in inspect.signature(execution.PromptExecutor.__init__).parameters:
        kwargs["cache_args"] = {"lru": 0, "ram": 0.0, "ram_inactive": 0.0}
    executor = execution.PromptExecutor(Server(), **kwargs)
    executor.execute(prompt, "run", {}, valid[2])
    assert executor.success, executor.status_messages[-3:]
    saved = sorted((io_dirs / "output" / "kandinsky_sr").glob("upscale_*.png"))
    assert saved
    from PIL import Image

    assert Image.open(saved[-1]).size == (60, 48)
