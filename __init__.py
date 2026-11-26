"""Native ComfyUI support for Kandinsky ImageSR 2.0."""

# ComfyUI imports this directory as a package. Test runners may import this file
# without a parent package; the node entry point is only meaningful in the former.
if __package__:
    from .nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
