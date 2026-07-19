"""Utilities for opt-in Keras-style torch.nn.Module method patching."""

from __future__ import annotations

from torch import nn

from .context import get_context

_MISSING = object()
_METHOD_NAMES = ("add_loss", "add_metric", "add_output", "add_hook")


class _PatchState:  # pylint: disable=too-few-public-methods
    patched = False
    original_methods: dict[str, object] = {}


def _delegate_method(name: str):
    def method(self, *args, **kwargs):
        context = get_context(self)
        return getattr(context, name)(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"{nn.Module.__name__}.{name}"
    method.__doc__ = f"Delegate `{name}` to the active ExtraContext."
    return method


def enable_keras_style_api() -> None:
    """Patch torch.nn.Module with Keras-style helper methods."""
    if _PatchState.patched or getattr(nn.Module, "_torchextractx_keras_style_enabled", False):
        _PatchState.patched = True
        return

    conflicts = [
        name for name in _METHOD_NAMES if getattr(nn.Module, name, _MISSING) is not _MISSING
    ]
    if conflicts:
        names = ", ".join(conflicts)
        raise RuntimeError(
            f"Cannot enable Keras-style API; torch.nn.Module already defines: {names}"
        )

    for name in _METHOD_NAMES:
        _PatchState.original_methods[name] = getattr(nn.Module, name, _MISSING)
        setattr(nn.Module, name, _delegate_method(name))

    setattr(nn.Module, "_torchextractx_keras_style_enabled", True)
    _PatchState.patched = True


def disable_keras_style_api() -> None:
    """Restore torch.nn.Module after enable_keras_style_api()."""
    if not _PatchState.patched and not getattr(
        nn.Module, "_torchextractx_keras_style_enabled", False
    ):
        return

    for name in _METHOD_NAMES:
        original = _PatchState.original_methods.get(name, _MISSING)
        if original is _MISSING:
            if hasattr(nn.Module, name):
                delattr(nn.Module, name)
        else:
            setattr(nn.Module, name, original)

    _PatchState.original_methods.clear()
    if hasattr(nn.Module, "_torchextractx_keras_style_enabled"):
        delattr(nn.Module, "_torchextractx_keras_style_enabled")
    _PatchState.patched = False


def is_keras_style_api_enabled() -> bool:
    """Return whether torch.nn.Module has been patched."""
    return bool(getattr(nn.Module, "_torchextractx_keras_style_enabled", False))
