"""TorchExtraContext: collect auxiliary losses, metrics, outputs, and shared state."""

from importlib.metadata import PackageNotFoundError, version

from ._keras_style import (
    disable_keras_style_api,
    enable_keras_style_api,
    is_keras_style_api_enabled,
)
from .context import ExtraContext, get_context
from .null_context import (
    NullContext,
    configure_null_context_behavior,
    get_null_context_behavior,
)

try:
    __version__ = version("torchextractx")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "5o1"
__license__ = "LICENSE"

__all__ = [
    "ExtraContext",
    "NullContext",
    "configure_null_context_behavior",
    "get_context",
    "get_null_context_behavior",
    "enable_keras_style_api",
    "disable_keras_style_api",
    "is_keras_style_api_enabled",
]
