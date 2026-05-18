"""
TorchExtraContext: A PyTorch utility library for managing extra losses, metrics, and outputs.

This library provides a context manager for non-invasively collecting auxiliary losses,
metrics, and outputs from deep nested PyTorch modules without modifying module interfaces.
"""

from .torchextractx import (
    ExtraContext,
    register_extra_loss,
    register_extra_metric,
    register_extra_hook,
    register_extra_output,
    get_extra_context,
    log_extra,
)

__version__ = "0.1.0"
__author__ = "5o1"
__license__ = "LICENSE"

__all__ = [
    "ExtraContext",
    "register_extra_loss",
    "register_extra_metric",
    "register_extra_hook",
    "register_extra_output",
    "get_extra_context",
    "log_extra",
]
