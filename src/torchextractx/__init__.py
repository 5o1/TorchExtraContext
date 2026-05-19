"""
TorchExtraContext: A PyTorch utility library for managing extra losses, metrics, and outputs.

This library provides a context manager for non-invasively collecting auxiliary losses,
metrics, and outputs from deep nested PyTorch modules without modifying module interfaces.
"""

from .torchextractx import (
    ExtraContext,
    add_loss,
    add_metric,
    add_hook,
    add_output,
    get_context,
    log,
)

__version__ = "0.1.0"
__author__ = "5o1"
__license__ = "LICENSE"

__all__ = [
    "ExtraContext",
    "add_loss",
    "add_metric",
    "add_hook",
    "add_output",
    "get_context",
    "log",
]
