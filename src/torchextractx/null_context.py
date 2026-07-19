"""No-op context used when modules are called outside ExtraContext."""

from __future__ import annotations

from typing import Callable, Literal
import warnings
import weakref

import torch
from torch import nn


ReduceOps = Literal["sum", "mean", "max", "min"]
NullContextBehavior = Literal["warn", "silent", "raise"]

_MISSING = object()


def configure_null_context_behavior(behavior: NullContextBehavior) -> NullContextBehavior:
    """Configure what happens when recording data without an active ExtraContext.

    Returns the previous behavior so callers can restore it after a scoped change.
    """
    if behavior not in ("warn", "silent", "raise"):
        raise ValueError("NullContext behavior must be one of: 'warn', 'silent', 'raise'.")

    previous = NullContext.behavior
    NullContext.behavior = behavior
    return previous


def get_null_context_behavior() -> NullContextBehavior:
    """Return the current NullContext behavior."""
    return NullContext.behavior


class NullContext:
    """A no-op context returned when a module is not inside ExtraContext."""

    active = False
    behavior: NullContextBehavior = "warn"

    _warned_modules: weakref.WeakKeyDictionary[nn.Module, set[str]] = weakref.WeakKeyDictionary()

    def __init__(self, module: nn.Module | None = None):
        self._module_ref = weakref.ref(module) if module is not None else None
        self._local_warned_ops: set[str] = set()

    def _module(self) -> nn.Module | None:
        return self._module_ref() if self._module_ref is not None else None

    def _warned_ops(self) -> set[str]:
        module = self._module()
        if module is None:
            return self._local_warned_ops
        return self._warned_modules.setdefault(module, set())

    def _should_report(self) -> bool:
        module = self._module()
        return True if module is None else bool(module.training)

    def _format_message(self, operation: str, name: object = _MISSING) -> str:
        if name is _MISSING:
            operation_text = f"{operation}()"
        else:
            operation_text = f"{operation}({name!r})"
        return (
            f"No active ExtraContext. {operation_text} was ignored. "
            "Wrap the forward pass with `with ExtraContext(model) as ctx:` to collect it."
        )

    def _report_ignored(self, operation: str, name: object = _MISSING) -> None:
        if not self._should_report():
            return

        behavior = get_null_context_behavior()
        if behavior == "silent":
            return

        message = self._format_message(operation, name)
        if behavior == "raise":
            raise RuntimeError(message)

        warned_ops = self._warned_ops()
        if operation in warned_ops:
            return

        warned_ops.add(operation)
        warnings.warn(message, UserWarning, stacklevel=3)

    def add_loss(self, prefix: str, loss: torch.Tensor, op: ReduceOps = "sum"):
        """Ignore a loss term and optionally report the missing context."""
        _ = loss, op
        self._report_ignored("add_loss", prefix)

    def add_metric(self, prefix: str, metric: torch.Tensor, op: ReduceOps = "mean"):
        """Ignore a metric term and optionally report the missing context."""
        _ = metric, op
        self._report_ignored("add_metric", prefix)

    def add_output(self, prefix: str, output: torch.Tensor):
        """Ignore an output tensor and optionally report the missing context."""
        _ = output
        self._report_ignored("add_output", prefix)

    def add_hook(self, prefix: str, hook: Callable):
        """Ignore a hook and optionally report the missing context."""
        _ = hook
        self._report_ignored("add_hook", prefix)

    def log(self, *args, **kwargs):
        """Ignore a log call and optionally report the missing context."""
        _ = args, kwargs
        self._report_ignored("log")

    def put(self, key: str, value):
        """Ignore a shared-state write and optionally report the missing context."""
        _ = value
        self._report_ignored("put", key)

    def get(self, key: str, default=None):
        """Return the default value because no shared state is active."""
        _ = key
        return default

    def pop(self, key: str, default=None):
        """Return the default value because no shared state is active."""
        _ = key
        return default

    def has(self, key: str) -> bool:
        """Return False because no shared state is active."""
        _ = key
        return False

    def __getitem__(self, key: str):
        raise KeyError(f"No active ExtraContext. Key {key!r} is unavailable.")

    def __setitem__(self, key: str, value) -> None:
        self.put(key, value)

    def get_losses(self, default_op: ReduceOps = "sum") -> dict[str, torch.Tensor]:
        """Return no losses because no collection context is active."""
        _ = default_op
        return {}

    def get_metrics(self, default_op: ReduceOps = "mean") -> dict[str, torch.Tensor]:
        """Return no metrics because no collection context is active."""
        _ = default_op
        return {}

    def get_outputs(self) -> dict[str, torch.Tensor]:
        """Return no outputs because no collection context is active."""
        return {}

    @property
    def losses(self) -> dict[str, list[torch.Tensor]]:
        """Return the empty raw loss store."""
        return {}

    @property
    def hooks(self) -> dict[str, list[Callable]]:
        """Return the empty raw hook store."""
        return {}

    def get_module_prefixes(self, module: nn.Module) -> list[str]:
        """Return no module prefixes because no model has been registered."""
        _ = module
        return []
