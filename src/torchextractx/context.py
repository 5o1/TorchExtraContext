"""Active ExtraContext implementation and module lookup helper."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Literal
import warnings

import torch
from torch import nn

from .null_context import NullContext


ReduceOps = Literal["sum", "mean", "max", "min"]
_MISSING = object()


class ExtraContext:  # pylint: disable=too-many-instance-attributes
    """
    Context manager for collecting extra losses, metrics, outputs, hooks, and shared state.

    Modules can retrieve the current context with `get_context(self)` inside `forward()`
    without changing their call signatures.
    """

    ReduceOps = ReduceOps

    def __init__(self, root_module: nn.Module, logger: Callable | None = None):
        self._root_module = root_module
        self.logger = logger
        self.active = True

        self._registered_modules: OrderedDict[int, nn.Module] = OrderedDict()
        self._registered_modules_prefix: OrderedDict[int, list[str]] = OrderedDict()

        self._losses: dict[str, list[torch.Tensor]] = {}
        self._metrics: dict[str, list[torch.Tensor]] = {}
        self._outputs: dict[str, list[torch.Tensor]] = {}
        self._hooks: dict[str, list[Callable]] = {}
        self._state: dict[str, object] = {}

        self._loss_ops: dict[str, ReduceOps] = {}
        self._metric_ops: dict[str, ReduceOps] = {}
        self._output_shapes: dict[str, torch.Size] = {}

    def __enter__(self):
        self._ensure_active()

        modules_to_bind: list[tuple[int, nn.Module, str]] = []
        seen_module_ids = set()
        for prefix, module in self._root_module.named_modules():
            module_id = id(module)

            if module_id in seen_module_ids:
                self._registered_modules_prefix.setdefault(module_id, []).append(prefix)
                continue

            seen_module_ids.add(module_id)

            if hasattr(module, "extra_context"):
                existing_ctx = getattr(module, "extra_context")
                if existing_ctx is self:
                    continue
                raise ValueError(
                    f"Module {module._get_name()} at {prefix} is already bound to another "
                    "ExtraContext. Nested or concurrent ExtraContext is not allowed."
                )

            modules_to_bind.append((module_id, module, prefix))

        for module_id, module, prefix in modules_to_bind:
            setattr(module, "extra_context", self)
            self._registered_modules[module_id] = module
            self._registered_modules_prefix[module_id] = [prefix]

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        self._losses.clear()
        self._metrics.clear()
        self._outputs.clear()
        self._hooks.clear()
        self._state.clear()
        self._loss_ops.clear()
        self._metric_ops.clear()
        self._output_shapes.clear()

        for module in self._registered_modules.values():
            if getattr(module, "extra_context", None) is self:
                delattr(module, "extra_context")

        self._registered_modules.clear()

    def _ensure_active(self) -> None:
        if not self.active:
            raise ValueError(
                "ExtraContext has been closed. Do not access it after exiting the context."
            )

    def _append_term(self, store: dict[str, list], prefix: str, value) -> None:
        store.setdefault(prefix, []).append(value)

    def __getitem__(self, key: str):
        self._ensure_active()
        return self._state[key]

    def __setitem__(self, key: str, value) -> None:
        self.put(key, value)

    def put(self, key: str, value) -> None:
        """Store a shared object in the active context."""
        self._ensure_active()
        self._state[key] = value

    def get(self, key: str, default=None):
        """Read a shared object from the active context."""
        self._ensure_active()
        return self._state.get(key, default)

    def pop(self, key: str, default=_MISSING):
        """Remove and return a shared object from the active context."""
        self._ensure_active()
        if default is _MISSING:
            return self._state.pop(key)
        return self._state.pop(key, default)

    def has(self, key: str) -> bool:
        """Return whether a shared object exists in the active context."""
        self._ensure_active()
        return key in self._state

    def add_loss(self, prefix: str, loss: torch.Tensor, op: ReduceOps = "sum") -> None:
        """Add a loss term to the context."""
        self._ensure_active()
        self._append_term(self._losses, prefix, loss)
        self._loss_ops[prefix] = op

    def add_metric(self, prefix: str, metric: torch.Tensor, op: ReduceOps = "mean") -> None:
        """Add a metric term to the context."""
        self._ensure_active()
        self._append_term(self._metrics, prefix, metric)
        self._metric_ops[prefix] = op

    def add_output(self, prefix: str, output: torch.Tensor) -> None:
        """Add an output tensor to the context."""
        self._ensure_active()

        expected_shape = self._output_shapes.get(prefix)
        if expected_shape is None:
            self._output_shapes[prefix] = output.shape
        elif expected_shape != output.shape:
            raise ValueError(
                f"Extra Output shape mismatch for prefix {prefix}. "
                f"Expected {expected_shape}, but got {output.shape}."
            )

        self._append_term(self._outputs, prefix, output)

    def add_hook(self, prefix: str, hook: Callable) -> None:
        """Add a hook to the context."""
        self._ensure_active()
        self._append_term(self._hooks, prefix, hook)

    def log(self, *args, **kwargs):
        """Log through the configured logger, if one exists."""
        if self.logger is None:
            warnings.warn(
                "No logger is set for ExtraContext. Logging will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return self.logger(*args, **kwargs)

    def _tensors_reduce(
        self, tensors: list[torch.Tensor], op: ReduceOps = "mean"
    ) -> torch.Tensor:
        """Reduce a tensor list using the specified operation."""
        if len(tensors) == 0:
            raise ValueError("Cannot reduce an empty tensor list.")
        if len(tensors) == 1:
            return tensors[0]
        if op == "sum":
            return torch.sum(torch.stack(tensors, dim=0), dim=0)
        if op == "mean":
            return torch.mean(torch.stack(tensors, dim=0), dim=0)
        if op == "max":
            return torch.max(torch.stack(tensors, dim=0), dim=0).values
        if op == "min":
            return torch.min(torch.stack(tensors, dim=0), dim=0).values
        raise ValueError(
            f"Unsupported operation {op}. Supported operations are {ExtraContext.ReduceOps}."
        )

    def get_losses(self, default_op: ReduceOps = "sum") -> dict[str, torch.Tensor]:
        """Get the registered losses in the context."""
        self._ensure_active()

        losses = {}
        for loss_name, loss_terms in self._losses.items():
            losses[loss_name] = self._tensors_reduce(
                loss_terms, op=self._loss_ops.get(loss_name, default_op)
            )
        return losses

    def get_metrics(self, default_op: ReduceOps = "mean") -> dict[str, torch.Tensor]:
        """Get the registered metrics in the context."""
        self._ensure_active()

        metrics = {}
        for metric_name, metric_terms in self._metrics.items():
            value = self._tensors_reduce(
                metric_terms, op=self._metric_ops.get(metric_name, default_op)
            )
            if "[int]" in metric_name:
                metrics[metric_name.replace("[int]", "")] = value.int()
            else:
                metrics[metric_name] = value
        return metrics

    def get_outputs(self) -> dict[str, torch.Tensor]:
        """Get the registered outputs in the context."""
        self._ensure_active()

        outputs = {}
        for output_name, output_terms in self._outputs.items():
            outputs[output_name] = (
                torch.stack(output_terms, dim=0)
                if len(output_terms) > 1
                else output_terms[0].unsqueeze(0)
            )
        return outputs

    @property
    def losses(self) -> dict[str, list[torch.Tensor]]:
        """Return the raw loss store."""
        self._ensure_active()
        return self._losses

    @property
    def hooks(self) -> dict[str, list[Callable]]:
        """Return the raw hook store."""
        self._ensure_active()
        return self._hooks

    def get_module_prefixes(self, module: nn.Module) -> list[str]:
        """
        Get all prefixes (paths) where a module appears in the registered model hierarchy.
        Useful for debugging and understanding module location.
        """
        return self._registered_modules_prefix.get(id(module), [])


def get_context(module: nn.Module) -> ExtraContext | NullContext:
    """Return the active ExtraContext for a module, or a no-op NullContext."""
    context = getattr(module, "extra_context", None)
    if context is None:
        return NullContext(module)
    return context
