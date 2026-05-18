import warnings
import torch
from torch import nn
from typing import Callable, Literal, List
from collections import OrderedDict
import uuid

class ExtraContext:
    """
    A context manager for managing extra losses, hooks, and objects in a PyTorch module.
    This context manager allows you to register additional losses, hooks, and objects
    that can be used during training or evaluation. It ensures that these additional
    components are properly cleaned up after use.
    
    Usage:
        with ExtraContext(model) as ctx:
            # Register extra losses, hooks, or objects
            model.extra_context.add_loss("loss_name", loss_tensor)
            model.extra_context.add_hook("hook_name", hook_function)
            model.extra_context.add_object("object_name", some_object)
    """
    ReduceOps = Literal["sum", "mean", "max", "min"]


    def __init__(self, root_module: nn.Module, logger:Callable = None):
        self.__root_module = root_module
        self.logger = logger

        self._context_id = str(uuid.uuid4())

        self.__registered_modules = OrderedDict()
        self.__registered_modules_prefix = OrderedDict()
        self.__extra_losses = dict()
        self.__extra_metrics = dict()
        self.__extra_outputs = dict()
        self.__extra_hooks = dict()
        self.__extra_objects = dict()

        self.__op_losses = dict()
        self.__op_metrics = dict()

    def __enter__(self):
        seen_module_ids = set()
        
        for prefix, module in self.__root_module.named_modules():
            module_id = id(module)
            
            # Deduplicate by module id (handle shared submodules)
            if module_id in seen_module_ids:
                if module_id not in self.__registered_modules_prefix:
                    self.__registered_modules_prefix[module_id] = []
                self.__registered_modules_prefix[module_id].append(prefix)
                continue
            
            seen_module_ids.add(module_id)
            
            # Check if already bound to a different context
            if hasattr(module, "extra_context"):
                existing_ctx = getattr(module, "extra_context")
                if existing_ctx._context_id != self._context_id:
                    raise ValueError(
                        f"Module {module._get_name()} at {prefix} is already bound to another ExtraContext. "
                        f"Nested or concurrent ExtraContext is not allowed."
                    )
                # Same context, skip
                continue
            
            # Bind this context to the module
            setattr(module, "extra_context", self)
            
            self.__registered_modules[module_id] = module
            self.__registered_modules_prefix[module_id] = [prefix]
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__extra_losses = None
        self.__extra_metrics = None
        self.__extra_outputs = None
        self.__extra_hooks = None
        self.__extra_objects = None

        for _, module in self.__registered_modules.items():
            if hasattr(module, "extra_context"):
                delattr(module, "extra_context")
    
    def __getitem__(self, key):
        if self.__extra_objects is None:
            raise ValueError("Objects dict has been cleared. Users should not access context manager after exiting context. This could be a bug.")
        return self.__extra_objects[key]

    def __setitem__(self, key, value):
        if self.__extra_objects is None:
            raise ValueError("Objects dict has been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_objects[key] = value

    def add_loss(self, prefix: str, loss: torch.Tensor, op: ReduceOps = "sum"):
        """
        Add a loss term to the context.
        """
        if self.__extra_losses is None:
            raise ValueError("Losses have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_losses.setdefault(prefix, []).append(loss)
        self.__op_losses[prefix] = op

    def add_metric(self, prefix: str, metric: torch.Tensor, op: ReduceOps = "mean"):
        """
        Add a metric term to the context.
        """
        if self.__extra_metrics is None:
            raise ValueError("Metrics have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_metrics.setdefault(prefix, []).append(metric)
        self.__op_metrics[prefix] = op

    def add_output(self, prefix: str, output: torch.Tensor):
        """
        Add an output tensor to the context.
        """
        if self.__extra_outputs is None:
            raise ValueError("Outputs have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_outputs.setdefault(prefix, []).append(output)

        # Check the shape must be consistent
        if len(self.__extra_outputs[prefix]) > 1:
            first_shape = self.__extra_outputs[prefix][0].shape
            last_shape = self.__extra_outputs[prefix][-1].shape
            if first_shape != last_shape:
                raise ValueError(f"Extra Output shape mismatch for prefix {prefix}. Expected {first_shape}, but got {last_shape}.")

    def add_hook(self, prefix: str, hook: Callable):
        """
        Add a hook to the context.
        """
        if self.__extra_hooks is None:
            raise ValueError("Hooks have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_hooks.setdefault(prefix, []).append(hook)

    def log(self, *args, **kwargs):
        if self.logger is None:
            warnings.warn("No logger is set for ExtraContext. Logging will be ignored.", UserWarning, stacklevel=2)
            return
        res = self.logger(*args, **kwargs)
        return res
    

    def _tensors_reduce(self, tensors: List[torch.Tensor], op: ReduceOps = "mean"):
        """
        Reduce a tensor using the specified operation.
        """
        if len(tensors) == 0:
            return None
        if len(tensors) == 1:
            return tensors[0]
        if op == "sum":
            return torch.sum(torch.stack(tensors, dim=0), dim=0)
        elif op == "mean":
            return torch.mean(torch.stack(tensors, dim=0), dim=0)
        elif op == "max":
            return torch.max(torch.stack(tensors, dim=0), dim=0).values
        elif op == "min":
            return torch.min(torch.stack(tensors, dim=0), dim=0).values
        else:
            raise ValueError(f"Unsupported operation {op}. Supported operations are {ExtraContext.ReduceOps}.")

    def get_losses(self, default_op: ReduceOps = "sum"):
        """
        Get the registered losses in the context.
        """
        if self.__extra_losses is None:
            raise ValueError("Losses have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        
        losses = {}
        for loss_name, loss_terms in self.__extra_losses.items():
            losses[loss_name] = self._tensors_reduce(loss_terms, op=self.__op_losses.get(loss_name, default_op))
        return losses

    def get_metrics(self, default_op: ReduceOps = "mean"):
        """
        Get the registered metrics in the context.
        """
        if self.__extra_metrics is None:
            raise ValueError("Metrics have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        
        metrics = {}
        for metric_name, metric_terms in self.__extra_metrics.items():
            if "[int]" in metric_name:
                metrics[metric_name.replace("[int]", "")] = self._tensors_reduce(metric_terms, op=self.__op_metrics.get(metric_name, default_op)).int()
            else:
                metrics[metric_name] = self._tensors_reduce(metric_terms, op=self.__op_metrics.get(metric_name, default_op))
        return metrics

    def get_outputs(self):
        """
        Get the registered outputs in the context.
        """
        if self.__extra_outputs is None:
            raise ValueError("Outputs have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        
        outputs = {}
        for output_name, output_terms in self.__extra_outputs.items():
            outputs[output_name] = torch.stack(output_terms, dim=0) if len(output_terms) > 1 else output_terms[0].unsqueeze(0)
        return outputs
    
    @property
    def losses(self):
        if self.__extra_losses is None:
            raise ValueError("Losses have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        return self.__extra_losses

    @property
    def hooks(self):
        if self.__extra_hooks is None:
            raise ValueError("Hooks have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        return self.__extra_hooks
    
    def get_module_prefixes(self, module: nn.Module) -> List[str]:
        """
        Get all prefixes (paths) where a module appears in the registered model hierarchy.
        Useful for debugging and understanding module location in the model.
        
        Args:
            module: The module to query.
        
        Returns:
            List of prefixes (paths) where this module was registered.
        """
        module_id = id(module)
        return self.__registered_modules_prefix.get(module_id, [])


def register_extra_loss(module: nn.Module, prefix: str, loss_term: torch.Tensor, op: ExtraContext.ReduceOps = "sum"):
    """
    Register an extra loss term to a module.
    """
    if not hasattr(module, "extra_context"):
        if module.training:
            warnings.warn("Training does not launch with an ExtraContext. This loss will be ignored.", UserWarning,stacklevel=2)
        return
    module.extra_context.add_loss(prefix, loss_term, op=op)


def register_extra_metric(module: nn.Module, prefix: str, metric_term: torch.Tensor, op: ExtraContext.ReduceOps = "mean"):
    """
    Register an extra metric term to a module.
    """
    if not hasattr(module, "extra_context"):
        if module.training:
            warnings.warn("Training does not launch with an ExtraContext. This metric will be ignored.", UserWarning, stacklevel=2)
        return
    module.extra_context.add_metric(prefix, metric_term, op=op)


def register_extra_hook(module: nn.Module, prefix: str, hook: Callable):
    """
    Register an extra hook to a module.
    """
    if not hasattr(module, "extra_context"):
        if module.training:
            warnings.warn("Training does not launch with an ExtraContext. This hook will be ignored.", UserWarning,stacklevel=2)
        return
    module.extra_context.add_hook(prefix, hook)

def register_extra_output(module: nn.Module, prefix: str, output: torch.Tensor):
    """
    Register an extra output tensor to a module.
    """
    if not hasattr(module, "extra_context"):
        if module.training:
            warnings.warn("Training does not launch with an ExtraContext. This output will be ignored.", UserWarning, stacklevel=2)
        return
    module.extra_context.add_output(prefix, output)


def get_extra_context(module: nn.Module):
    """
    Get the extra context of a module.
    """
    if not hasattr(module, "extra_context"):
        if module.training:
            warnings.warn("Training does not launch with an ExtraContext. This will return None.", UserWarning, stacklevel=2)
        return None
    return module.extra_context


def log_extra(module: nn.Module, *args, **kwargs):
    """
    Log extra information to the module's logger.
    """
    if not hasattr(module, "extra_context"):
        if module.training:
            warnings.warn("Training does not launch with an ExtraContext. Logging will be ignored.", UserWarning, stacklevel=2)
        return
    return module.extra_context.log(*args, **kwargs)
