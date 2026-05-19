# TorchExtraContext

[English](README.md) | [中文](README_CN.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch utility library for collecting losses, metrics, and outputs from nested modules.

## Why Use This?

When doing deep learning, you often need to add losses in the middle of the network, like distillation loss or auxiliary classifier loss. Common approaches:

1. Pass losses up layer by layer, which requires modifying `forward()` signatures
2. Use global variables or callbacks, making code complicated

This library lets you register losses directly in modules without changing interfaces:

```python
def forward(self, x):
    x = self.layer1(x)
    add_loss(self, "loss_a", compute_loss(x))
    return x
```

Then collect all losses in one line:

```python
with ExtraContext(model) as ctx:
    output = model(x)
    all_losses = ctx.get_losses()  # Done
```

## 🚀 Installation

### From source
```bash
pip install -e .
```

### With dev tools
```bash
pip install -e ".[dev]"  # pytest, black, mypy
```

## 📖 Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from torchextractx import ExtraContext, add_loss, add_metric

# Define a model with intermediate losses
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        
        # Register an auxiliary loss (no interface modification needed)
        aux_loss = x.mean()
        add_loss(self, "auxiliary_loss", aux_loss)
        
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.classifier = nn.Linear(10, 2)
    
    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x

# Training loop
model = Classifier()
optimizer = torch.optim.Adam(model.parameters())

# Use ExtraContext
with ExtraContext(model) as ctx:
    x = torch.randn(32, 10)
    logits = model(x)
    
    # Main loss
    main_loss = torch.nn.functional.cross_entropy(logits, targets)
    
    # Collect losses from all nested modules
    aux_losses = ctx.get_losses()  # {'auxiliary_loss': tensor(...), ...}
    
    # Total loss = main loss + weighted auxiliary losses
    total_loss = main_loss
    for name, loss_val in aux_losses.items():
        print(f"aux {name}: {loss_val:.4f}")
        total_loss = total_loss + 0.1 * loss_val  # weight is tunable
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## 🔧 API Reference

### `ExtraContext`

Main context manager for collecting auxiliary information.

```python
ctx = ExtraContext(root_module, logger=None)
```

**Parameters:**
- `root_module` (nn.Module): Root module to scan
- `logger` (Callable, optional): Logger function

**Methods:**

#### `add_loss(prefix, loss, op="sum")`

Register a loss.
- `prefix`: Name of the loss
- `loss`: Loss value (tensor)
- `op`: Merge strategy, default `"sum"` | options: `"mean"` `"max"` `"min"`

#### `add_metric(prefix, metric, op="mean")`

Register metrics (e.g., accuracy, F1), merged using average by default.

#### `add_output(prefix, output)`

Save intermediate output for later analysis, enforces shape consistency.

#### `add_hook(prefix, hook)`

Register a hook function.

#### `get_losses(default_op="sum")`

Get all registered losses as a dictionary.

#### `get_metrics(default_op="mean")`

Get all registered metrics as a dictionary.

#### `get_outputs()`

Get saved output tensors.

#### `get_module_prefixes(module)`

Query the path name of a module in the model. Useful for debugging.

### Helper Functions

#### `add_loss(module, prefix, loss_term, op="sum")`

Register a loss in a module:

```python
def forward(self, x):
    x = self.process(x)
    add_loss(self, "aux_loss", x.sum())
    return x
```

#### `add_metric(module, prefix, metric_term, op="mean")`

Register a metric.

#### `add_output(module, prefix, output)`

Register an output.

#### `add_hook(module, prefix, hook)`

Register a hook.

#### `get_context(module)`

Get the context object in a module for storing debug data:

```python
if ctx := get_context(self):
    ctx['debug_data'] = some_value
```

#### `log(module, *args, **kwargs)`

Log debug information through the context.

## 📚 Advanced Usage

### Multi-Task Learning

Combine multiple losses with different weights:

```python
with ExtraContext(model) as ctx:
    output = model(x)
    losses = ctx.get_losses()
    
    total_loss = primary_loss
    for name, loss_val in losses.items():
        weight = {'loss_a': 0.5, 'loss_b': 0.3}.get(name, 0.1)
        total_loss += weight * loss_val
    
    total_loss.backward()
```

### Custom Merge Strategy

Different losses with different merge strategies:

```python
with ExtraContext(model) as ctx:
    output = model(x)
    add_loss(model.layer1, "loss_a", tensor_a, op="mean")
    add_loss(model.layer2, "loss_b", tensor_b, op="max")
    losses = ctx.get_losses()
```

## ⚠️ Notes

### Thread Safety

Not thread-safe. Don't use from multiple threads simultaneously.

### Nested Contexts

Nested ExtraContext on the same model is not allowed and will raise an error:

```python
with ExtraContext(model):
    with ExtraContext(model):  # Raises ValueError
        pass
```

### Memory Management

All data is cleared after exiting the `with` block. Don't access outside:

```python
with ExtraContext(model) as ctx:
    output = model(x)
    losses = ctx.get_losses()  # OK

losses = ctx.get_losses()  # Error
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Or with unittest:

```bash
python -m unittest discover tests/ -v
```

## 📋 Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0

## 📝 License

MIT - see [LICENSE](LICENSE)

## 🤝 Contributing

Welcome to submit issues and PRs.
