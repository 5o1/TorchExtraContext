# TorchExtraContext

[English](README.md) | [中文](README_CN.md)

[![CI](https://github.com/5o1/TorchExtraContext/actions/workflows/ci.yml/badge.svg)](https://github.com/5o1/TorchExtraContext/actions/workflows/ci.yml)
[![Release](https://github.com/5o1/TorchExtraContext/actions/workflows/release.yml/badge.svg)](https://github.com/5o1/TorchExtraContext/actions/workflows/release.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch context manager for collecting auxiliary losses, metrics, intermediate outputs, hooks, and shared state from nested `torch.nn.Module` objects without changing `forward()` signatures.

## What It Does

In PyTorch, auxiliary losses and intermediate values are often produced deep inside nested `torch.nn.Module` objects. The traditional approach is to return them layer by layer:

```python
def forward(self, x):
    x, loss_a = self.block_a(x)
    x, loss_b = self.block_b(x)
    logits = self.head(x)
    return logits, {"loss_a": loss_a, "loss_b": loss_b}
```

This approach couples auxiliary data collection to the return values of every module on the call path. Modules that do not create the auxiliary data still need to receive, merge, and return it.

Instead of modifying `forward()` signatures layer by layer, we prefer to register the data in the module where it is produced:

```python
import torchextractx.keras_style  # Enable Keras-style API

def forward(self, x):
    x = self.layer1(x)
    self.add_loss("loss_a", compute_loss(x))
    return x
```

Then collect all losses in one line:

```python
with ExtraContext(model) as ctx:
    output = model(x)
    all_losses = ctx.get_losses()  # Done
```

## 🚀 Installation

### From PyPI
```bash
pip install torchextractx
```

### From source
```bash
git clone https://github.com/5o1/TorchExtraContext.git
cd TorchExtraContext
pip install -e .
```

### With dev tools
```bash
git clone https://github.com/5o1/TorchExtraContext.git
cd TorchExtraContext
pip install -e ".[dev]"  # pytest, black, mypy
```

## 📖 Quick Start

### Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchextractx import ExtraContext, get_context  # Step #1: Import the context helpers

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        get_context(self).add_loss("feature_mean_loss", x.mean())  # Step #2: Register the loss where it is created
        
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


model = Classifier()
optimizer = torch.optim.Adam(model.parameters())
x = torch.randn(32, 10)
targets = torch.randint(0, 2, (32,))

with ExtraContext(model) as ctx:  # Step #3: Create the collection context in the training step
    logits = model(x)
    main_loss = F.cross_entropy(logits, targets)
    extra_losses = ctx.get_losses()  # Step #4: Collect losses registered inside the model

    total_loss = main_loss
    for loss in extra_losses.values():
        total_loss = total_loss + 0.1 * loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Enable Keras Style

```python
import torchextractx.keras_style  # Step #1: Enable torch.nn.Module.add_loss
from torchextractx import ExtraContext

class FeatureExtractor(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        self.add_loss("feature_mean_loss", x.mean())  # Step #2: Register the loss where it is created

        x = self.fc2(x)
        return x


...

with ExtraContext(model) as ctx:  # Step #3: Create the collection context in the training step
    logits = model(x)
    main_loss = F.cross_entropy(logits, targets)
    extra_losses = ctx.get_losses()  # Step #4: Collect losses registered inside the model

    ...
```

### Use with Lightning

Lightning's `LightningModule` is a subclass of `torch.nn.Module`, so the same APIs can be used in Lightning `training_step`, `validation_step`, or inside module `forward()` methods. Wrap one model call with `ExtraContext(self)` in the step, then collect losses, metrics, hooks, shared objects, and logs from the context.

## 📚 Other Usage

### Export Metrics

`loss` usually goes into `total_loss` and participates in `backward()`. A `metric` is only for monitoring, such as accuracy. Keeping them separate avoids mixing observation-only values into the optimized loss.

```python
class ClassifierHead(nn.Module):
    def __init__(self):
        ...

    def forward(self, x, targets):
        logits = self.classifier(x)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
        get_context(self).add_metric("train/accuracy", acc)  # For monitoring only

        return logits


head = ClassifierHead()
with ExtraContext(head) as ctx:
    logits = head(x, targets)
    main_loss = F.cross_entropy(logits, targets)
    metrics = ctx.get_metrics()
```

### Hooks

A hook can store a callback that should run after `forward()`. ExtraContext only collects these functions; the training step decides when to execute them.

```python
class FeatureBlock(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        y = self.block(x)
        get_context(self).add_hook("feature_norm", lambda: y.detach().norm().item())
        return y


block = FeatureBlock()
with ExtraContext(block) as ctx:
    output = block(x)
    hook_results = {
        name: [hook() for hook in hooks]
        for name, hooks in ctx.hooks.items()
    }
```

### Shared Objects

`put/get/pop` can pass values out to the training step, but it can also share objects between modules in the same forward pass. This bypasses `forward()` arguments and return values. **Note that this method is not very safe**: it relies on key names and module call order, so it is best kept to small experimental uses.

```python
class Encoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        latent = self.backbone(x)
        get_context(self).put("latent", latent)
        return latent


class Decoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ctx = get_context(self)
        latent = ctx.pop("latent")
        return self.head(x + latent)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


model = AutoEncoder()
with ExtraContext(model):
    output = model(x)
```

### Shared Logger

Pass a logger when creating `ExtraContext` if several modules need to write logs. Modules still call `get_context(self).log(...)`; **Note:** Keras style does not monkey patch `log` onto `torch.nn.Module`.

```python
records = []


def logger(event, **fields):
    records.append((event, fields))


class FeatureBlock(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        y = self.block(x)
        get_context(self).log("feature_block", mean=y.detach().mean().item())
        return y


block = FeatureBlock()
with ExtraContext(block, logger=logger):
    output = block(x)
```

## 🔧 API

### Package Map

| Import path | Public symbols | Notes |
| --- | --- | --- |
| `torchextractx` | `ExtraContext`, `get_context`, `NullContext`, `configure_null_context_behavior`, `get_null_context_behavior`, `enable_keras_style_api`, `disable_keras_style_api`, `is_keras_style_api_enabled` | Preferred public import path for application code. |
| `torchextractx.context` | `ExtraContext`, `get_context` | Active context manager and module lookup helper. |
| `torchextractx.null_context` | `NullContext`, `configure_null_context_behavior`, `get_null_context_behavior` | No-op fallback and missing-context policy. |
| `torchextractx.keras_style` | side-effect import | Import this module to enable Keras-style helpers immediately. |

### `torchextractx.context.ExtraContext`

| Member | Kind | Access | Notes |
| --- | --- | --- | --- |
| `ExtraContext(root_module, logger=None)` | class | `with ExtraContext(model) as ctx:` | Main context manager for nested modules. |
| `add_loss(prefix, loss, op="sum")` | method | `ctx.add_loss(...)` | Register an auxiliary loss tensor. |
| `add_metric(prefix, metric, op="mean")` | method | `ctx.add_metric(...)` | Register a metric tensor. |
| `add_output(prefix, output)` | method | `ctx.add_output(...)` | Save an intermediate output tensor with shape checks. |
| `add_hook(prefix, hook)` | method | `ctx.add_hook(...)` | Register a callable hook. |
| `put(key, value)` | method | `ctx.put(...)` | Store shared state for module-to-module communication. |
| `get(key, default=None)` | method | `ctx.get(...)` | Read shared state. |
| `pop(key, default=None)` | method | `ctx.pop(...)` | Remove and return shared state. |
| `has(key)` | method | `ctx.has(...)` | Check whether a shared state item exists. |
| `get_losses(default_op="sum")` | method | `ctx.get_losses(...)` | Return reduced losses as a dictionary. |
| `get_metrics(default_op="mean")` | method | `ctx.get_metrics(...)` | Return reduced metrics as a dictionary. |
| `get_outputs()` | method | `ctx.get_outputs(...)` | Return saved outputs as a dictionary. |
| `get_module_prefixes(module)` | method | `ctx.get_module_prefixes(...)` | Return model path names for a module. |
| `log(*args, **kwargs)` | method | `ctx.log(...)` | Forward logging calls to the configured logger. |
| `losses` | property | `ctx.losses` | Raw loss store. |
| `hooks` | property | `ctx.hooks` | Raw hook store. |

### `torchextractx.null_context.NullContext`

| Member | Kind | Access | Notes |
| --- | --- | --- | --- |
| `NullContext(module=None)` | class | `NullContext(module)` or returned by `get_context(module)` | No-op context used when no `ExtraContext` is active. |
| `add_loss`, `add_metric`, `add_output`, `add_hook`, `put`, `log` | methods | `ctx.add_loss(...)`, `ctx.put(...)`, etc. | Ignored; may warn, stay silent, or raise based on the configured behavior. |
| `get`, `pop`, `has`, `get_losses`, `get_metrics`, `get_outputs`, `get_module_prefixes` | methods | `ctx.get(...)`, `ctx.get_losses()`, etc. | Return defaults or empty results. |
| `losses`, `hooks` | properties | `ctx.losses`, `ctx.hooks` | Return empty stores. |
| `__getitem__` | method | `ctx[key]` | Raises `KeyError` because no shared state is active. |

### `torchextractx.keras_style`

| Import | Effect | Notes |
| --- | --- | --- |
| `import torchextractx.keras_style` | Enables `torch.nn.Module.add_loss`, `add_metric`, `add_output`, and `add_hook`. | Process-wide side effect import. |
| `from torchextractx import enable_keras_style_api` | Same effect explicitly. | Useful when you want the switch in code instead of import side effects. |
| `disable_keras_style_api()` | Restores the original `torch.nn.Module` methods. | Re-exported from `torchextractx`. |
| `is_keras_style_api_enabled()` | Reports whether the shim is active. | Re-exported from `torchextractx`. |

**Note:** Use `get_context(self).log(...)` for logging; Keras style does not monkey patch `log` onto `torch.nn.Module`.

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

## 🚢 Release Process

- The canonical version lives in `pyproject.toml` under `[project].version`.
- Release tags must use `vX.Y.Z` and match that version exactly.
- GitHub Release notes are sourced from `CHANGELOG.md`.
- Pushing a matching tag runs tests, builds distributions, creates a GitHub Release, and publishes to PyPI through PyPI Trusted Publishing.
- PyPI publishing does not require a PyPI token; authentication is handled by GitHub Actions OpenID Connect.

## 🤝 Contributing

Welcome to submit issues and PRs.
