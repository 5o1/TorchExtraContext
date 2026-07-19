# TorchExtraContext

[English](README.md) | [中文](README_CN.md)

[![CI](https://github.com/5o1/TorchExtraContext/actions/workflows/ci.yml/badge.svg)](https://github.com/5o1/TorchExtraContext/actions/workflows/ci.yml)
[![Release](https://github.com/5o1/TorchExtraContext/actions/workflows/release.yml/badge.svg)](https://github.com/5o1/TorchExtraContext/actions/workflows/release.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个 PyTorch context manager 工具，用于从嵌套的 `torch.nn.Module` 中收集 auxiliary loss、指标、中间输出、钩子和共享状态，同时不需要修改 `forward()` 签名。

## 它做了什么？

在 PyTorch 里，辅助损失和中间值经常是在很深的 `torch.nn.Module` 里产生的。传统做法是把这些值一层层返回：

```python
def forward(self, x):
    x, loss_a = self.block_a(x)
    x, loss_b = self.block_b(x)
    logits = self.head(x)
    return logits, {"loss_a": loss_a, "loss_b": loss_b}
```

一旦新增了一个 loss，网络越深，我们就要修改越多的层的返回值。这么写太罗嗦了，我们更希望只在产生数据的模块里注册一次：

```python
import torchextractx.keras_style  # 启用 Keras 风格 API

def forward(self, x):
    x = self.layer1(x)
    self.add_loss("loss_a", compute_loss(x))
    return x
```

然后在 training step 收集损失：

```python
with ExtraContext(model) as ctx:
    output = model(x)
    all_losses = ctx.get_losses()  # 收集loss
```

现在让我们来实现你的愿望！

## 🚀 安装

### 从 PyPI 安装
```bash
pip install torchextractx
```

### 从源代码安装
```bash
git clone https://github.com/5o1/TorchExtraContext.git
cd TorchExtraContext
pip install -e .
```

### 使用开发工具安装
```bash
git clone https://github.com/5o1/TorchExtraContext.git
cd TorchExtraContext
pip install -e ".[dev]"  # 包含 pytest, black, mypy
```

## 📖 快速开始

### 用法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchextractx import ExtraContext, get_context  # Step #1: 导入上下文和查询函数

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        get_context(self).add_loss("feature_mean_loss", x.mean())  # Step #2: loss 产生时注册一次
        
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

with ExtraContext(model) as ctx:  # Step #3: 在 training step 创建收集上下文
    logits = model(x)
    main_loss = F.cross_entropy(logits, targets)
    extra_losses = ctx.get_losses()  # Step #4: 收集模型内部注册的 loss

    total_loss = main_loss
    for loss in extra_losses.values():
        total_loss = total_loss + 0.1 * loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### 启用 keras 风格

```python
import torchextractx.keras_style  # Step #1: 启用 torch.nn.Module.add_loss
from torchextractx import ExtraContext

class FeatureExtractor(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        self.add_loss("feature_mean_loss", x.mean())  # Step #2: loss 产生时注册一次

        x = self.fc2(x)
        return x


...

with ExtraContext(model) as ctx:  # Step #3: 在 training step 创建收集上下文
    logits = model(x)
    extra_losses = ctx.get_losses()  # Step #4: 收集模型内部注册的 loss

    ...
```

### 在lightning中使用

Lightning 的 `LightningModule` 是 `torch.nn.Module` 的子类。因此，前面这些用法也可以放到 Lightning 的 `training_step`、`validation_step` 或模块内部的 `forward()` 中使用；只要用 `ExtraContext(self)` 包住一次模型调用，就可以在 step 里统一收集 loss、metric、hook、共享对象和日志。

## 📚 其它用法

### 传出 metrics

`loss` 通常会进入 `total_loss` 并参与 `backward()`；`metric` 只用于观察训练状态，比如 accuracy。把它们分开记录，可以避免把只用于检测的值误加到 loss 里。

```python
class ClassifierHead(nn.Module):
    def __init__(self):
        ...

    def forward(self, x, targets):
        logits = self.classifier(x)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
        get_context(self).add_metric("train/accuracy", acc)  # 只用于观察，不参与 total_loss

        return logits


head = ClassifierHead()
with ExtraContext(head) as ctx:
    logits = head(x, targets)
    main_loss = F.cross_entropy(logits, targets)
    metrics = ctx.get_metrics()
```

### hook

`hook` 适合保存一些 forward 之后再执行的回调。ExtraContext 只负责收集这些函数，什么时候执行由 training step 决定。

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

### 共享对象

`put/get/pop` 不只是把值传出给 training step，也可以让同一次 forward 里的不同模块共享对象。这样可以绕过 `forward()` 参数和返回值。**注意这个方法不太安全**：它依赖 key 名和模块调用顺序，适合实验时少量使用。

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

### 共享 logger

如果多个模块都需要写日志，可以在创建 `ExtraContext` 时传入一个 logger。模块里继续通过 `get_context(self).log(...)` 调用；**注意：** keras style 不会把 `log` monkey patch 到 `torch.nn.Module` 上。

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

### 包映射

| 导入路径 | 导出符号 | 说明 |
| --- | --- | --- |
| `torchextractx` | `ExtraContext`, `get_context`, `NullContext`, `configure_null_context_behavior`, `get_null_context_behavior`, `enable_keras_style_api`, `disable_keras_style_api`, `is_keras_style_api_enabled` | 应用代码的主入口。 |
| `torchextractx.context` | `ExtraContext`, `get_context` | active context 管理器和模块查找函数。 |
| `torchextractx.null_context` | `NullContext`, `configure_null_context_behavior`, `get_null_context_behavior` | 没有上下文时的空实现和缺省策略。 |
| `torchextractx.keras_style` | 副作用导入 | 导入后立即启用 Keras 风格辅助方法。 |

### `torchextractx.context.ExtraContext`

| 成员 | 类型 | 访问方式 | 说明 |
| --- | --- | --- | --- |
| `ExtraContext(root_module, logger=None)` | 类 | `with ExtraContext(model) as ctx:` | 绑定到嵌套模块的主上下文管理器。 |
| `add_loss(prefix, loss, op="sum")` | 方法 | `ctx.add_loss(...)` | 注册辅助损失张量。 |
| `add_metric(prefix, metric, op="mean")` | 方法 | `ctx.add_metric(...)` | 注册指标张量。 |
| `add_output(prefix, output)` | 方法 | `ctx.add_output(...)` | 保存中间输出并检查形状。 |
| `add_hook(prefix, hook)` | 方法 | `ctx.add_hook(...)` | 注册可调用 hook。 |
| `put(key, value)` | 方法 | `ctx.put(...)` | 存储共享状态，供模块间传递。 |
| `get(key, default=None)` | 方法 | `ctx.get(...)` | 读取共享状态。 |
| `pop(key, default=None)` | 方法 | `ctx.pop(...)` | 移除并返回共享状态。 |
| `has(key)` | 方法 | `ctx.has(...)` | 检查共享状态是否存在。 |
| `get_losses(default_op="sum")` | 方法 | `ctx.get_losses(...)` | 返回归约后的损失字典。 |
| `get_metrics(default_op="mean")` | 方法 | `ctx.get_metrics(...)` | 返回归约后的指标字典。 |
| `get_outputs()` | 方法 | `ctx.get_outputs(...)` | 返回保存的输出字典。 |
| `get_module_prefixes(module)` | 方法 | `ctx.get_module_prefixes(...)` | 返回模块在模型中的路径。 |
| `log(*args, **kwargs)` | 方法 | `ctx.log(...)` | 转发到配置好的 logger。 |
| `losses` | 属性 | `ctx.losses` | 原始 loss 存储。 |
| `hooks` | 属性 | `ctx.hooks` | 原始 hook 存储。 |

### `torchextractx.null_context.NullContext`

| 成员 | 类型 | 访问方式 | 说明 |
| --- | --- | --- | --- |
| `NullContext(module=None)` | 类 | `NullContext(module)` 或 `get_context(module)` 返回 | 没有 active `ExtraContext` 时使用的空实现。 |
| `add_loss`, `add_metric`, `add_output`, `add_hook`, `put`, `log` | 方法 | `ctx.add_loss(...)`、`ctx.put(...)` 等 | 会被忽略；可按配置选择警告、静默或抛错。 |
| `get`, `pop`, `has`, `get_losses`, `get_metrics`, `get_outputs`, `get_module_prefixes` | 方法 | `ctx.get(...)`、`ctx.get_losses()` 等 | 返回默认值或空结果。 |
| `losses`, `hooks` | 属性 | `ctx.losses`, `ctx.hooks` | 返回空存储。 |
| `__getitem__` | 方法 | `ctx[key]` | 因为没有共享状态，会抛 `KeyError`。 |

### `torchextractx.keras_style`

| 导入 / 函数 | 效果 | 备注 |
| --- | --- | --- |
| `import torchextractx.keras_style` | 立即启用 `torch.nn.Module.add_loss`、`add_metric`、`add_output`、`add_hook`。 | 这是进程级副作用导入。 |
| `from torchextractx import enable_keras_style_api` | 显式启用同样的效果。 | 适合把开关写进代码。 |
| `disable_keras_style_api()` | 恢复原始 `torch.nn.Module` 方法。 | 从 `torchextractx` 重新导出。 |
| `is_keras_style_api_enabled()` | 查看 shim 是否已启用。 | 从 `torchextractx` 重新导出。 |

**注意：** 日志请继续用 `get_context(self).log(...)`，keras style 不会把 `log` monkey patch 到 `torch.nn.Module` 上。

## ⚠️ 注意

### 线程安全

不是线程安全的，不要从多个线程同时使用。

### 嵌套上下文

不允许嵌套使用同一个模型的多个 ExtraContext，会报错：

```python
with ExtraContext(model):
    with ExtraContext(model):  # 会抛出 ValueError
        pass
```

### 内存管理

退出 `with` 块后所有数据会被清空。不要在外面访问：

```python
with ExtraContext(model) as ctx:
    output = model(x)
    losses = ctx.get_losses()  # 正常

losses = ctx.get_losses()  # 会报错
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

或者用 unittest：

```bash
python -m unittest discover tests/ -v
```

## 📋 需求

- Python ≥ 3.10
- PyTorch ≥ 2.0.0

## 📝 协议

MIT - 查看 [LICENSE](LICENSE)

## 🚢 发布流程

- 版本号以 `pyproject.toml` 里的 `[project].version` 为准。
- 发布标签必须使用 `vX.Y.Z`，并且要和该版本完全一致。
- GitHub Release 的版本说明来自 `CHANGELOG.md`。
- 推送匹配的标签后，会自动跑测试、构建制品、创建 GitHub Release，并通过 PyPI Trusted Publishing 发布到 PyPI。
- 发布到 PyPI 不需要配置 PyPI token；认证由 GitHub Actions OpenID Connect 完成。

## 🤝 贡献

欢迎提 issue 和 PR。
