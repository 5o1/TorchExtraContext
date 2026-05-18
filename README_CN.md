# TorchExtraContext

[English](README.md) | [中文](README_CN.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个 PyTorch 工具库，用于从嵌套的模块中收集损失、指标和输出。

## 为什么用这个库？

做深度学习时经常需要在网络中间添加损失，比如蒸馏损失或辅助分类器的损失。常见的做法是：

1. 把损失层层向上传递，这需要改每个模块的 `forward()` 签名
2. 用全局变量或回调函数，代码变得很复杂

这个库让你直接在模块里注册损失，不改任何接口：

```python
def forward(self, x):
    x = self.layer1(x)
    register_extra_loss(self, "loss_a", compute_loss(x))
    return x
```

然后一行代码收集所有损失：

```python
with ExtraContext(model) as ctx:
    output = model(x)
    all_losses = ctx.get_losses()  # 完成
```

## 🚀 安装

### 从源代码安装
```bash
pip install -e .
```

### 使用开发工具安装
```bash
pip install -e ".[dev]"  # 包含 pytest, black, mypy
```

## 📖 快速开始

### 基础用法

```python
import torch
import torch.nn as nn
from torchextractx import ExtraContext, register_extra_loss, register_extra_metric

# 定义具有中间损失的模型
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        
        # 注册辅助损失（无需修改接口）
        aux_loss = x.mean()
        register_extra_loss(self, "auxiliary_loss", aux_loss)
        
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

# 训练循环
model = Classifier()
optimizer = torch.optim.Adam(model.parameters())

# 使用 ExtraContext：一行代码启用所有功能
with ExtraContext(model) as ctx:
    x = torch.randn(32, 10)
    logits = model(x)
    
    # 主损失
    main_loss = torch.nn.functional.cross_entropy(logits, targets)
    
    # 收集所有深层模块注册的损失
    aux_losses = ctx.get_losses()  # {'auxiliary_loss': tensor(...), ...}
    
    # 总损失 = 主损失 + 权重化的辅助损失
    total_loss = main_loss
    for name, loss_val in aux_losses.items():
        print(f"辅助 {name}: {loss_val:.4f}")
        total_loss = total_loss + 0.1 * loss_val  # 权重可调
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## 🔧 API 参考

### `ExtraContext`

用于收集辅助信息的主要上下文管理器。

```python
ctx = ExtraContext(root_module, logger=None)
```

**参数：**
- `root_module` (nn.Module): 要扫描的根模块
- `logger` (Callable, 可选): 日志函数

**方法：**

#### `add_loss(prefix, loss, op="sum")`

注册一个损失。
- `prefix`: 损失的名字
- `loss`: 损失值（张量）
- `op`: 合并方式，默认 `"sum"` | 可选 `"mean"` `"max"` `"min"`

#### `add_metric(prefix, metric, op="mean")`

注册指标（如准确率、F1），默认用平均值合并。

#### `add_output(prefix, output)`

保存中间输出供事后分析，会检查输出形状一致性。

#### `add_hook(prefix, hook)`
注册可调用钩子。

#### `get_losses(default_op="sum")`
获取所有收集的损失作为字典。
```python
losses = ctx.get_losses()
# 返回: {'loss_name': tensor(...), ...}
```

#### `get_metrics(default_op="mean")`

获取所有注册过的指标，返回字典。

#### `get_outputs()`

获取保存的输出张量。

#### `get_module_prefixes(module)`

查询某个模块在模型中的路径名。调试时有用。

### 辅助函数

#### `register_extra_loss(module, prefix, loss_term, op="sum")`

在模块里注册损失：

```python
def forward(self, x):
    x = self.process(x)
    register_extra_loss(self, "aux_loss", x.sum())
    return x
```

#### `register_extra_metric(module, prefix, metric_term, op="mean")`

注册指标。

#### `register_extra_output(module, prefix, output)`

注册输出。

#### `register_extra_hook(module, prefix, hook)`

注册钩子。

#### `get_extra_context(module)`

在模块里获取上下文对象，用于保存调试数据：

```python
if ctx := get_extra_context(self):
    ctx['debug_data'] = some_value
```

#### `log_extra(module, *args, **kwargs)`

通过上下文记录调试信息。

## 📚 高级用法

### 多任务学习

用不同权重组合多个损失：

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

### 自定义合并方式

不同的损失用不同的合并方式：

```python
with ExtraContext(model) as ctx:
    output = model(x)
    register_extra_loss(model.layer1, "loss_a", tensor_a, op="mean")
    register_extra_loss(model.layer2, "loss_b", tensor_b, op="max")
    losses = ctx.get_losses()
```

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

## 🤝 贡献

欢迎提 issue 和 PR。
