# TorchExtraContext
Append an extra context object to all child modules of the top-level module, and collect losses and hooks.

This method is useful when you need to pass losses from `nn.Module` in a deep level to the training step, to avoid modifying the interface of each level.

# Usage

```python
from torchextractx import ExtraContext, register_extra_loss, register_extra_hook, get_extra_context

# In sub module:
class SubModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        register_extra_loss(self, x.mean(), "localloss")

        def dosomething_hook():
            return x.mean()
        register_extra_hook(self, dosomething_hook, "This is a submodel hook")
        if ctx := get_extra_context(self):
            ctx['local_var'] = 'This is a local variable'
        return x

# In top module:
class TopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.subm = SubModel()
    
    def forward(self, x):
        x = self.subm(x)
        return x

    
# In training step:

model = TopModel()
with ExtraContext(model) as ctx:
    pred = model(batch)

    # ... training step

    losses = ctx.losses
    hooks = ctx.hooks
    # losses, hooks, objects are lists of tuples:
    # (prefix, loss) for every loss
    # (prefix, hook) for every hook

    loss = lossfn(pred, target)
    for prefix, localoss in losses:
        print(f"Loss {prefix}: {loss.item()}")
        loss += localoss
    loss.backward()
```
