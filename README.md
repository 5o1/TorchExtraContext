# TorchExtraContext
Append an extra context object to all child modules of the top-level module, and collect losses and hooks.

This method is useful when you need to pass losses from `nn.Module` in a deep level to the training step, to avoid modifying the interface of each level.

# Usage

```python
from torchextractx import ExtraContext, register_extra_loss, register_extra_hook, register_extra_object

# In sub module:
class SubModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        register_extra_loss(self, x.mean(), "localloss")

        def dosomething_hook():
            return x.mean()
        register_extra_hook(self, dosomething_hook, "This is a submodel hook")
        register_extra_object(self, x, "This is a submodel object")
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

m = TopModel()
with ExtraContext(model) as ctx:
    pred = model(batch)

    # ... training step

    losses, hooks_res, objects = ctx.release() # run all the hooks and collect results, popping all the losses and objects
    # losses, hooks_res, objects are lists of tuples:
    # (prefix, loss) for every loss
    # (prefix, hook_res) for every hook result
    # (prefix, objects) for every object registered in the context

    loss = lossfn(pred, target)
    for prefix, localoss in losses:
        print(f"Loss {prefix}: {loss.item()}")
        loss += localoss
    loss.backward()
```