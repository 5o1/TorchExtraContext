# TorchExtraContext
Append an extra context object to all child modules of the top-level module, and collect losses and hooks. This method is useful when you need to pass losses from `nn.Module` in a deep level to the training step. You can avoid modifying the model interface of each level.
