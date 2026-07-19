# Changelog

## 0.2.0 - 2026-07-19

Compared with the May 2026 code state, this release adds:

- Formal versioned release automation with GitHub Actions CI, tag-based release workflow, and PyPI Trusted Publishing.
- A split runtime layout for `ExtraContext`, `NullContext`, and opt-in Keras-style helpers.
- Loss, metric, output, hook, and shared-state APIs, including `put/get/pop`, `get_metrics()`, and `get_outputs()`.
- Configurable behavior when `get_context(self)` is called outside an active context.
- Keras-style `torch.nn.Module` helpers via `import torchextractx.keras_style` or `enable_keras_style_api()`.
- Expanded tests, docs, and Lightning usage notes.
