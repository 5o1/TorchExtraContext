"""Unit tests for ExtraContext functionality."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-many-public-methods

import importlib
import unittest
import warnings
from importlib.metadata import version as package_version

import torch
from torch import nn

import torchextractx
from torchextractx import (
    ExtraContext,
    NullContext,
    configure_null_context_behavior,
    disable_keras_style_api,
    enable_keras_style_api,
    get_context,
    get_null_context_behavior,
    is_keras_style_api_enabled,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestExtraContext(unittest.TestCase):
    """Test cases for ExtraContext."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.model.train()

    def test_context_binding(self):
        """Test that ExtraContext binds to modules correctly."""
        with ExtraContext(self.model):
            # Check that extra_context is set on modules
            self.assertTrue(hasattr(self.model, "extra_context"))
            self.assertTrue(hasattr(self.model.fc1, "extra_context"))
            self.assertTrue(hasattr(self.model.fc2, "extra_context"))

        # Check that extra_context is removed after exit
        self.assertFalse(hasattr(self.model, "extra_context"))
        self.assertFalse(hasattr(self.model.fc1, "extra_context"))

    def test_add_loss(self):
        """Test adding losses to context."""
        with ExtraContext(self.model) as ctx:
            loss_tensor = torch.tensor(1.5)
            get_context(self.model.fc1).add_loss("test_loss", loss_tensor)

            losses = ctx.get_losses()
            self.assertIn("test_loss", losses)
            self.assertEqual(losses["test_loss"], loss_tensor)

    def test_loss_reduction_sum(self):
        """Test loss reduction with 'sum' operation."""
        with ExtraContext(self.model) as ctx:
            loss1 = torch.tensor(1.0)
            loss2 = torch.tensor(2.0)

            get_context(self.model.fc1).add_loss("test_loss", loss1, op="sum")
            get_context(self.model.fc2).add_loss("test_loss", loss2, op="sum")

            losses = ctx.get_losses()
            expected = torch.sum(torch.stack([loss1, loss2]))
            self.assertEqual(losses["test_loss"], expected)

    def test_loss_reduction_mean(self):
        """Test loss reduction with 'mean' operation."""
        with ExtraContext(self.model) as ctx:
            loss1 = torch.tensor(1.0)
            loss2 = torch.tensor(3.0)

            get_context(self.model.fc1).add_loss("test_loss", loss1, op="mean")
            get_context(self.model.fc2).add_loss("test_loss", loss2, op="mean")

            losses = ctx.get_losses()
            expected = torch.mean(torch.stack([loss1, loss2]))
            self.assertEqual(losses["test_loss"], expected)

    def test_loss_reduction_max_and_min(self):
        """Test loss reduction with 'max' and 'min' operations."""
        cases = {
            "max": torch.tensor(3.0),
            "min": torch.tensor(1.0),
        }

        for op, expected in cases.items():
            with self.subTest(op=op):
                with ExtraContext(self.model) as ctx:
                    get_context(self.model.fc1).add_loss("test_loss", torch.tensor(1.0), op=op)
                    get_context(self.model.fc2).add_loss("test_loss", torch.tensor(3.0), op=op)

                    losses = ctx.get_losses()
                    torch.testing.assert_close(losses["test_loss"], expected)

    def test_invalid_loss_reduction_raises_error(self):
        """Test that unsupported reduction operations raise an error."""
        with ExtraContext(self.model) as ctx:
            ctx.add_loss("bad_loss", torch.tensor(1.0), op="median")
            ctx.add_loss("bad_loss", torch.tensor(2.0), op="median")

            with self.assertRaisesRegex(ValueError, "Unsupported operation"):
                ctx.get_losses()

    def test_add_metric(self):
        """Test adding metrics to context."""
        with ExtraContext(self.model) as ctx:
            metric_tensor = torch.tensor(0.95)
            get_context(self.model.fc1).add_metric("accuracy", metric_tensor)

            metrics = ctx.get_metrics()
            self.assertIn("accuracy", metrics)
            self.assertEqual(metrics["accuracy"], metric_tensor)

    def test_metric_int_suffix_casts_and_renames_metric(self):
        """Test that [int] metrics are reduced, cast to int, and renamed."""
        with ExtraContext(self.model) as ctx:
            get_context(self.model.fc1).add_metric("sample_count[int]", torch.tensor(1.0), op="sum")
            get_context(self.model.fc2).add_metric("sample_count[int]", torch.tensor(2.0), op="sum")

            metrics = ctx.get_metrics()
            self.assertIn("sample_count", metrics)
            self.assertNotIn("sample_count[int]", metrics)
            self.assertEqual(metrics["sample_count"].dtype, torch.int32)
            torch.testing.assert_close(metrics["sample_count"], torch.tensor(3, dtype=torch.int32))

    def test_add_output(self):
        """Test adding outputs to context."""
        with ExtraContext(self.model) as ctx:
            output = torch.randn(4, 5)
            ctx.add_output("feature_map", output)

            outputs = ctx.get_outputs()
            self.assertIn("feature_map", outputs)
            torch.testing.assert_close(outputs["feature_map"], output.unsqueeze(0))

    def test_multiple_outputs_are_stacked(self):
        """Test that multiple outputs under the same prefix are stacked."""
        with ExtraContext(self.model) as ctx:
            output1 = torch.randn(4, 5)
            output2 = torch.randn(4, 5)

            get_context(self.model.fc1).add_output("feature_map", output1)
            get_context(self.model.fc2).add_output("feature_map", output2)

            outputs = ctx.get_outputs()
            torch.testing.assert_close(outputs["feature_map"], torch.stack([output1, output2]))

    def test_output_shape_mismatch_raises_error(self):
        """Test that output tensors with mismatched shapes are rejected."""
        with ExtraContext(self.model) as ctx:
            ctx.add_output("feature_map", torch.randn(4, 5))

            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                ctx.add_output("feature_map", torch.randn(4, 6))

    def test_add_hook(self):
        """Test adding hooks to context."""
        with ExtraContext(self.model) as ctx:
            def hook(_module, _inputs, output):
                return output

            get_context(self.model.fc1).add_hook("activation_hook", hook)

            self.assertIn("activation_hook", ctx.hooks)
            self.assertEqual(ctx.hooks["activation_hook"], [hook])

    def test_context_object_storage(self):
        """Test dictionary-style context object storage."""
        with ExtraContext(self.model) as ctx:
            ctx.put("debug_value", {"step": 1})

            self.assertTrue(ctx.has("debug_value"))
            self.assertEqual(ctx.get("debug_value"), {"step": 1})
            self.assertEqual(ctx["debug_value"], {"step": 1})
            self.assertEqual(ctx.pop("debug_value"), {"step": 1})
            self.assertFalse(ctx.has("debug_value"))

    def test_get_context_returns_active_context(self):
        """Test retrieving the active context from a module."""
        with ExtraContext(self.model) as ctx:
            self.assertIs(get_context(self.model.fc1), ctx)

    def test_log_uses_configured_logger(self):
        """Test logging through the configured context logger."""
        calls = []

        def logger(*args, **kwargs):
            calls.append((args, kwargs))
            return "logged"

        with ExtraContext(self.model, logger=logger):
            result = get_context(self.model.fc1).log("message", step=3)

        self.assertEqual(result, "logged")
        self.assertEqual(calls, [(("message",), {"step": 3})])

    def test_log_without_logger_warns(self):
        """Test that logging without a configured logger warns and returns None."""
        with ExtraContext(self.model):
            with self.assertWarnsRegex(UserWarning, "No logger is set"):
                result = get_context(self.model.fc1).log("message")

        self.assertIsNone(result)

    def test_null_context_warns_once_on_writes_in_training(self):
        """Test that NullContext warns once per write operation in training mode."""
        self.model.train()
        ctx = get_context(self.model.fc1)
        self.assertIsInstance(ctx, NullContext)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx.add_loss("ignored_loss", torch.tensor(1.0))
            ctx.add_loss("ignored_loss", torch.tensor(2.0))
            ctx.add_metric("ignored_metric", torch.tensor(1.0))
            ctx.add_output("ignored_output", torch.randn(2, 2))
            ctx.add_hook("ignored_hook", lambda *args: None)
            ctx.put("ignored_value", 1)
            ctx.log("message")

        self.assertEqual(len(caught), 6)
        self.assertTrue(all("No active ExtraContext" in str(item.message) for item in caught))
        self.assertEqual(ctx.get_losses(), {})
        self.assertEqual(ctx.get_metrics(), {})
        self.assertEqual(ctx.get_outputs(), {})
        self.assertFalse(ctx.has("ignored_value"))
        self.assertFalse(hasattr(self.model.fc1, "extra_context"))

    def test_null_context_raise_mode(self):
        """Test that NullContext can be configured to raise."""
        previous = configure_null_context_behavior("raise")
        try:
            self.assertEqual(get_null_context_behavior(), "raise")
            ctx = get_context(self.model.fc1)
            with self.assertRaisesRegex(RuntimeError, "No active ExtraContext"):
                ctx.add_loss("ignored_loss", torch.tensor(1.0))
        finally:
            configure_null_context_behavior(previous)

    def test_null_context_silent_mode(self):
        """Test that NullContext can be configured to stay silent."""
        previous = configure_null_context_behavior("silent")
        try:
            self.assertEqual(get_null_context_behavior(), "silent")
            ctx = get_context(self.model.fc1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ctx.add_loss("ignored_loss", torch.tensor(1.0))
                ctx.put("ignored_value", 1)
            self.assertEqual(len(caught), 0)
        finally:
            configure_null_context_behavior(previous)

    def test_get_module_prefixes(self):
        """Test retrieving module prefixes."""
        with ExtraContext(self.model) as ctx:
            prefixes = ctx.get_module_prefixes(self.model.fc1)
            self.assertIn("fc1", prefixes)

    def test_nested_context_error(self):
        """Test that nested ExtraContext raises error."""
        with ExtraContext(self.model):
            with self.assertRaises(ValueError):
                with ExtraContext(self.model):
                    self.fail("Nested context unexpectedly entered.")

    def test_concurrent_context_error(self):
        """Test that concurrent ExtraContext raises error."""
        ctx1 = ExtraContext(self.model)
        ctx2 = ExtraContext(self.model)

        with ctx1:
            with self.assertRaises(ValueError):
                with ctx2:
                    self.fail("Concurrent context unexpectedly entered.")

    def test_failed_context_entry_does_not_leave_partial_binding(self):
        """Test that failed context entry does not leave partial module bindings."""
        occupied_context = object()
        setattr(self.model.fc2, "extra_context", occupied_context)

        with self.assertRaises(ValueError):
            with ExtraContext(self.model):
                self.fail("Context unexpectedly entered.")

        self.assertFalse(hasattr(self.model, "extra_context"))
        self.assertFalse(hasattr(self.model.fc1, "extra_context"))
        self.assertIs(self.model.fc2.extra_context, occupied_context)

    def test_context_cleanup_on_error(self):
        """Test that context cleans up properly even when error occurs."""
        try:
            with ExtraContext(self.model):
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        # Check that context was cleaned up
        self.assertFalse(hasattr(self.model, "extra_context"))
        self.assertFalse(hasattr(self.model.fc1, "extra_context"))

    def test_access_after_exit_raises_error(self):
        """Test that accessing context after exit raises error."""
        ctx = ExtraContext(self.model)
        with ctx:
            ctx["debug_value"] = 1

        with self.assertRaises(ValueError):
            ctx.get_losses()
        with self.assertRaises(ValueError):
            ctx.get_metrics()
        with self.assertRaises(ValueError):
            ctx.get_outputs()
        with self.assertRaises(ValueError):
            _ = ctx.hooks
        with self.assertRaises(ValueError):
            _ = ctx["debug_value"]
        with self.assertRaises(ValueError):
            ctx["debug_value"] = 2
        with self.assertRaises(ValueError):
            ctx.put("debug_value", 2)
        with self.assertRaises(ValueError):
            ctx.get("debug_value")
        with self.assertRaises(ValueError):
            ctx.pop("debug_value")
        with self.assertRaises(ValueError):
            ctx.has("debug_value")

    def test_package_version_matches_metadata(self):
        """Test that package version is sourced from installed metadata."""
        self.assertEqual(torchextractx.__version__, package_version("torchextractx"))

    def test_forward_backward_integration(self):
        """Test a real forward/backward pass that records extra context data."""
        enable_keras_style_api()
        self.assertTrue(is_keras_style_api_enabled())

        try:
            class InstrumentedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Linear(10, 5)
                    self.classifier = nn.Linear(5, 2)

                def forward(self, x):
                    hidden = torch.relu(self.features(x))
                    self.features.add_loss("activation_loss", hidden.pow(2).mean(), op="mean")
                    self.features.add_metric("positive_rate", (hidden > 0).float().mean())
                    self.features.add_output("hidden", hidden)
                    return self.classifier(hidden)

            model = InstrumentedModel()
            model.train()
            inputs = torch.randn(4, 10)
            targets = torch.tensor([0, 1, 0, 1])

            with ExtraContext(model) as ctx:
                logits = model(inputs)
                primary_loss = nn.functional.cross_entropy(logits, targets)
                total_loss = primary_loss + ctx.get_losses()["activation_loss"]
                total_loss.backward()

                self.assertIn("positive_rate", ctx.get_metrics())
                self.assertEqual(ctx.get_outputs()["hidden"].shape, torch.Size([1, 4, 5]))

            self.assertIsNotNone(model.features.weight.grad)
            self.assertIsNotNone(model.classifier.weight.grad)
        finally:
            disable_keras_style_api()
            self.assertFalse(is_keras_style_api_enabled())

    def test_keras_style_side_effect_import_enables_api(self):
        """Test that importing torchextractx.keras_style enables the shim."""
        disable_keras_style_api()
        self.assertFalse(is_keras_style_api_enabled())

        keras_style = importlib.import_module("torchextractx.keras_style")
        importlib.reload(keras_style)
        self.assertTrue(is_keras_style_api_enabled())
        self.assertTrue(hasattr(nn.Module, "add_loss"))

        disable_keras_style_api()
        self.assertFalse(is_keras_style_api_enabled())

    def test_keras_style_enable_rejects_existing_module_method(self):
        """Test that Keras-style shim does not overwrite existing nn.Module methods."""
        disable_keras_style_api()

        def sentinel(_self, *args, **kwargs):
            _ = args, kwargs

        setattr(nn.Module, "add_loss", sentinel)
        try:
            with self.assertRaisesRegex(RuntimeError, "already defines: add_loss"):
                enable_keras_style_api()
            self.assertIs(nn.Module.add_loss, sentinel)
        finally:
            delattr(nn.Module, "add_loss")
            disable_keras_style_api()


class TestSharedSubmodules(unittest.TestCase):
    """Test cases for shared submodule handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.shared_layer = nn.Linear(5, 3)

        class ModelWithSharedLayer(nn.Module):
            def __init__(self, shared_layer):
                super().__init__()
                self.layer1 = shared_layer
                self.layer2 = shared_layer
                self.fc = nn.Linear(3, 2)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.fc(x)
                return x

        self.model = ModelWithSharedLayer(self.shared_layer)
        self.model.train()

    def test_shared_layer_prefixes(self):
        """Test that shared layers are correctly identified.

        Note: PyTorch's named_modules() automatically deduplicates module objects,
        so even if layer1 and layer2 point to the same object, it only appears once
        in the iteration. ExtraContext correctly handles this deduplication.
        """
        with ExtraContext(self.model) as ctx:
            prefixes = ctx.get_module_prefixes(self.shared_layer)
            # Should have at least one prefix (PyTorch deduplicates in named_modules)
            self.assertGreaterEqual(len(prefixes), 1)
            # Verify the binding was successful
            self.assertTrue(hasattr(self.shared_layer, "extra_context"))


if __name__ == "__main__":
    unittest.main()
