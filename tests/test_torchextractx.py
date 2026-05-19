"""
Unit tests for ExtraContext functionality.
"""

import unittest
import torch
import torch.nn as nn
from torchextractx import ExtraContext, add_loss, add_metric


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
        with ExtraContext(self.model) as ctx:
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
            add_loss(self.model.fc1, "test_loss", loss_tensor)
            
            losses = ctx.get_losses()
            self.assertIn("test_loss", losses)
            self.assertEqual(losses["test_loss"], loss_tensor)
    
    def test_loss_reduction_sum(self):
        """Test loss reduction with 'sum' operation."""
        with ExtraContext(self.model) as ctx:
            loss1 = torch.tensor(1.0)
            loss2 = torch.tensor(2.0)
            
            add_loss(self.model.fc1, "test_loss", loss1, op="sum")
            add_loss(self.model.fc2, "test_loss", loss2, op="sum")
            
            losses = ctx.get_losses()
            expected = torch.sum(torch.stack([loss1, loss2]))
            self.assertEqual(losses["test_loss"], expected)
    
    def test_loss_reduction_mean(self):
        """Test loss reduction with 'mean' operation."""
        with ExtraContext(self.model) as ctx:
            loss1 = torch.tensor(1.0)
            loss2 = torch.tensor(3.0)
            
            add_loss(self.model.fc1, "test_loss", loss1, op="mean")
            add_loss(self.model.fc2, "test_loss", loss2, op="mean")
            
            losses = ctx.get_losses()
            expected = torch.mean(torch.stack([loss1, loss2]))
            self.assertEqual(losses["test_loss"], expected)
    
    def test_add_metric(self):
        """Test adding metrics to context."""
        with ExtraContext(self.model) as ctx:
            metric_tensor = torch.tensor(0.95)
            add_metric(self.model.fc1, "accuracy", metric_tensor)
            
            metrics = ctx.get_metrics()
            self.assertIn("accuracy", metrics)
            self.assertEqual(metrics["accuracy"], metric_tensor)
    
    def test_add_output(self):
        """Test adding outputs to context."""
        with ExtraContext(self.model) as ctx:
            output = torch.randn(4, 5)
            ctx.add_output("feature_map", output)
            
            outputs = ctx.get_outputs()
            self.assertIn("feature_map", outputs)
    
    def test_get_module_prefixes(self):
        """Test retrieving module prefixes."""
        with ExtraContext(self.model) as ctx:
            prefixes = ctx.get_module_prefixes(self.model.fc1)
            self.assertIn("fc1", prefixes)
    
    def test_nested_context_error(self):
        """Test that nested ExtraContext raises error."""
        with ExtraContext(self.model) as ctx1:
            with self.assertRaises(ValueError):
                with ExtraContext(self.model) as ctx2:
                    pass
    
    def test_concurrent_context_error(self):
        """Test that concurrent ExtraContext raises error."""
        ctx1 = ExtraContext(self.model)
        ctx2 = ExtraContext(self.model)
        
        with ctx1:
            with self.assertRaises(ValueError):
                ctx2.__enter__()
            ctx2.__exit__(None, None, None)
    
    def test_context_cleanup_on_error(self):
        """Test that context cleans up properly even when error occurs."""
        try:
            with ExtraContext(self.model) as ctx:
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
            pass
        
        # Try to access after exit
        with self.assertRaises(ValueError):
            ctx.get_losses()


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
