"""Small offline Torch regression tests, not model-accuracy or latency evidence."""

import unittest
from types import SimpleNamespace

import torch

from benchmark_single import accelerate_cpu


class QuantizationRuntimeTests(unittest.TestCase):
    def test_quantized_encoder_retains_a_working_float_decision_head(self) -> None:
        torch.set_num_threads(1)
        encoder = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU()).eval()
        head = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(8, 2, batch_first=True, norm_first=True),
            1,
            enable_nested_tensor=False,
        ).eval()
        model = torch.nn.Module()
        model.encoder = encoder
        model.head = head
        agent = SimpleNamespace(device=torch.device("cpu"), model=model)
        count, backend = accelerate_cpu(agent, "int8")
        self.assertEqual(count, 1)
        self.assertIn(backend, ("x86", "fbgemm", "qnnpack"))
        self.assertIsInstance(model.head.layers[0].linear1, torch.nn.Linear)
        self.assertIsInstance(model.head.layers[0].linear1.weight, torch.Tensor)
        with torch.no_grad():
            result = model.head(model.encoder(torch.zeros(1, 2, 8)))
        self.assertEqual(tuple(result.shape), (1, 2, 8))
        self.assertTrue(torch.isfinite(result).all().item())


if __name__ == "__main__":
    unittest.main()
