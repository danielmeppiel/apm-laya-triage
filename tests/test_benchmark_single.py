"""Check mode selection without importing or simulating model execution."""

import unittest
from types import SimpleNamespace

from cpu_runtime import accelerate_cpu


class SingleIssueConfigurationTests(unittest.TestCase):
    def test_fp32_is_an_explicit_no_conversion_mode(self) -> None:
        agent = SimpleNamespace(device=SimpleNamespace(type="cpu"))
        self.assertEqual(accelerate_cpu(agent, "fp32"), (0, None))

    def test_wrong_device_and_unknown_precision_fail(self) -> None:
        with self.assertRaises(ValueError):
            accelerate_cpu(SimpleNamespace(device=SimpleNamespace(type="mps")), "fp32")
        with self.assertRaises(ValueError):
            accelerate_cpu(
                SimpleNamespace(device=SimpleNamespace(type="cpu")), "unknown"
            )


if __name__ == "__main__":
    unittest.main()
