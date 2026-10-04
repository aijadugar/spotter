from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.hf_spaces import router_torch_device, zero_gpu_enabled


class HfSpacesRuntimeTests(unittest.TestCase):
    def test_cpu_is_default_runtime_device(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(zero_gpu_enabled())
            self.assertEqual(router_torch_device(), "cpu")

    def test_zero_gpu_keeps_router_on_cpu_by_default(self) -> None:
        with patch.dict(os.environ, {"SPACES_ZERO_GPU": "1"}, clear=True):
            self.assertTrue(zero_gpu_enabled())
            self.assertEqual(router_torch_device(), "cpu")

    def test_explicit_router_device_overrides_default(self) -> None:
        with patch.dict(
            os.environ,
            {"SPACES_ZERO_GPU": "1", "SPOTTER_ROUTER_DEVICE": "cuda"},
            clear=True,
        ):
            self.assertEqual(router_torch_device(), "cuda")


if __name__ == "__main__":
    unittest.main()
