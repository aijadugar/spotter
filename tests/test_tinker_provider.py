from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class TinkerProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        # Clear any cached model
        import spotter.slm.providers as providers
        providers._COACH_SUMMARY_SINGLETON = None
        providers._COACH_SUMMARY_CONFIG_KEY = None

    def tearDown(self) -> None:
        # Clean up env vars
        for key in list(os.environ.keys()):
            if key.startswith("SPOTTER_"):
                os.environ.pop(key, None)
        os.environ.pop("TINKER_API_KEY", None)

    def test_tinker_provider_fallback_on_missing_key(self) -> None:
        """Test that Tinker provider falls back to deterministic when no API key."""
        os.environ["SPOTTER_COACH_SUMMARY_PROVIDER"] = "tinker"
        os.environ.pop("TINKER_API_KEY", None)

        from spotter.slm.providers import get_coach_summary_model
        model = get_coach_summary_model()

        # Should return None (fallback to deterministic)
        self.assertIsNone(model)

    def test_tinker_provider_fallback_on_import_error(self) -> None:
        """Test that Tinker provider falls back when SDK not installed."""
        os.environ["SPOTTER_COACH_SUMMARY_PROVIDER"] = "tinker"
        os.environ["TINKER_API_KEY"] = "test_key"

        # Remove tinker from modules if present
        sys.modules.pop("tinker", None)
        sys.modules.pop("tinker.types", None)

        with patch.dict(sys.modules, {"tinker": None}):
            import importlib
            import spotter.slm.providers as providers
            importlib.reload(providers)

            model = providers.get_coach_summary_model()
            self.assertIsNone(model)

    def test_tinker_provider_fallback_on_runtime_error(self) -> None:
        """Test that Tinker provider falls back on any runtime error."""
        os.environ["SPOTTER_COACH_SUMMARY_PROVIDER"] = "tinker"
        os.environ["TINKER_API_KEY"] = "test_key"

        mock_tinker = MagicMock()
        mock_tinker.ServiceClient.side_effect = RuntimeError("Connection failed")

        with patch.dict(sys.modules, {"tinker": mock_tinker, "tinker.types": MagicMock()}):
            import importlib
            import spotter.slm.providers as providers
            importlib.reload(providers)

            model = providers.get_coach_summary_model()
            self.assertIsNone(model)

    def test_tinker_provider_creates_client_when_available(self) -> None:
        """Test that Tinker provider creates client when SDK and key available."""
        os.environ["SPOTTER_COACH_SUMMARY_PROVIDER"] = "tinker"
        os.environ["TINKER_API_KEY"] = "test_key"
        os.environ["SPOTTER_COACH_SUMMARY_MODEL"] = "Qwen/Qwen3.5-4B"

        mock_tinker = MagicMock()
        mock_types = MagicMock()

        mock_service_client = MagicMock()
        mock_sampling_client = MagicMock()
        mock_tokenizer = MagicMock()

        mock_tinker.ServiceClient.return_value = mock_service_client
        mock_tinker.types = mock_types
        mock_service_client.create_sampling_client.return_value = mock_sampling_client
        mock_sampling_client.get_tokenizer.return_value = mock_tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = '{"focus": "test", "targets": [], "next_session_cues": [], "encouragement": "test", "confidence_notes": []}'

        mock_result = MagicMock()
        mock_result.sequences = [MagicMock(tokens=[4, 5, 6])]
        mock_sampling_client.sample_async = AsyncMock(return_value=mock_result)

        with patch.dict(sys.modules, {"tinker": mock_tinker, "tinker.types": mock_types}):
            import importlib
            import spotter.slm.providers as providers
            importlib.reload(providers)

            model = providers.get_coach_summary_model()
            self.assertIsNotNone(model)

            # Test generation
            generation = model.generate_summary('{"test": "prompt"}')
            self.assertEqual(generation.provider, "tinker")
            self.assertEqual(generation.model, "Qwen/Qwen3.5-4B")


if __name__ == "__main__":
    unittest.main()