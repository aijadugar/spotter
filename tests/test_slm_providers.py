from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.slm.providers import (  # noqa: E402
    get_coach_summary_model,
    HFInferenceCoachSummaryModel,
    LocalTransformersCoachSummaryModel,
    _tokenize_local_chat,
)


class SlmProviderTests(unittest.TestCase):
    def test_local_chat_tokenization_preserves_prompt_tail_on_truncation(self) -> None:
        class _Tokenizer:
            truncation_side = "right"

            def __init__(self) -> None:
                self.seen_truncation_side = None

            def apply_chat_template(self, messages, **kwargs):
                del messages, kwargs
                self.seen_truncation_side = self.truncation_side
                return {"input_ids": [[1, 2, 3]]}

        tokenizer = _Tokenizer()

        result = _tokenize_local_chat(
            tokenizer=tokenizer,
            messages=[{"role": "user", "content": "coach prompt"}],
            max_input_tokens=8,
        )

        self.assertEqual(result, {"input_ids": [[1, 2, 3]]})
        self.assertEqual(tokenizer.seen_truncation_side, "left")
        self.assertEqual(tokenizer.truncation_side, "right")

    def test_returns_local_transformers_model_when_local_dir_is_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR": "/tmp/local-model",
                "SPOTTER_COACH_SUMMARY_BASE_MODEL": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "SPOTTER_COACH_SUMMARY_ADAPTER_ID": "spotter/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, LocalTransformersCoachSummaryModel)

    def test_hf_space_local_dir_uses_local_transformers(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPACE_ID": "owner/space",
                "SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR": "/tmp/local-model",
                "SPOTTER_COACH_SUMMARY_MODEL": "aijadugar/spotter-coach-summary1",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, LocalTransformersCoachSummaryModel)

    def test_hf_space_local_transformers_provider_uses_local_transformers(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPACE_ID": "owner/space",
                "SPOTTER_COACH_SUMMARY_PROVIDER": "local_transformers",
                "SPOTTER_COACH_SUMMARY_MODEL": "aijadugar/spotter-coach-summary1",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, LocalTransformersCoachSummaryModel)

    def test_zero_gpu_defaults_to_local_transformers_without_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPACES_ZERO_GPU": "1",
                "SPOTTER_COACH_SUMMARY_MODEL": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, LocalTransformersCoachSummaryModel)

    def test_local_transformers_uses_configured_max_input_tokens(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPOTTER_COACH_SUMMARY_PROVIDER": "local_transformers",
                "SPOTTER_COACH_SUMMARY_MODEL": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "SPOTTER_COACH_SUMMARY_MAX_INPUT_TOKENS": "512",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, LocalTransformersCoachSummaryModel)
        self.assertEqual(model.max_input_tokens, 512)

    def test_regular_hf_space_defaults_to_remote_inference(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPACE_ID": "owner/space",
                "SPOTTER_COACH_SUMMARY_MODEL": "aijadugar/spotter-coach-summary1",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)

    def test_returns_hf_inference_model_when_remote_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPOTTER_COACH_SUMMARY_MODEL": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)

    def test_remote_model_uses_runtime_model_not_adapter_repo(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPOTTER_COACH_SUMMARY_MODEL": "owner/custom-coach-summary",
                "SPOTTER_COACH_SUMMARY_ADAPTER_ID": "spotter/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)
        self.assertEqual(model.model, "owner/custom-coach-summary")

    def test_remote_model_falls_back_to_default_when_only_adapter_repo_is_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SPOTTER_COACH_SUMMARY_ADAPTER_ID": "spotter/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)
        self.assertEqual(model.model, "aijadugar/spotter-coach-summary1")

    def test_hf_inference_falls_back_to_text_generation_for_non_chat_model(self) -> None:
        class _TextGenerationClient:
            def __init__(self) -> None:
                self.text_generation_kwargs = None

            def chat_completion(self, **_kwargs):
                raise RuntimeError("not a chat model")

            def text_generation(self, prompt: str, **kwargs):
                self.text_generation_kwargs = {"prompt": prompt, **kwargs}
                return '{"summary":"ok"}'

        client = _TextGenerationClient()
        model = HFInferenceCoachSummaryModel(
            model="aijadugar/spotter-coach-summary1",
            max_tokens=123,
            temperature=0.2,
        )
        model._client = client

        generation = model.generate_summary("coach prompt")

        self.assertEqual(generation.provider, "hf_inference")
        self.assertEqual(generation.model, "aijadugar/spotter-coach-summary1")
        self.assertEqual(generation.text, '{"summary":"ok"}')
        self.assertEqual(
            client.text_generation_kwargs,
            {
                "prompt": "coach prompt",
                "model": "aijadugar/spotter-coach-summary1",
                "max_new_tokens": 123,
                "temperature": 0.2,
                "return_full_text": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
