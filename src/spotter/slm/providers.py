from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol
import urllib.error
import urllib.request

from spotter.env import env_truthy, load_local_env
from spotter.hf_spaces import default_spaces_gpu_duration, spaces_gpu, zero_gpu_enabled


PROVIDER_ENV = "SPOTTER_COACH_SUMMARY_PROVIDER"
MODEL_ENV = "SPOTTER_COACH_SUMMARY_MODEL"
BASE_MODEL_ENV = "SPOTTER_COACH_SUMMARY_BASE_MODEL"
LOCAL_MODEL_DIR_ENV = "SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR"
ADAPTER_ENV = "SPOTTER_COACH_SUMMARY_ADAPTER_ID"
DISABLE_REMOTE_ENV = "SPOTTER_COACH_SUMMARY_DISABLE_REMOTE"
MAX_TOKENS_ENV = "SPOTTER_COACH_SUMMARY_MAX_TOKENS"
MAX_INPUT_TOKENS_ENV = "SPOTTER_COACH_SUMMARY_MAX_INPUT_TOKENS"
TEMPERATURE_ENV = "SPOTTER_COACH_SUMMARY_TEMPERATURE"
HF_TOKEN_ENV = "HF_TOKEN"
LLAMA_CPP_BASE_URL_ENV = "SPOTTER_LLAMA_CPP_BASE_URL"
LLAMA_CPP_TIMEOUT_ENV = "SPOTTER_LLAMA_CPP_TIMEOUT"

DEFAULT_PROVIDER = "hf_inference"
DEFAULT_MODEL = "build-small-hackathon/pozify-coach-summary1"
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MAX_INPUT_TOKENS = 2048
NEMOTRON_NAIVE_MAX_INPUT_TOKENS = 1024
LOCAL_TRANSFORMERS_PROVIDER = "local_transformers"
LOCAL_TRANSFORMERS_ALIASES = {LOCAL_TRANSFORMERS_PROVIDER, "local", "transformers"}
LLAMA_CPP_PROVIDER = "llama_cpp"
LLAMA_CPP_ALIASES = {
    LLAMA_CPP_PROVIDER,
    "llamacpp",
    "llama-cpp",
    "llama_server",
    "llama-server",
}

_LOCAL_TRANSFORMERS_CACHE: dict[str, tuple[Any, Any]] = {}


@dataclass(frozen=True)
class CoachSummaryGeneration:
    text: str
    provider: str
    model: str


class CoachSummaryModel(Protocol):
    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        ...


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _configured_provider() -> str:
    configured_provider = os.getenv(PROVIDER_ENV)
    if configured_provider:
        return configured_provider.strip().lower()
    if zero_gpu_enabled():
        return LOCAL_TRANSFORMERS_PROVIDER
    return DEFAULT_PROVIDER


class HFInferenceCoachSummaryModel:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 700,
        temperature: float = 0.1,
        token: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.token = token
        self._client = None

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - dependency declared in project
            raise RuntimeError("huggingface_hub is required for Hugging Face inference") from exc
        self._client = InferenceClient(api_key=self.token)
        return self._client

    def _generate_with_chat_completion(
        self,
        client: Any,
        prompt: str,
    ) -> CoachSummaryGeneration:
        response = client.chat_completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("Hugging Face inference returned no choices")
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Hugging Face inference returned an empty message")
        return CoachSummaryGeneration(
            text=text,
            provider="hf_inference",
            model=self.model,
        )

    def _generate_with_text_generation(
        self,
        client: Any,
        prompt: str,
    ) -> CoachSummaryGeneration:
        text = client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            return_full_text=False,
        )
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Hugging Face text generation returned empty output")
        return CoachSummaryGeneration(
            text=text,
            provider="hf_inference",
            model=self.model,
        )

    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        client = self._client_instance()
        chat_error: Exception | None = None
        if hasattr(client, "chat_completion"):
            try:
                return self._generate_with_chat_completion(client, prompt)
            except Exception as exc:
                chat_error = exc

        if hasattr(client, "text_generation"):
            try:
                return self._generate_with_text_generation(client, prompt)
            except Exception as exc:
                if chat_error is None:
                    raise
                raise RuntimeError(
                    "Hugging Face chat completion and text generation both failed: "
                    f"chat={chat_error}; text_generation={exc}"
                ) from exc

        if chat_error is not None:
            raise RuntimeError(
                f"Hugging Face chat completion failed and text generation is unavailable: {chat_error}"
            ) from chat_error

        raise RuntimeError("No supported Hugging Face inference method is available")


def _gpu_duration(*_args: object, **_kwargs: object) -> int:
    return default_spaces_gpu_duration()


def _load_local_transformers_backend(model: str, token: str | None) -> tuple[Any, Any]:
    cache_key = f"{model}:{token or ''}"
    cached = _LOCAL_TRANSFORMERS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency declared in project
        raise RuntimeError(
            "transformers and torch are required for local coach summary inference"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)
    language_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        dtype="auto",
        token=token,
    )
    language_model.eval()

    cache_entry = (tokenizer, language_model)
    _LOCAL_TRANSFORMERS_CACHE[cache_key] = cache_entry
    return cache_entry


def _local_transformers_device(language_model: Any) -> Any:
    device = getattr(language_model, "device", None)
    if device is not None:
        return device

    try:
        return next(language_model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Local coach summary model has no parameters") from exc


def _move_model_inputs_to_device(model_inputs: Any, device: Any) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _nemotron_fast_kernels_available(language_model: Any) -> bool:
    model_type = getattr(getattr(language_model, "config", None), "model_type", None)
    if model_type != "nemotron_h":
        return True

    try:
        from transformers.models.nemotron_h import modeling_nemotron_h
    except Exception:
        return False

    required_kernel_names = (
        "selective_state_update",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    return all(getattr(modeling_nemotron_h, name, None) is not None for name in required_kernel_names)


def _tokenize_local_chat(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_input_tokens: int,
) -> Any:
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    if original_truncation_side is not None:
        tokenizer.truncation_side = "left"
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
                "truncation": True,
                "max_length": max_input_tokens,
            }
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_kwargs,
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, **template_kwargs)

        text = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    finally:
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side


@spaces_gpu(duration=_gpu_duration)
def _generate_local_transformers_summary(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    max_input_tokens: int,
    temperature: float,
    token: str | None,
) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency declared in project
        raise RuntimeError("torch is required for local coach summary inference") from exc

    tokenizer, language_model = _load_local_transformers_backend(model, token)
    if not _nemotron_fast_kernels_available(language_model):
        max_input_tokens = min(max_input_tokens, NEMOTRON_NAIVE_MAX_INPUT_TOKENS)

    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one valid JSON object. Start with `{` and end with `}`. "
                "Do not include reasoning, markdown, prose, or code fences."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    inputs = _tokenize_local_chat(
        tokenizer=tokenizer,
        messages=messages,
        max_input_tokens=max_input_tokens,
    )
    model_inputs = _move_model_inputs_to_device(
        inputs,
        _local_transformers_device(language_model),
    )
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "top_p": 1.0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = language_model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated = generated[:, prompt_length:]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Local transformers inference returned empty output")
    return text


class LocalTransformersCoachSummaryModel:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 700,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        temperature: float = 0.1,
        token: str | None = None,
        base_model: str | None = None,
        adapter_id: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self.token = token
        self.base_model = base_model
        self.adapter_id = adapter_id

    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        text = _generate_local_transformers_summary(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            max_input_tokens=self.max_input_tokens,
            temperature=self.temperature,
            token=self.token,
        )
        model_name = self.adapter_id or self.model
        if self.base_model:
            model_name = f"{model_name} (base: {self.base_model})"
        return CoachSummaryGeneration(
            text=text,
            provider=LOCAL_TRANSFORMERS_PROVIDER,
            model=model_name,
        )


class LlamaCppServerCoachSummaryModel:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_LLAMA_CPP_BASE_URL,
        max_tokens: int = 700,
        temperature: float = 0.1,
        timeout_sec: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw_response = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama.cpp server is unavailable: {exc.reason}") from exc

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("llama.cpp server returned invalid JSON") from exc

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp server returned no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("llama.cpp server returned an invalid choice")
        message = first_choice.get("message")
        text = message.get("content") if isinstance(message, dict) else first_choice.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("llama.cpp server returned an empty message")
        return CoachSummaryGeneration(
            text=text,
            provider=LLAMA_CPP_PROVIDER,
            model=self.model,
        )


def get_coach_summary_model() -> CoachSummaryModel | None:
    load_local_env()

    local_model_dir = os.getenv(LOCAL_MODEL_DIR_ENV)
    if local_model_dir:
        return LocalTransformersCoachSummaryModel(
            model=local_model_dir,
            max_tokens=_env_int(MAX_TOKENS_ENV, 700),
            max_input_tokens=_env_int(MAX_INPUT_TOKENS_ENV, DEFAULT_MAX_INPUT_TOKENS),
            temperature=_env_float(TEMPERATURE_ENV, 0.0),
            token=os.getenv(HF_TOKEN_ENV),
            base_model=os.getenv(BASE_MODEL_ENV),
            adapter_id=os.getenv(ADAPTER_ENV),
        )

    provider = _configured_provider()
    if provider == "hf_inference":
        if env_truthy(os.getenv(DISABLE_REMOTE_ENV)):
            return None
        return HFInferenceCoachSummaryModel(
            model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
            max_tokens=_env_int(MAX_TOKENS_ENV, 700),
            temperature=_env_float(TEMPERATURE_ENV, 0.1),
            token=os.getenv(HF_TOKEN_ENV),
        )

    if provider in LOCAL_TRANSFORMERS_ALIASES:
        return LocalTransformersCoachSummaryModel(
            model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
            max_tokens=_env_int(MAX_TOKENS_ENV, 700),
            max_input_tokens=_env_int(MAX_INPUT_TOKENS_ENV, DEFAULT_MAX_INPUT_TOKENS),
            temperature=_env_float(TEMPERATURE_ENV, 0.0),
            token=os.getenv(HF_TOKEN_ENV),
        )

    if provider in LLAMA_CPP_ALIASES:
        return LlamaCppServerCoachSummaryModel(
            model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
            base_url=os.getenv(LLAMA_CPP_BASE_URL_ENV, DEFAULT_LLAMA_CPP_BASE_URL),
            max_tokens=_env_int(MAX_TOKENS_ENV, 700),
            temperature=_env_float(TEMPERATURE_ENV, 0.1),
            timeout_sec=_env_float(LLAMA_CPP_TIMEOUT_ENV, 120.0),
        )

    if env_truthy(os.getenv(DISABLE_REMOTE_ENV)):
        return None
    return None
