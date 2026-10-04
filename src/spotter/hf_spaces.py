from __future__ import annotations

import os
from typing import Any, Callable, TypeVar, cast


F = TypeVar("F", bound=Callable[..., Any])
SPACES_GPU_DURATION_ENV = "SPOTTER_SPACES_GPU_DURATION"
ROUTER_DEVICE_ENV = "SPOTTER_ROUTER_DEVICE"


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def zero_gpu_enabled() -> bool:
    if _env_truthy(os.getenv("SPACES_ZERO_GPU")):
        return True

    try:
        from spaces.zero import Config
    except Exception:
        return False

    return bool(getattr(Config, "zero_gpu", False))


def router_torch_device() -> str:
    configured_device = os.getenv(ROUTER_DEVICE_ENV)
    if configured_device:
        return configured_device.strip().lower()
    return "cpu"


def spaces_gpu(*, duration: Any = None, size: str | None = None) -> Callable[[F], F]:
    try:
        import spaces
    except Exception:
        return lambda function: function

    gpu = getattr(spaces, "GPU", None)
    if not callable(gpu):
        return lambda function: function

    def decorator(function: F) -> F:
        if duration is None and size is None:
            return cast(F, gpu(function))
        return cast(F, gpu(duration=duration, size=size)(function))

    return decorator


def default_spaces_gpu_duration() -> int:
    configured_duration = os.getenv(SPACES_GPU_DURATION_ENV)
    if configured_duration:
        try:
            return max(1, int(configured_duration))
        except ValueError:
            return 120
    return 120
