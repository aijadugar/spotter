from __future__ import annotations

import os
from pathlib import Path


_LOADED_ENV_FILES: set[Path] = set()


def env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_local_env(filename: str = ".env") -> None:
    candidates = (
        Path.cwd() / filename,
        Path(__file__).resolve().parents[2] / filename,
    )
    for path in candidates:
        resolved = path.resolve()
        if resolved in _LOADED_ENV_FILES or not resolved.is_file():
            continue
        for line in resolved.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'").strip('"')
        _LOADED_ENV_FILES.add(resolved)
