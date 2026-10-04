from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spotter.contracts import to_dict, validate_contract


def write_json(run_dir: Path, filename: str, payload: Any, *, validate: bool = True) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if validate:
        validate_contract(filename, payload)
    path = run_dir / filename
    path.write_text(json.dumps(to_dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
