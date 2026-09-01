from __future__ import annotations

import os
from pathlib import Path


def _find_data_dir() -> Path:
    env_data_dir = os.environ.get("JFBENCH_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)

    # swallow-evaluation-instruct: upstream then walked parent directories looking for a
    # pyproject.toml to anchor a "data" fallback dir; inside this repo that walk would
    # escape _vendor_jfbench and resolve to an unrelated directory in the host project,
    # so the fallback is now just the package-relative data dir, unconditionally.
    return Path(__file__).resolve().parent / "data"


DATA_DIR = _find_data_dir()
