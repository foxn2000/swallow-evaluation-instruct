# MIT License

# Copyright (c) 2026 Swallow LLM team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""BFCL v4 のテストデータの取得．

BFCL v4 のテストデータは HuggingFace Hub ではなく
[gorilla リポジトリ](https://github.com/ShishirPatil/gorilla) の
`berkeley-function-call-leaderboard/bfcl_eval/data/` で配布されている
（HuggingFace の `gorilla-llm/Berkeley-Function-Calling-Leaderboard` は v3 まで）．

そのため本モジュールでは，バージョンを固定したコミットから GitHub 経由で
テストデータを取得し，ローカルにキャッシュする．オフライン環境や，
別のバージョンのデータで評価したい場合は，環境変数 `BFCL_V4_DATA_DIR` に
`bfcl_eval/data` 相当のディレクトリのパスを指定すると，そちらが使われる．
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from lighteval.utils.local_datasets import get_cache_dir


logger = logging.getLogger(__name__)


# テストデータを取得する gorilla リポジトリのコミット．
# 評価の再現性のためにバージョンを固定する．
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_DATA_BASE_URL = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_COMMIT}/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

# ダウンロードのタイムアウト（秒）．
_DOWNLOAD_TIMEOUT = 120

# 正解（ground truth）が存在しないカテゴリ．
# irrelevance / relevance は「関数呼び出しを出力すべきか否か」で採点するため，
# 正解の関数呼び出しを持たない．
CATEGORIES_WITHOUT_GROUND_TRUTH = frozenset(
    {"irrelevance", "live_irrelevance", "live_relevance"}
)


def _local_data_dir() -> Path | None:
    data_dir = os.getenv("BFCL_V4_DATA_DIR")
    if not data_dir:
        return None
    path = Path(data_dir)
    if not path.is_dir():
        raise ValueError(f"BFCL_V4_DATA_DIR is set to '{data_dir}', which is not a directory.")
    return path


def _download(url: str, destination: Path) -> None:
    logger.info(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:
            payload = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download BFCL v4 test data from {url} (HTTP {e.code}). "
            "If this machine cannot reach GitHub, download "
            "berkeley-function-call-leaderboard/bfcl_eval/data from "
            "https://github.com/ShishirPatil/gorilla and set BFCL_V4_DATA_DIR to it."
        ) from e

    # 書き込み途中のファイルがキャッシュとして残らないよう，一時ファイル経由で置き換える．
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def _cached_file(relative_path: str) -> Path:
    """テストデータのファイルを取得し，そのパスを返す．

    Args:
        relative_path: `bfcl_eval/data` 以下の相対パス
            （例：``BFCL_v4_simple_python.json``）．
    """
    local_dir = _local_data_dir()
    if local_dir is not None:
        path = local_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"'{path}' does not exist. BFCL_V4_DATA_DIR must point at a directory "
                "laid out like berkeley-function-call-leaderboard/bfcl_eval/data."
            )
        return path

    cache_dir = get_cache_dir(f"bfcl_v4/{BFCL_COMMIT}")
    path = cache_dir / relative_path
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        _download(f"{BFCL_DATA_BASE_URL}/{relative_path}", path)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    """BFCL のテストデータを読む．

    拡張子は `.json` だが，実際には1行1エントリの JSON Lines 形式．
    """
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse {path}:{line_number} as JSON.") from e
    return entries


def load_test_entries(test_category: str) -> list[dict]:
    """指定したカテゴリのテストエントリを，正解と結合して返す．

    Args:
        test_category: BFCL のテストカテゴリ名（例：``simple_python``）．

    Returns:
        list[dict]: 各要素は BFCL のテストエントリ（``id`` / ``question`` /
            ``function``）に，正解がある場合は ``ground_truth`` を加えたもの．
    """
    entries = _read_jsonl(_cached_file(f"BFCL_v4_{test_category}.json"))

    if test_category in CATEGORIES_WITHOUT_GROUND_TRUTH:
        for entry in entries:
            entry["ground_truth"] = None
        return entries

    ground_truths = _read_jsonl(_cached_file(f"possible_answer/BFCL_v4_{test_category}.json"))
    ground_truth_by_id = {row["id"]: row["ground_truth"] for row in ground_truths}

    missing = [entry["id"] for entry in entries if entry["id"] not in ground_truth_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} entries in BFCL category '{test_category}' have no ground truth "
            f"(first few: {missing[:5]})."
        )

    for entry in entries:
        entry["ground_truth"] = ground_truth_by_id[entry["id"]]
    return entries
