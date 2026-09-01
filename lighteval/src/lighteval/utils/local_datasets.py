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

"""HuggingFace Hub 以外の場所からデータセットを読み込む仕組み（Swallow独自拡張）．

lighteval のタスク設定は，データセットが HuggingFace Hub にあることを前提に
`hf_repo` / `hf_subset` を指定する．しかしベンチマークによっては，データが
Hub ではなく GitHub で配布されていたり（BFCL v4），評価の直前にプログラムで
生成する必要があったり（JFBench）する．

そこで `hf_repo` に ``swallow-local:{ローダー名}`` という形式の値を指定できる
ようにし，`lighteval.utils.utils.download_dataset_worker()` から本モジュールの
ローダーへ処理を振り分ける．ローダーは `hf_subset` を引数として受け取り，
`datasets.DatasetDict` を返す．

ローダーはタスクを定義するモジュールの読み込み時に `register_local_dataset()`
で登録する．データの取得・生成はローダーが呼ばれた時点（=評価の直前）に
行われるため，タスク一覧の表示など，データを必要としない操作は高速なまま保たれる．
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from datasets import DatasetDict


logger = logging.getLogger(__name__)


LOCAL_DATASET_PREFIX = "swallow-local:"

_LOADERS: dict[str, Callable[[str], DatasetDict]] = {}


def register_local_dataset(name: str, loader: Callable[[str], DatasetDict]) -> str:
    """ローカルデータセットのローダーを登録し，`hf_repo` に指定する値を返す．

    Args:
        name: ローダー名．
        loader: `hf_subset` を受け取り `DatasetDict` を返す関数．

    Returns:
        str: タスク設定の `hf_repo` に指定する値（``swallow-local:{name}``）．
    """
    if name in _LOADERS and _LOADERS[name] is not loader:
        raise ValueError(f"A different loader is already registered for local dataset '{name}'.")
    _LOADERS[name] = loader
    return f"{LOCAL_DATASET_PREFIX}{name}"


def is_local_dataset(dataset_path: str) -> bool:
    """`hf_repo` の値がローカルデータセットを指しているかどうかを返す．"""
    return isinstance(dataset_path, str) and dataset_path.startswith(LOCAL_DATASET_PREFIX)


def load_local_dataset(dataset_path: str, dataset_config_name: str) -> DatasetDict:
    """ローカルデータセットを読み込む．

    Args:
        dataset_path: ``swallow-local:{ローダー名}`` 形式の文字列．
        dataset_config_name: ローダーに渡す設定名（`hf_subset`）．

    Returns:
        DatasetDict: 読み込んだデータセット．
    """
    name = dataset_path[len(LOCAL_DATASET_PREFIX) :]
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"No loader is registered for local dataset '{name}'. "
            f"Registered loaders: {sorted(_LOADERS)}. "
            "Make sure the module that defines the task has been imported."
        )
    logger.info(f"Loading local dataset '{name}' (subset={dataset_config_name})")
    return loader(dataset_config_name)


def get_cache_dir(subdirectory: str) -> Path:
    """ローカルデータセットのキャッシュ用ディレクトリを返す（必要なら作成する）．

    環境変数 `SWALLOW_EVAL_CACHE_DIR` で保存先を変更できる．
    """
    base = os.getenv("SWALLOW_EVAL_CACHE_DIR")
    if base:
        root = Path(base)
    else:
        root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "swallow-evaluation-instruct"
    path = root / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path
