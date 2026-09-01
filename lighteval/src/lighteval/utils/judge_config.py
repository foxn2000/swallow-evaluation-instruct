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

"""LLM-as-a-Judge に使う推論APIの設定を環境変数から解決する（Swallow独自拡張）．

ベンチマークの実装（例：`lighteval/tasks/swallow/japanese_mt_bench/main.py`）は
ジャッジモデルを固定値で宣言しているため，そのままではOpenAI以外のプロバイダを
ジャッジに使えない．本モジュールは以下の環境変数による上書きを提供し，
OpenRouter などOpenAI互換APIを提供する任意のプロバイダをジャッジに使えるようにする．

* ``JUDGE_MODEL_NAME``：ジャッジモデル名．ベンチマーク側の既定値を上書きする．
  （例：``openai/gpt-5.4-mini``）
* ``JUDGE_BASE_URL``：ジャッジ用推論APIのURL．
  （例：``https://openrouter.ai/api/v1``）
* ``JUDGE_API_KEY``：ジャッジ用推論APIのAPIキー．
* ``JUDGE_REASONING_EFFORT``：ジャッジの推論の深さ．``none`` を指定すると無効化する．

利便性のため，``JUDGE_BASE_URL`` が未設定で ``OPENROUTER_API_KEY`` のみが
設定されている場合は，OpenRouter を使うものとみなす．
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass


logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 環境変数の値としてジャッジ設定を「明示的に無効化」する意味で受け付ける文字列．
_NONE_LITERALS = frozenset({"", "none", "null", "false"})

# 同じ警告を繰り返し出さないためのフラグ．
# ジャッジ設定はメトリクスごとに解決されるため，何度も呼ばれる．
_warned_once: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_once:
        return
    _warned_once.add(key)
    logger.warning(message)


def _env(name: str) -> str | None:
    """環境変数を読み，空文字列は未設定として扱う．"""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _is_openrouter(base_url: str | None) -> bool:
    return base_url is not None and "openrouter.ai" in base_url


@dataclass(frozen=True)
class JudgeEndpoint:
    """LLM-as-a-Judge の呼び出し先設定．

    Attributes:
        model: ジャッジモデル名．
        base_url: 推論APIのURL．``None`` の場合はOpenAIの既定URLを使う．
        api_key: 推論APIのAPIキー．
        reasoning_effort: 推論の深さ．``None`` の場合は指定しない．
        short_name: 評価結果のカラム名などに使う短いジャッジ名．
        is_overridden: 環境変数によってベンチマーク側の既定値が上書きされたか．
    """

    model: str
    base_url: str | None
    api_key: str | None
    reasoning_effort: str | None
    short_name: str
    is_overridden: bool


def _resolve_base_url() -> str | None:
    base_url = _env("JUDGE_BASE_URL")
    if base_url is not None:
        if base_url.lower() in _NONE_LITERALS:
            return None
        return base_url

    # JUDGE_BASE_URL が未設定でも，OpenRouter のキーだけが設定されている場合は
    # OpenRouter を使う意図とみなす．OPENAI_API_KEY が併存する場合は，
    # 既存の挙動（OpenAI を使う）を壊さないために上書きしない．
    if _env("OPENROUTER_API_KEY") is not None and _env("OPENAI_API_KEY") is None:
        _warn_once(
            "openrouter-fallback",
            "OPENROUTER_API_KEY is set and OPENAI_API_KEY is not; "
            f"routing LLM-as-a-Judge requests to OpenRouter ({OPENROUTER_BASE_URL}). "
            "Set JUDGE_BASE_URL explicitly to override."
        )
        return OPENROUTER_BASE_URL

    return None


def _resolve_api_key(base_url: str | None) -> str | None:
    api_key = _env("JUDGE_API_KEY")
    if api_key is not None:
        return api_key
    if _is_openrouter(base_url):
        return _env("OPENROUTER_API_KEY") or _env("OPENAI_API_KEY")
    return _env("OPENAI_API_KEY")


def _resolve_reasoning_effort(default_reasoning_effort: str | None) -> str | None:
    reasoning_effort = _env("JUDGE_REASONING_EFFORT")
    if reasoning_effort is None:
        return default_reasoning_effort
    if reasoning_effort.lower() in _NONE_LITERALS:
        return None
    return reasoning_effort


def _qualify_model_for_openrouter(model: str) -> str:
    """OpenRouter は ``{プロバイダ}/{モデルID}`` 形式のモデル名を要求する．

    ベンチマーク側の既定値は ``gpt-4o-2024-08-06`` のようなOpenAIの素のモデル名
    なので，OpenRouter に向ける場合はプロバイダ名を補う．
    """
    if "/" in model:
        return model
    qualified = f"openai/{model}"
    _warn_once(
        f"qualify-{model}",
        f"Judge model '{model}' has no provider prefix, which OpenRouter requires. "
        f"Using '{qualified}' instead. Set JUDGE_MODEL_NAME to choose explicitly.",
    )
    return qualified


def _short_name(model: str) -> str:
    """モデル名から評価結果のカラム名に使う短い名前を作る．"""
    # "openai/gpt-5.4-mini" -> "gpt-5.4-mini"
    return model.rsplit("/", maxsplit=1)[-1]


def resolve_judge_endpoint(
    default_model: str,
    default_reasoning_effort: str | None = None,
    default_short_name: str | None = None,
) -> JudgeEndpoint:
    """ベンチマーク側の既定値に環境変数の設定を反映したジャッジ設定を返す．

    Args:
        default_model: ベンチマークが宣言しているジャッジモデル名．
        default_reasoning_effort: ベンチマークが宣言している推論の深さ．
        default_short_name: ベンチマークが宣言している短いジャッジ名．

    Returns:
        JudgeEndpoint: 解決済みのジャッジ設定．
    """
    override_model = _env("JUDGE_MODEL_NAME")
    base_url = _resolve_base_url()

    model = override_model or default_model
    if _is_openrouter(base_url):
        model = _qualify_model_for_openrouter(model)

    is_overridden = override_model is not None or base_url is not None

    if default_short_name is not None and not is_overridden:
        short_name = default_short_name
    else:
        short_name = _short_name(model)

    endpoint = JudgeEndpoint(
        model=model,
        base_url=base_url,
        api_key=_resolve_api_key(base_url),
        reasoning_effort=_resolve_reasoning_effort(default_reasoning_effort),
        short_name=short_name,
        is_overridden=is_overridden,
    )

    if is_overridden:
        logger.info(
            f"LLM-as-a-Judge endpoint: model={endpoint.model}, "
            f"base_url={endpoint.base_url or 'https://api.openai.com/v1 (default)'}, "
            f"reasoning_effort={endpoint.reasoning_effort}"
        )

    return endpoint
