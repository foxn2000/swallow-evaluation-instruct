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

"""JFBench の LLM-as-a-Judge 制約を，本フレームワークのジャッジ設定で動かすための
LLMクライアント．

JFBench の一部の制約（敬体か否かなど機械的に判定できないもの）は
LLM-as-a-Judge で評価される．JFBench 本体の `LLMClient` は OpenRouter の
APIキーを `OPENROUTER_API_KEY` から直接読むため，本フレームワークの
ジャッジ設定（`JUDGE_MODEL_NAME` / `JUDGE_BASE_URL` / `JUDGE_API_KEY`．
詳細は `lighteval/utils/judge_config.py` を参照）が反映されない．

そこで JFBench の制約が使うインタフェース（`async_ask`）だけを実装した
アダプタを用意し，他のベンチマークと同じ設定でジャッジを指定できるようにする．
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any

from lighteval.utils.judge_config import resolve_judge_endpoint


logger = logging.getLogger(__name__)


# JFBench 公式の既定のジャッジモデル（OpenRouter 経由）．
# Ref. https://github.com/pfnet-research/jfbench の src/jfbench/llm.py
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

# JFBench 公式の既定値（src/jfbench/llm.py および benchmark/eval.py より）．
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_CONCURRENCY = 32
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_SECONDS = 1.0


class JudgeBackedLLMClient:
    """JFBench の制約が使う `LLMClient` の最小互換アダプタ．

    JFBench の制約は `await client.async_ask([prompt])` のみを呼ぶため，
    そのインタフェースだけを提供する．
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_concurrency: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        endpoint = resolve_judge_endpoint(default_model=model or DEFAULT_JUDGE_MODEL)

        self.model = endpoint.model
        self.base_url = base_url or endpoint.base_url
        self.api_key = api_key or endpoint.api_key
        self.reasoning_effort = endpoint.reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(1, int(max_retries))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        if max_concurrency is None:
            max_concurrency = int(os.getenv("JUDGE_CONCURRENT_CALLS", DEFAULT_MAX_CONCURRENCY))
        self.max_concurrency = max(1, int(max_concurrency))

        # クライアントとセマフォはイベントループに紐づくため，
        # 実際に使われるイベントループ上で遅延生成する．
        self._client = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        logger.info(
            f"JFBench LLM-as-a-Judge constraints will use model={self.model}, "
            f"base_url={self.base_url or 'https://api.openai.com/v1 (default)'}"
        )

    def _ensure_client(self):
        from openai import AsyncOpenAI

        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._loop = loop
        return self._client, self._semaphore

    async def async_ask(
        self, prompts: list[str], *, use_tqdm: bool = False
    ) -> tuple[list[str], list[Any]]:
        """プロンプトのリストに対する応答を返す．

        Returns:
            tuple[list[str], list[Any]]: (応答テキストのリスト, 生の応答オブジェクトのリスト)
        """
        client, semaphore = self._ensure_client()

        async def _ask_one(index: int, prompt: str) -> tuple[int, str, Any]:
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "n": 1,
            }
            if self.reasoning_effort is not None and self.reasoning_effort != "none":
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    async with semaphore:
                        response = await client.chat.completions.create(**request_kwargs)
                    text = response.choices[0].message.content or ""
                    return index, text.strip(), response
                except Exception as e:  # noqa: BLE001 - リトライして最後に再送出する
                    last_error = e
                    if attempt == self.max_retries - 1:
                        break
                    backoff = self.retry_base_seconds * (2**attempt)
                    logger.warning(
                        f"Judge request failed ({type(e).__name__}: {e}); "
                        f"retrying in {backoff:.1f}s ({attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(backoff + random.uniform(0.0, backoff * 0.25))

            raise RuntimeError(
                f"Failed to get a judge response after {self.max_retries} attempts."
            ) from last_error

        results: list[str] = [""] * len(prompts)
        raw_responses: list[Any] = [None] * len(prompts)
        if not prompts:
            return results, raw_responses

        for coroutine in asyncio.as_completed(
            [_ask_one(index, prompt) for index, prompt in enumerate(prompts)]
        ):
            index, text, raw = await coroutine
            results[index] = text
            raw_responses[index] = raw

        return results, raw_responses

    def to_serializable_dict(self) -> dict[str, Any]:
        """JFBench の制約のシリアライズ処理から呼ばれても壊れないようにしておく．"""
        return {"provider": "swallow-judge", "model": self.model}
