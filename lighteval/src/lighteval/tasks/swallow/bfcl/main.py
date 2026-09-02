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

"""BFCL v4（Berkeley Function Calling Leaderboard v4）の非Agent部分の実装．

BFCL v4 のテストカテゴリのうち，本実装が対象とするのは **単一ターンの
非Agentカテゴリ** である．具体的には以下の13カテゴリを実装している．

* Non-live（専門家が作成したカテゴリ）：`simple_python`，`simple_java`，
  `simple_javascript`，`multiple`，`parallel`，`parallel_multiple`，`irrelevance`
* Live（ユーザ投稿のカテゴリ）：`live_simple`，`live_multiple`，`live_parallel`，
  `live_parallel_multiple`，`live_irrelevance`，`live_relevance`

対象外としたカテゴリとその理由：

* `memory_*`，`web_search_*`：BFCL v4 が `agentic` として分類しているカテゴリで，
  メモリバックエンドや検索エンジンとの多段のやりとりを必要とするため．
* `multi_turn_*`：`agentic` グループには含まれないが，採点にはモデルが出力した
  関数を模擬APIに対して実行し，その状態遷移を検証する必要があるため，
  実質的にAgent的な実行環境を要する．
* `format_sensitivity`：リーダーボードのスコアに算入されない補助的なカテゴリ．

**採点方法**（BFCL 公式の prompting モードに準拠）：

1. 関数定義（`function`）に言語固有の注記を付与する
   （`_func_doc_language_specific_pre_processing`）．
2. BFCL の既定のシステムメッセージを組み立て，モデルに
   `[func_name(param=value)]` 形式で関数呼び出しだけを出力させる．
3. モデルの応答を AST として解析する（`default_decode_ast_prompting`）．
4. AST系カテゴリでは `ast_checker` で正解と照合する．
   irrelevance では「関数呼び出しを出力しなかったこと」，relevance では
   「関数呼び出しを出力したこと」を正解とする．

**注意：** 本実装は推論APIに `tools` パラメータを渡す Function Calling モードでは
なく，システムメッセージで出力形式を指示する prompting モードで評価します
（BFCL のリーダーボードにおける "(Prompt)" 表記のモデルに相当）．
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy

import numpy as np

from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.swallow.bfcl._vendor_bfcl.ast_checker import ast_checker
from lighteval.tasks.swallow.bfcl._vendor_bfcl.default_prompts import (
    DEFAULT_SYSTEM_PROMPT_FORMAT,
)
from lighteval.tasks.swallow.bfcl._vendor_bfcl.enums import Language, ReturnFormat
from lighteval.tasks.swallow.bfcl._vendor_bfcl.parse_utils import (
    _func_doc_language_specific_pre_processing,
    default_decode_ast_prompting,
    formulate_system_prompt,
    is_empty_output,
    is_function_calling_format_output,
    is_java,
    is_js,
)
from lighteval.tasks.swallow.bfcl.data import (
    CATEGORIES_WITHOUT_GROUND_TRUTH,
    load_test_entries,
)
from lighteval.utils.local_datasets import register_local_dataset


logger = logging.getLogger(__name__)


# 本実装が対象とする，BFCL v4 の単一ターン非Agentカテゴリ．
NON_LIVE_AST_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
]
LIVE_AST_CATEGORIES = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]
RELEVANCE_CATEGORIES = ["irrelevance", "live_irrelevance", "live_relevance"]

NON_AGENTIC_CATEGORIES = NON_LIVE_AST_CATEGORIES + LIVE_AST_CATEGORIES + RELEVANCE_CATEGORIES

SUBMETRIC_NAMES = ["bfcl_accuracy"]


def _language_for(test_category: str) -> tuple[Language, ReturnFormat]:
    if is_java(test_category):
        return Language.JAVA, ReturnFormat.JAVA
    if is_js(test_category):
        return Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
    return Language.PYTHON, ReturnFormat.PYTHON


def _load_bfcl_dataset(test_category: str):
    """`swallow-local:` 経由で呼ばれるデータセットローダー．"""
    from datasets import Dataset, DatasetDict

    if test_category not in NON_AGENTIC_CATEGORIES:
        raise ValueError(
            f"'{test_category}' is not a supported BFCL v4 test category. "
            f"Supported categories: {NON_AGENTIC_CATEGORIES}"
        )

    entries = load_test_entries(test_category)

    # `question` / `function` / `ground_truth` は入れ子構造が一定ではないため，
    # datasets の型推論に任せず JSON 文字列として保持し，prompt_function 側で戻す．
    rows = [
        {
            "id": entry["id"],
            "test_category": test_category,
            "question": json.dumps(entry["question"], ensure_ascii=False),
            "function": json.dumps(entry["function"], ensure_ascii=False),
            "ground_truth": json.dumps(entry["ground_truth"], ensure_ascii=False),
        }
        for entry in entries
    ]
    return DatasetDict({"test": Dataset.from_list(rows)})


BFCL_HF_REPO = register_local_dataset("bfcl_v4", _load_bfcl_dataset)


def bfcl_prompt(line, task_name: str = "") -> Doc:
    """BFCL のテストエントリを lighteval の Doc に変換する．

    BFCL の prompting モードにならい，システムメッセージで利用可能な関数と
    出力形式を指示し，ユーザメッセージに設問を置く．
    """
    test_category = line["test_category"]
    question = json.loads(line["question"])
    functions = json.loads(line["function"])
    ground_truth = json.loads(line["ground_truth"])

    # BFCL 公式実装では，モデルに与える関数定義には言語固有の注記を付与するが
    # （`load_dataset_entry(include_language_specific_hint=True)`），採点時には
    # 付与前の関数定義を使う（`include_language_specific_hint=False`）．
    # 注記の付与処理は Java / JavaScript の引数の型を "string" に書き換えるため，
    # 書き換え後の定義を ast_checker に渡すと型チェックが破綻する．
    # そのため両方を保持する．なお前処理は引数を破壊的に書き換えるので複製して渡す．
    prompt_functions = _func_doc_language_specific_pre_processing(
        deepcopy(functions), test_category
    )

    system_prompt = formulate_system_prompt(
        format_sensitivity_config=DEFAULT_SYSTEM_PROMPT_FORMAT, functions=prompt_functions
    )

    # BFCL の `question` は「ターンのリスト」の「メッセージのリスト」という
    # 二重のリスト．単一ターンのカテゴリなので最初のターンだけを使う．
    messages = question[0] if question and isinstance(question[0], list) else question
    user_turns = [message["content"] for message in messages if message.get("role") == "user"]
    # BFCL のテストエントリ自体がシステムメッセージを持つ場合は，公式実装と同じく
    # 既定のシステムメッセージの後に連結する．
    entry_system_prompts = [
        message["content"] for message in messages if message.get("role") == "system"
    ]
    if entry_system_prompts:
        system_prompt = system_prompt + "\n\n" + "\n\n".join(entry_system_prompts)

    # lighteval は `instruction` が `query` の接頭辞であることを要求し
    # （`PromptManager.doc_to_text()` を参照），`--use-chat-template` を付けた場合は
    # ユーザ発話の先頭に連結する（`--system-prompt` を併用した場合は
    # システムメッセージ側に置かれる）．そのためBFCLのシステムメッセージも
    # `query` の接頭辞として渡す．
    instruction = system_prompt + "\n\n"

    return Doc(
        task_name=task_name,
        query=instruction + "\n\n".join(user_turns),
        choices=[""],
        gold_index=0,
        instruction=instruction,
        specific={
            "id": line["id"],
            "test_category": test_category,
            # 関数定義と正解は入れ子構造のスキーマがエントリごとに異なるため，
            # `--save-details` の Parquet 書き出しが型を決められない．
            # JSON文字列のまま保持し，採点時に復元する．
            # なお採点には言語固有の注記を付与する前の関数定義を使う（上のコメント参照）．
            "function_json": line["function"],
            "ground_truth_json": line["ground_truth"],
        },
    )


def _score_relevance_entry(prediction: str, test_category: str) -> tuple[int, str]:
    """irrelevance / relevance カテゴリを採点する．

    irrelevance は「関数呼び出しを出力しないこと」，relevance は
    「関数呼び出しを出力すること」が正解．AST として解析できない応答や
    空のリストは，関数呼び出しなしとみなす．
    """
    contains_function_call = False
    decode_error = None
    try:
        decoded = default_decode_ast_prompting(prediction, ReturnFormat.PYTHON)
        contains_function_call = not is_empty_output(decoded)
    except Exception as e:  # noqa: BLE001 - BFCL 公式実装と同じく解析失敗は「呼び出しなし」
        decode_error = f"{type(e).__name__}: {e}"

    if "irrelevance" in test_category:
        success = not contains_function_call
        error = "" if success else "Valid syntax. Successfully decoded AST when it should not."
    else:
        success = contains_function_call
        error = "" if success else f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"

    return int(success), error


def _score_ast_entry(prediction: str, formatted_doc: Doc) -> tuple[int, str]:
    """AST系カテゴリを採点する．"""
    test_category = formatted_doc.specific["test_category"]
    language, return_format = _language_for(test_category)

    try:
        decoded = default_decode_ast_prompting(prediction, return_format)
    except Exception as e:  # noqa: BLE001 - 解析できない応答は不正解
        return 0, f"ast_decoder:decoder_failed - {type(e).__name__}: {e}"

    if not is_function_calling_format_output(decoded):
        return 0, "ast_decoder:decoder_wrong_output_format"

    try:
        checker_result = ast_checker(
            json.loads(formatted_doc.specific["function_json"]),
            decoded,
            json.loads(formatted_doc.specific["ground_truth_json"]),
            language,
            test_category,
        )
    except Exception as e:  # noqa: BLE001 - checker 内の想定外エラーも不正解として扱う
        logger.warning(
            f"ast_checker raised for {formatted_doc.specific['id']}: {type(e).__name__}: {e}"
        )
        return 0, f"ast_checker:exception - {type(e).__name__}: {e}"

    if checker_result["valid"]:
        return 1, ""
    return 0, f"{checker_result.get('error_type', 'unknown')} - {checker_result.get('error', '')}"


def bfcl_sample_level_fn(formatted_doc: Doc, predictions: list[str], **kwargs) -> dict:
    prediction = predictions[0]
    test_category = formatted_doc.specific["test_category"]

    if test_category in CATEGORIES_WITHOUT_GROUND_TRUTH:
        accuracy, error = _score_relevance_entry(prediction, test_category)
    else:
        accuracy, error = _score_ast_entry(prediction, formatted_doc)

    return {
        "bfcl_accuracy": accuracy,
        # 以下は集計対象外．--save-details で保存される詳細情報．
        "bfcl_error": error,
    }


bfcl_metrics = SampleLevelMetricGrouping(
    metric_name=SUBMETRIC_NAMES,
    higher_is_better=dict.fromkeys(SUBMETRIC_NAMES, True),
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=bfcl_sample_level_fn,
    corpus_level_fn={"bfcl_accuracy": np.mean},
)


def _make_bfcl_task(test_category: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"bfcl_v4:{test_category}",
        prompt_function=bfcl_prompt,
        suite=["swallow"],
        hf_repo=BFCL_HF_REPO,
        hf_subset=test_category,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        metric=[bfcl_metrics],
        generation_size=2048,
        stop_sequence=[],
        version="0.1",
    )


bfcl_v4_tasks = [_make_bfcl_task(test_category) for test_category in NON_AGENTIC_CATEGORIES]

TASKS_TABLE = bfcl_v4_tasks
