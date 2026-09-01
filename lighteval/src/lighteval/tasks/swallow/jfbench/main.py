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

"""JFBench（日本語指示追従ベンチマーク）の実装．

[JFBench](https://github.com/pfnet-research/jfbench) は Preferred Networks が
公開した，日本語の指示追従性能を測るベンチマークである．「プロンプト（指示文）」と
「制約（回答が満たすべき規則）」を組み合わせて設問を生成し，モデルの回答が
全ての制約を満たしたかどうかで採点する．制約の判定は，機械的に検証できるものは
ルールベースで，敬体か否かなど機械的に検証できないものは LLM-as-a-Judge で行う．

本実装は JFBench 公式の評価手順（`jfbench.benchmark.eval`）にならい，
プロンプト種として IFBench の日本語訳（`ifbench`），制約集合として `test` を用い，
制約数 1 / 2 / 4 / 8 それぞれについて 200 件（乱数シード42）を評価する．
設問はプログラムで生成されるため，評価の直前に JFBench 公式の生成器で構築する
（`lighteval/utils/local_datasets.py` の仕組みを使用）．

指標：

* `jfbench_score`：制約充足率．設問の全ての制約を満たした割合（公式の主要指標）．
* `jfbench_constraint_score`：制約単位の充足率．全設問の全制約のうち満たした割合．
* `jfbench_group_{グループ名}`：制約グループ別の充足率．該当グループの制約の
  充足数を，そのグループの制約の出現数で割った値．

LLM-as-a-Judge に使うモデルは，他のベンチマークと同じ環境変数
（`JUDGE_MODEL_NAME` / `JUDGE_BASE_URL` / `JUDGE_API_KEY`）で指定する．
既定値は JFBench 公式と同じ `openai/gpt-oss-120b`（OpenRouter 経由）．
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache

import numpy as np

from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.swallow.jfbench._vendor_jfbench.benchmark import build
from lighteval.tasks.swallow.jfbench.judge_client import JudgeBackedLLMClient
from lighteval.utils.local_datasets import register_local_dataset


logger = logging.getLogger(__name__)


# JFBench 公式の評価設定（`jfbench.benchmark.eval` の既定値および
# Preferred Networks の技術ブログにおけるテストデータの構成に準拠）．
CONSTRAINT_SET = "test"
PROMPT_SOURCE = "ifbench"
SEED = 42
N_BENCHMARK_DATA = 200
N_CONSTRAINTS_VARIANTS = (1, 2, 4, 8)

# 制約グループの一覧（`jfbench/constraints/` のサブパッケージ名に対応）．
CONSTRAINT_GROUPS = (
    "Character",
    "Content",
    "Format",
    "IfbenchCount",
    "IfbenchFormat",
    "IfbenchRatio",
    "IfbenchRepeat",
    "IfbenchSentence",
    "IfbenchWords",
    "Length",
    "Logic",
    "MetaOutput",
    "Notation",
    "Processing",
    "Structure",
    "Style",
)

# 設問を同時に評価する数．LLM-as-a-Judge の制約は1設問ずつ順に判定されるため，
# 設問間の並列化で処理時間を短縮する．
DEFAULT_MAX_CONCURRENCY = 16

SUBMETRIC_NAMES = ["jfbench_score", "jfbench_constraint_score"] + [
    f"jfbench_group_{group}" for group in CONSTRAINT_GROUPS
]


@lru_cache(maxsize=None)
def _judge_client() -> JudgeBackedLLMClient:
    """LLM-as-a-Judge 制約用のクライアント（プロセス内で共有する）．"""
    return JudgeBackedLLMClient()


@lru_cache(maxsize=None)
def _build_benchmark_data(n_constraints: int) -> list:
    """JFBench の設問（`BenchmarkData`）を公式の生成器で構築する．

    同じ引数に対して常に同じ結果を返す（乱数シードを固定しているため）．
    データセットの読み込み時とスコアの算出時の双方から呼ばれるので，
    結果をキャッシュして二度目以降は再構築しない．
    """
    client = _judge_client()
    if n_constraints == 1:
        logger.info(
            "Building JFBench benchmark data for n_constraints=1. "
            "JFBench enumerates every (prompt, constraint) pair before sampling, "
            "so this takes a while."
        )
        data = list(
            build.get_ifbench_benchmark_data(
                client, seed=SEED, constraint_set=CONSTRAINT_SET
            )
        )
        # 公式実装（`jfbench.benchmark.eval._build_dataset`）と同じく，
        # 生成器の側でシャッフルされたリストの先頭から必要な件数を取る．
        data = data[:N_BENCHMARK_DATA]
    else:
        data = list(
            build.get_ifbench_benchmark_data_with_multiple_constraints(
                client,
                n_constraints=n_constraints,
                n_benchmark_data=N_BENCHMARK_DATA,
                seed=SEED,
                constraint_set=CONSTRAINT_SET,
            )
        )
    logger.info(f"Built {len(data)} JFBench entries for n_constraints={n_constraints}.")
    return data


def _load_jfbench_dataset(subset: str):
    """`swallow-local:` 経由で呼ばれるデータセットローダー．

    Args:
        subset: ``n1`` のように制約数を表す文字列．
    """
    from datasets import Dataset, DatasetDict

    n_constraints = _parse_subset(subset)
    data = _build_benchmark_data(n_constraints)

    rows = []
    for index, benchmark_data in enumerate(data):
        meta_data = benchmark_data.meta_data
        rows.append(
            {
                # `index` は `_build_benchmark_data()` が返すリスト内の位置．
                # 制約オブジェクトは datasets に格納できないため，スコアの算出時に
                # この位置で引き当てる（`data_id` はプロンプトが異なっても
                # 同じ値になり得るので識別子には使えない）．
                "index": index,
                "data_id": meta_data.data_id,
                "n_constraints": meta_data.n_constraints,
                "prompt": meta_data.prompt,
                "constraint_types": json.dumps(meta_data.constraint_types, ensure_ascii=False),
                "constraint_groups": json.dumps(meta_data.constraint_groups, ensure_ascii=False),
                "constraint_instructions": json.dumps(
                    meta_data.constraint_instructions, ensure_ascii=False
                ),
            }
        )
    return DatasetDict({"test": Dataset.from_list(rows)})


JFBENCH_HF_REPO = register_local_dataset("jfbench", _load_jfbench_dataset)


def _parse_subset(subset: str) -> int:
    if not subset.startswith("n"):
        raise ValueError(f"JFBench subset must look like 'n1'/'n2'/'n4'/'n8', got '{subset}'.")
    n_constraints = int(subset[1:])
    if n_constraints not in N_CONSTRAINTS_VARIANTS:
        raise ValueError(
            f"JFBench supports n_constraints in {N_CONSTRAINTS_VARIANTS}, got {n_constraints}."
        )
    return n_constraints


def jfbench_prompt(line, task_name: str = "") -> Doc:
    """JFBench の設問を lighteval の Doc に変換する．"""
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={
            "index": line["index"],
            "data_id": line["data_id"],
            "n_constraints": line["n_constraints"],
            "constraint_types": json.loads(line["constraint_types"]),
            "constraint_groups": json.loads(line["constraint_groups"]),
            "constraint_instructions": json.loads(line["constraint_instructions"]),
        },
    )


async def _evaluate_all(
    benchmark_data_list: list, predictions: list[str], max_concurrency: int
) -> list[tuple[dict[str, bool] | None, str]]:
    """全設問の制約充足状況を並行して判定する．

    Returns:
        list[tuple[dict[str, bool] | None, str]]: 各設問について
            (制約クラス名 -> 充足したか, エラーメッセージ)．
            判定に失敗した場合は最初の要素が None になる．
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _evaluate_one(benchmark_data, prediction: str):
        async with semaphore:
            try:
                return await benchmark_data.evaluate(prediction), ""
            except Exception as e:  # noqa: BLE001 - 1件の失敗で評価全体を止めない
                logger.error(
                    f"Failed to evaluate JFBench entry "
                    f"'{benchmark_data.meta_data.data_id}': {type(e).__name__}: {e}"
                )
                return None, f"{type(e).__name__}: {e}"

    return await asyncio.gather(
        *(
            _evaluate_one(benchmark_data, prediction)
            for benchmark_data, prediction in zip(benchmark_data_list, predictions)
        )
    )


def _constraint_satisfaction_rate(list_of_lists: list[list[bool]]) -> float:
    """制約単位の充足率を求める（サンプルごとのリストを平坦化して平均を取る）．

    該当する要素が1つもない場合（そのグループの制約が出現しなかった場合）は
    NaN を返す．

    注意：関数名に "mean" を含めてはならない．lighteval は集計関数の名前に
    "mean" が含まれるかどうかで標準誤差の計算方法を決めており
    （`lighteval/metrics/stderr.py` の `get_stderr_function()` を参照），
    含まれる場合はサンプル値がスカラーであることを前提とした計算を行うため，
    本関数のようにリストを受け取る集計関数では例外が発生する．
    """
    flattened = [value for sublist in list_of_lists for value in sublist]
    if not flattened:
        return float("nan")
    return float(np.mean(flattened))


def jfbench_sample_level_fn(
    sample_ids: list[str], responses: list, formatted_docs: list[Doc], **kwargs
) -> list[dict]:
    """JFBench のスコアを算出する．

    LLM-as-a-Judge の制約を含むため，設問をまとめて非同期に判定する．
    """
    n_constraints_values = {doc.specific["n_constraints"] for doc in formatted_docs}
    if len(n_constraints_values) != 1:
        raise ValueError(
            f"All JFBench docs in a task must share n_constraints, got {n_constraints_values}."
        )
    n_constraints = next(iter(n_constraints_values))

    benchmark_data_all = _build_benchmark_data(n_constraints)
    benchmark_data_list = [benchmark_data_all[doc.specific["index"]] for doc in formatted_docs]
    predictions = [response[0].result[0] for response in responses]

    # 構築した設問と Doc が対応していることを確認する．
    for benchmark_data, doc in zip(benchmark_data_list, formatted_docs):
        if benchmark_data.meta_data.data_id != doc.specific["data_id"]:
            raise ValueError(
                "JFBench benchmark data does not match the evaluated document "
                f"(expected data_id '{doc.specific['data_id']}', "
                f"got '{benchmark_data.meta_data.data_id}')."
            )

    max_concurrency = int(os.getenv("JFBENCH_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
    evaluations = asyncio.run(_evaluate_all(benchmark_data_list, predictions, max_concurrency))

    metrics = []
    for doc, (evaluation, error) in zip(formatted_docs, evaluations):
        constraint_types = doc.specific["constraint_types"]
        constraint_groups = doc.specific["constraint_groups"]

        if evaluation is None:
            # 判定に失敗した設問は，すべての制約を満たさなかったものとして扱う．
            # 失敗の理由は `jfbench_error` として詳細に残る．
            satisfied_per_constraint = [False] * len(constraint_types)
        else:
            satisfied_per_constraint = [
                bool(evaluation.get(constraint_type, False)) for constraint_type in constraint_types
            ]

        per_group: dict[str, list[bool]] = {group: [] for group in CONSTRAINT_GROUPS}
        for group, satisfied in zip(constraint_groups, satisfied_per_constraint):
            if group not in per_group:
                logger.warning(f"Unknown JFBench constraint group '{group}'; adding it on the fly.")
                per_group[group] = []
            per_group[group].append(satisfied)

        metric = {
            "jfbench_score": int(all(satisfied_per_constraint)) if satisfied_per_constraint else 0,
            "jfbench_constraint_score": satisfied_per_constraint,
            # 以下は集計対象外．--save-details で保存される詳細情報．
            "jfbench_data_id": doc.specific["data_id"],
            "jfbench_constraint_types": constraint_types,
            "jfbench_constraint_results": satisfied_per_constraint,
            "jfbench_error": error,
        }
        for group in CONSTRAINT_GROUPS:
            metric[f"jfbench_group_{group}"] = per_group[group]
        metrics.append(metric)

    return metrics


jfbench_metrics = SampleLevelMetricGrouping(
    metric_name=SUBMETRIC_NAMES,
    higher_is_better=dict.fromkeys(SUBMETRIC_NAMES, True),
    # LLM-as-a-Judge の制約があるため，設問をまとめて処理できるカテゴリを指定する．
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=jfbench_sample_level_fn,
    corpus_level_fn={
        "jfbench_score": np.mean,
        "jfbench_constraint_score": _constraint_satisfaction_rate,
        **{f"jfbench_group_{group}": _constraint_satisfaction_rate for group in CONSTRAINT_GROUPS},
    },
)


def _make_jfbench_task(n_constraints: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"jfbench:n{n_constraints}",
        prompt_function=jfbench_prompt,
        suite=["swallow"],
        hf_repo=JFBENCH_HF_REPO,
        hf_subset=f"n{n_constraints}",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        metric=[jfbench_metrics],
        generation_size=4096,
        stop_sequence=[],
        version="0.1",
    )


jfbench_tasks = [
    _make_jfbench_task(n_constraints) for n_constraints in N_CONSTRAINTS_VARIANTS
]

TASKS_TABLE = jfbench_tasks
