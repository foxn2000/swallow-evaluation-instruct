#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Swallow LLM team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""評価結果のディレクトリを走査して，スコアの一覧をMarkdownの表として出力する．

使い方：
    python scripts/summarize_run.py ./runs/suite [--out summary.md]

`--output-dir` で指定したディレクトリ配下の `results/**/results_*.json` を読み，
タスクごとの主要な評価尺度を表にまとめる．同じタスクが複数回実行されている場合は，
最も新しい結果を採用する．
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# タスクごとに「主要な評価尺度」とみなす指標名．
# ここに載っていないタスクは，タスクが返した全ての指標を列挙する．
PRIMARY_METRICS = {
    "jfbench": ["jfbench_score", "jfbench_constraint_score"],
    "answercarefully": ["safety_score", "violation_rate", "acceptable_rate"],
    "bfcl_v4": ["bfcl_accuracy"],
    "japanese_mt_bench": ["judge_score_overall_avg", "japanese_ratio_overall"],
    "english_mt_bench": ["judge_score_overall_avg"],
    "mifeval_ja": ["inst_level_strict_acc", "prompt_level_strict_acc"],
    "ifbench_singleturn": ["inst_level_strict_acc", "prompt_level_strict_acc"],
}

# 表に出さない補助的な指標（プロンプトやジャッジの生出力など）．
EXCLUDED_SUFFIXES = ("_stderr",)
EXCLUDED_PREFIXES = ("user_prompt", "judgement", "judge_prompt")


def _is_reportable(name: str, value: object) -> bool:
    if name.endswith(EXCLUDED_SUFFIXES) or name.startswith(EXCLUDED_PREFIXES):
        return False
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _task_family(task: str) -> str:
    # "swallow|bfcl_v4:simple_python|0" -> "bfcl_v4"
    body = task.split("|")[1] if "|" in task else task
    return body.split(":")[0]


def collect_results(output_dir: Path) -> dict[str, dict]:
    """タスク名 -> 指標の辞書．同じタスクは新しい結果で上書きする．"""
    results: dict[str, dict] = {}
    for path in sorted(output_dir.glob("results/**/results_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"warning: could not parse {path}")
            continue
        for task, metrics in payload.get("results", {}).items():
            if task == "all":
                continue
            results[task] = metrics
    return results


def format_markdown(results: dict[str, dict]) -> str:
    lines = ["| タスク | 評価尺度 | スコア |", "| --- | --- | ---: |"]
    for task in sorted(results):
        metrics = results[task]
        family = _task_family(task)
        preferred = PRIMARY_METRICS.get(family)
        if preferred:
            names = [name for name in preferred if name in metrics]
        else:
            names = [name for name, value in sorted(metrics.items()) if _is_reportable(name, value)]
        if not names:
            continue
        for index, name in enumerate(names):
            value = metrics[name]
            # タスク名には "|" が含まれるため，Markdownの表の列区切りと衝突しないよう退避する．
            shown_task = task.replace("|", "\\|") if index == 0 else ""
            if isinstance(value, float) and math.isnan(value):
                formatted = "NaN"
            else:
                formatted = f"{value:.4f}"
            lines.append(f"| {shown_task} | {name} | {formatted} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="lighteval の --output-dir に指定したディレクトリ")
    parser.add_argument("--out", type=Path, default=None, help="Markdownの出力先（省略時は標準出力）")
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="主要な評価尺度に絞らず，タスクが返した全ての指標を出力する",
    )
    args = parser.parse_args()

    if args.all_metrics:
        PRIMARY_METRICS.clear()

    results = collect_results(args.output_dir)
    if not results:
        raise SystemExit(f"No results found under {args.output_dir}/results/")

    markdown = format_markdown(results)
    header = f"# 評価結果（{len(results)} タスク）\n\n"
    if args.out is not None:
        args.out.write_text(header + markdown + "\n")
        print(f"Wrote {args.out}")
    else:
        print(header + markdown)


if __name__ == "__main__":
    main()
