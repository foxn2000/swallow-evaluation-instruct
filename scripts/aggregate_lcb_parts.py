#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Swallow LLM team
"""分割実行した LiveCodeBench の結果を1つにまとめる（Swallow独自拡張）．

`LCB_NUM_PARTS` を指定して実行すると，LiveCodeBench は
`lcb:codegeneration_{サブセット}_part{i}of{N}` という分割タスクとして採点される．
pass@k は part ごとの平均値になるため，全体の値を得るには part ごとの問題数で
重み付けして平均する必要がある．

使い方：
    python scripts/aggregate_lcb_parts.py <出力ディレクトリ> [...]

例：
    python scripts/aggregate_lcb_parts.py ./runs/gemma-4-31b-it-reasoning
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


PART_PATTERN = re.compile(r"^(?P<base>lcb:codegeneration_[A-Za-z0-9_]+?)_part(?P<index>\d+)of(?P<total>\d+)$")


def _iter_results(run_dir: Path):
    """results_*.json を新しい順に読み，タスクごとに最新の結果だけを返す．"""
    latest: dict[str, tuple[float, dict, dict]] = {}
    for path in sorted(run_dir.glob("results/**/results_*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        mtime = path.stat().st_mtime
        for task_key, metrics in (payload.get("results") or {}).items():
            if task_key == "all":
                continue
            # "swallow|lcb:codegeneration_v6_part1of5|0" -> "lcb:codegeneration_v6_part1of5"
            name = task_key.split("|")[1] if "|" in task_key else task_key
            if name not in latest or mtime > latest[name][0]:
                latest[name] = (mtime, metrics, payload)
    for name, (_, metrics, payload) in latest.items():
        yield name, metrics, payload


def _num_samples(payload: dict, task_name: str) -> int | None:
    """そのタスクで評価した問題数を取り出す．

    summary_tasks のキーは "swallow|lcb:codegeneration_v6_part1of5|0" の形式で，
    問題数は truncated と non_truncated の和．
    """
    summary = payload.get("summary_tasks") or {}
    for key, value in summary.items():
        if not isinstance(value, dict):
            continue
        name = key.split("|")[1] if "|" in key else key
        if name != task_name:
            continue
        total = 0
        for field in ("truncated", "non_truncated"):
            count = value.get(field)
            if isinstance(count, int):
                total += count
        if total:
            return total
    return None


def aggregate(run_dir: Path) -> list[dict]:
    parts: dict[str, list[dict]] = defaultdict(list)
    for name, metrics, payload in _iter_results(run_dir):
        matched = PART_PATTERN.match(name)
        if not matched:
            continue
        count = _num_samples(payload, name)
        parts[matched.group("base")].append(
            {
                "name": name,
                "index": int(matched.group("index")),
                "total": int(matched.group("total")),
                "num_samples": count,
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
        )

    aggregated = []
    for base, entries in sorted(parts.items()):
        entries.sort(key=lambda e: e["index"])
        expected = entries[0]["total"]
        missing = sorted({i for i in range(1, expected + 1)} - {e["index"] for e in entries})
        if any(e["num_samples"] is None for e in entries):
            weights = None
        else:
            weights = [e["num_samples"] for e in entries]

        combined = {}
        metric_names = {k for e in entries for k in e["metrics"] if not k.endswith("_stderr")}
        for metric in sorted(metric_names):
            values, used_weights = [], []
            for entry, weight in zip(entries, weights or [1] * len(entries)):
                if metric in entry["metrics"]:
                    values.append(entry["metrics"][metric])
                    used_weights.append(weight)
            if values:
                total_weight = sum(used_weights)
                combined[metric] = sum(v * w for v, w in zip(values, used_weights)) / total_weight

        aggregated.append(
            {
                "task": base,
                "parts_found": len(entries),
                "parts_expected": expected,
                "missing_parts": missing,
                "num_samples_total": sum(weights) if weights else None,
                "weighted": weights is not None,
                "metrics": combined,
                "per_part": entries,
            }
        )
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="lighteval の出力ディレクトリ")
    parser.add_argument("--json", action="store_true", help="JSONで出力する")
    args = parser.parse_args()

    everything = {}
    for run_dir in args.run_dirs:
        results = aggregate(run_dir)
        everything[str(run_dir)] = results
        if args.json:
            continue
        print(f"\n== {run_dir}")
        if not results:
            print("   分割実行された LiveCodeBench の結果が見つかりません")
            continue
        for item in results:
            status = "完了" if not item["missing_parts"] else f"未完了（不足 part: {item['missing_parts']}）"
            weight_note = "" if item["weighted"] else "（問題数が取得できないため単純平均）"
            print(f"   {item['task']}  {item['parts_found']}/{item['parts_expected']} part  {status}{weight_note}")
            if item["num_samples_total"]:
                print(f"     問題数合計: {item['num_samples_total']}")
            for metric, value in item["metrics"].items():
                print(f"     {metric:28s} {value:.4f}")
            for part in item["per_part"]:
                shown = " ".join(f"{k}={v:.4f}" for k, v in sorted(part["metrics"].items()) if not k.endswith("_stderr"))
                print(f"       part{part['index']}: n={part['num_samples']} {shown}")

    if args.json:
        print(json.dumps(everything, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
