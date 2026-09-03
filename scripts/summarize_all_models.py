#!/usr/bin/env python3
"""全モデル・全ベンチマークのスコアを1つのMarkdown表にする（Swallow独自拡張）．

使い方：
    python scripts/summarize_all_models.py

カレントディレクトリの runs/<モデル名>/ を読み，Markdown の表を標準出力に書く．
LiveCodeBench を分割実行している場合は part を問題数で重み付けして統合する．
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from aggregate_lcb_parts import aggregate

MODELS = [
    ("gemma-4-31b-it-reasoning", "gemma-4-31b-it 推論有効"),
    ("qwen3.8-27b",              "qwen3.8-27b"),
    ("kimi-k2.6",                "kimi-k2.6"),
    ("deepseek-v4-flash",        "deepseek-v4-flash"),
    ("suite",                    "gemma-4-31b-it 非推論"),
]
# 表示するベンチマークと採用する指標（ラベル, タスク名, 指標キー, 100倍するか）
ROWS = [
    ("MT-Bench 日本語 (総合)",       "japanese_mt_bench",            "judge_score_overall_avg", False),
    ("MT-Bench 英語 (総合)",         "english_mt_bench",             "judge_score_overall_avg", False),
    ("JamC-QA",                      "jamcqa",                       None, True),
    ("M-IFEval-Ja",                  "mifeval_ja",                   None, True),
    ("IFBench",                      "ifbench_singleturn",           None, True),
    ("MMLU-ProX-JA (91科目平均)",    "__MMLU-ProX-JA (91科目平均)",  "acc", True),
    ("GPQA-JA",                      "swallow_gpqa_ja",              None, True),
    ("PolyMath-JA low",              "polymath_japanese:low",              None, True),
    ("PolyMath-JA medium",           "polymath_japanese:medium",           None, True),
    ("PolyMath-JA high",             "polymath_japanese:high",             None, True),
    ("PolyMath-JA top",              "polymath_japanese:top",              None, True),
    ("AIME 2024",                    "aime_N4:24",                   None, True),
    ("AIME 2025",                    "aime_N4:25",                   None, True),
    ("JHumanEval",                   "swallow_jhumaneval",           None, True),
    ("LiveCodeBench v6 pass@1",      "lcb:codegeneration_v6",        "codegen_pass@1:10", True),
    ("LiveCodeBench v6 pass@10",     "lcb:codegeneration_v6",        "codegen_pass@10:10", True),
    ("WMT20 en-ja",                  "wmt20:en-ja",                  None, False),
    ("WMT20 ja-en",                  "wmt20:ja-en",                  None, False),
    ("JFBench n=1",                  "jfbench:n1",                   "jfbench_score", False),
    ("JFBench n=2",                  "jfbench:n2",                   "jfbench_score", False),
    ("JFBench n=4",                  "jfbench:n4",                   "jfbench_score", False),
    ("JFBench n=8",                  "jfbench:n8",                   "jfbench_score", False),
    ("AnswerCarefully 安全性スコア", "answercarefully",              "safety_score", False),
    ("AnswerCarefully 違反率(%)",    "answercarefully",              "violation_rate", False),
]
BFCL = ["simple_python","simple_java","simple_javascript","multiple","parallel",
        "parallel_multiple","irrelevance","live_simple","live_parallel",
        "live_parallel_multiple","live_relevance"]

def collect(run_dir: Path):
    latest = {}
    for path in sorted(run_dir.glob("results/**/results_*.json")):
        try: payload = json.loads(path.read_text())
        except Exception: continue
        mtime = path.stat().st_mtime
        for task_key, metrics in (payload.get("results") or {}).items():
            if task_key == "all": continue
            name = task_key.split("|")[1] if "|" in task_key else task_key
            if name not in latest or mtime > latest[name][0]:
                latest[name] = (mtime, metrics)
    res = {n: m for n, (_, m) in latest.items()}
    for item in aggregate(run_dir):
        if not item["missing_parts"]:
            res[item["task"]] = item["metrics"]
    mm = [v for k, v in res.items() if k.startswith("mmlu_prox_japanese:")]
    vals = [v.get("extractive_match") if v.get("extractive_match") is not None else v.get("acc") for v in mm]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if vals:
        res["__MMLU-ProX-JA (91科目平均)"] = {"acc": sum(vals)/len(vals)}
    bf = []
    for c in BFCL:
        m = res.get(f"bfcl_v4:{c}")
        if m:
            v = m.get("bfcl_accuracy")
            if isinstance(v, (int, float)): bf.append(v)
    if bf: res["__BFCL v4 (11カテゴリ平均)"] = {"acc": sum(bf)/len(bf)}
    return res

def pick(metrics, key):
    if metrics is None: return None
    if key: 
        v = metrics.get(key)
        return v if isinstance(v, (int, float)) else None
    for k, v in metrics.items():
        if k.endswith("_stderr") or not isinstance(v, (int, float)): continue
        return v
    return None

data = {}
for d, label in MODELS:
    p = Path("runs")/d
    if p.exists(): data[label] = collect(p)

labels = list(data.keys())
print("| ベンチマーク | " + " | ".join(labels) + " |")
print("|---|" + "---:|"*len(labels))
rows = ROWS + [("BFCL v4 (11カテゴリ平均, %)", "__BFCL v4 (11カテゴリ平均)", "acc", True)]
for disp, task, key, pct in rows:
    cells = []
    for l in labels:
        v = pick(data[l].get(task), key)
        cells.append("—" if v is None else (f"{v*100:.2f}" if pct else f"{v:.4f}"))
    print(f"| {disp} | " + " | ".join(cells) + " |")
print()
print("### BFCL v4 カテゴリ別（正解率 %）")
print("| カテゴリ | " + " | ".join(labels) + " |")
print("|---|" + "---:|"*len(labels))
for c in BFCL:
    cells = []
    for l in labels:
        v = pick(data[l].get(f"bfcl_v4:{c}"), "bfcl_accuracy")
        cells.append("—" if v is None else f"{v*100:.2f}")
    print(f"| {c} | " + " | ".join(cells) + " |")
