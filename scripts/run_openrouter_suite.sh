#!/bin/bash
# OpenRouter のモデルに対してベンチマーク群を順番に実行する．
#
# 各ベンチマークを個別の lighteval プロセスで実行し，ベンチマークごとに
# ログ・結果・詳細（プロンプト・モデル出力・ジャッジの応答など）を保存する．
# 途中で中断した場合は，同じ引数で再実行すると完了済みのベンチマークを飛ばして再開する．
#
# 使い方：
#   OPENROUTER_API_KEY=sk-or-... ./scripts/run_openrouter_suite.sh [出力ディレクトリ]
#
# 生成パラメータは BENCHMARKS.md の推奨設定に従う（非推論型モデルを想定）．
#   * 推奨設定がないベンチマーク：貪欲法（temperature=0）
#   * JHumanEval：temperature=0.2, top_p=0.95
#   * LiveCodeBench, AIME：temperature=0.6, top_p=0.95
#   * MT-Bench：カテゴリごとに定められた temperature が自動適用されるため指定しない

set -uo pipefail

MODEL_NAME="${MODEL_NAME:-openrouter/google/gemma-4-31b-it}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.4-mini}"
OUT_DIR="${1:-./runs/suite}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY を設定してください}"

# LLM-as-a-Judge の設定（MT-Bench・AnswerCarefully・JFBench が使用する）．
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_MODEL_NAME="$JUDGE_MODEL"

# 推論APIの並列数．
export LITELLM_CONCURRENT_CALLS="${LITELLM_CONCURRENT_CALLS:-32}"
export JUDGE_CONCURRENT_CALLS="${JUDGE_CONCURRENT_CALLS:-32}"
export JFBENCH_MAX_CONCURRENCY="${JFBENCH_MAX_CONCURRENCY:-32}"

# トークナイザのダウンロードで警告が出るのを抑える．
export TOKENIZERS_PARALLELISM=false


LIGHTEVAL="${LIGHTEVAL:-.venv/bin/lighteval}"

LOG_DIR="$OUT_DIR/logs"
STATE_DIR="$OUT_DIR/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"
# 推論APIの呼び出し1回ごとの情報（トークン数・課金額・応答時間）を記録する．
export LITELLM_REQUEST_LOG="${LITELLM_REQUEST_LOG:-$OUT_DIR/api_requests.jsonl}"

SUITE_LOG="$LOG_DIR/00_suite.log"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$SUITE_LOG"
}

# run_task <ラベル> <タスクID> <生成パラメータ> [追加の実行時引数...]
run_task() {
    local label="$1"; shift
    local task_id="$1"; shift
    local gen_params="$1"; shift

    local done_marker="$STATE_DIR/$label.done"
    if [[ -f "$done_marker" ]]; then
        log "SKIP  $label (already completed)"
        return 0
    fi

    local model_args="model=$MODEL_NAME,api_key=$OPENROUTER_API_KEY"
    if [[ -n "$gen_params" ]]; then
        model_args="$model_args,generation_parameters={$gen_params}"
    fi

    local task_log="$LOG_DIR/$label.log"
    log "START $label -> $task_id"
    local started
    started=$(date +%s)

    if "$LIGHTEVAL" endpoint litellm \
            "$model_args" \
            "${task_id}|0|0" \
            --use-chat-template \
            --output-dir "$OUT_DIR" \
            --save-details \
            "$@" >"$task_log" 2>&1; then
        local elapsed=$(( $(date +%s) - started ))
        printf '%s\n' "$task_id" >"$done_marker"
        log "OK    $label (${elapsed}s)"
    else
        local elapsed=$(( $(date +%s) - started ))
        log "FAIL  $label (${elapsed}s) -- see $task_log"
        printf '%s\n' "$task_id" >"$STATE_DIR/$label.failed"
    fi
}

GREEDY="temperature:0.0,max_new_tokens:4096"
GREEDY_LONG="temperature:0.0,max_new_tokens:8192"
CODE="temperature:0.2,top_p:0.95,max_new_tokens:4096"
SAMPLING="temperature:0.6,top_p:0.95,max_new_tokens:8192"
# MT-Bench はカテゴリごとの temperature が自動適用されるため temperature を指定しない．
MT_BENCH="max_new_tokens:4096"

log "===== suite start: model=$MODEL_NAME judge=$JUDGE_MODEL out=$OUT_DIR ====="

# ---- 追加したベンチマーク -------------------------------------------------
run_task jfbench_n1               "swallow|jfbench:n1"               "$GREEDY"
run_task jfbench_n2               "swallow|jfbench:n2"               "$GREEDY"
run_task jfbench_n4               "swallow|jfbench:n4"               "$GREEDY"
run_task jfbench_n8               "swallow|jfbench:n8"               "$GREEDY"
run_task answercarefully          "swallow|answercarefully"          "$GREEDY"

for category in simple_python simple_java simple_javascript multiple parallel \
                parallel_multiple irrelevance live_simple live_parallel \
                live_parallel_multiple live_relevance; do
    run_task "bfcl_$category" "swallow|bfcl_v4:$category" "$GREEDY"
done

# ---- 指示追従 -------------------------------------------------------------
run_task mifeval_ja               "swallow|mifeval_ja"               "$GREEDY"
run_task ifbench                  "swallow|ifbench_singleturn"       "$GREEDY"

# ---- 対話（LLM-as-a-Judge） ----------------------------------------------
run_task japanese_mt_bench        "swallow|japanese_mt_bench"        "$MT_BENCH"
run_task english_mt_bench         "swallow|english_mt_bench"         "$MT_BENCH"

# ---- 知識・一般教養 -------------------------------------------------------
run_task jamcqa                   "swallow|jamcqa"                   "$GREEDY"
run_task swallow_gpqa_ja          "swallow|swallow_gpqa_ja"          "$GREEDY"

# ---- 翻訳 -----------------------------------------------------------------
run_task wmt20_enja               "swallow|wmt20:en-ja"              "$GREEDY"
run_task wmt20_jaen               "swallow|wmt20:ja-en"              "$GREEDY"

# ---- 数学 -----------------------------------------------------------------
for level in low medium high top; do
    run_task "polymath_ja_$level" "swallow|polymath_japanese:$level" "$GREEDY_LONG"
done
run_task aime_24                  "swallow|aime_N4:24"               "$SAMPLING"
run_task aime_25                  "swallow|aime_N4:25"               "$SAMPLING"

# ---- コード生成 -----------------------------------------------------------
run_task jhumaneval               "swallow|swallow_jhumaneval"       "$CODE"
run_task lcb_v6                   "swallow|lcb:codegeneration_v6"    "$SAMPLING"

# ---- MMLU-ProX（日本語）：91科目 -----------------------------------------
# 科目ごとにサブセットが分かれているため，1プロセスにまとめて実行する．
MMLU_PROX_TASKS=$("$LIGHTEVAL" tasks list 2>/dev/null \
    | grep -oE 'swallow\|mmlu_prox_japanese:[A-Za-z0-9_]+' | sort -u \
    | sed 's/$/|0|0/' | paste -sd,)
if [[ -n "$MMLU_PROX_TASKS" ]]; then
    if [[ -f "$STATE_DIR/mmlu_prox_japanese.done" ]]; then
        log "SKIP  mmlu_prox_japanese (already completed)"
    else
        log "START mmlu_prox_japanese ($(printf '%s' "$MMLU_PROX_TASKS" | tr ',' '\n' | wc -l) subsets)"
        started=$(date +%s)
        if "$LIGHTEVAL" endpoint litellm \
                "model=$MODEL_NAME,api_key=$OPENROUTER_API_KEY,generation_parameters={$GREEDY}" \
                "$MMLU_PROX_TASKS" \
                --use-chat-template \
                --output-dir "$OUT_DIR" \
                --save-details >"$LOG_DIR/mmlu_prox_japanese.log" 2>&1; then
            printf 'mmlu_prox_japanese (all subsets)\n' >"$STATE_DIR/mmlu_prox_japanese.done"
            log "OK    mmlu_prox_japanese ($(( $(date +%s) - started ))s)"
        else
            log "FAIL  mmlu_prox_japanese ($(( $(date +%s) - started ))s) -- see $LOG_DIR/mmlu_prox_japanese.log"
            printf 'mmlu_prox_japanese\n' >"$STATE_DIR/mmlu_prox_japanese.failed"
        fi
    fi
else
    log "FAIL  mmlu_prox_japanese -- could not enumerate the subsets"
fi

log "===== suite done ====="
log "completed: $(ls -1 "$STATE_DIR"/*.done 2>/dev/null | wc -l), failed: $(ls -1 "$STATE_DIR"/*.failed 2>/dev/null | wc -l)"
