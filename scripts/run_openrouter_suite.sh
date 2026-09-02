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
# タスクIDには few-shot 数の指定（|0|0）を自動で付ける．
run_task() {
    local label="$1"; shift
    local task_id="$1"; shift
    run_task_raw "$label" "${task_id}|0|0" "$@"
}

# run_task_raw <ラベル> <タスク指定> <生成パラメータ> [追加の実行時引数...]
# タスク指定は lighteval にそのまま渡す（複数タスクのカンマ区切りも可）．
run_task_raw() {
    local label="$1"; shift
    local task_spec="$1"; shift
    local gen_params="$1"; shift

    # SUITE_INCLUDE / SUITE_EXCLUDE で実行するベンチマークを絞れる（拡張正規表現）．
    # 別のプロセスで並行して実行したい場合に使う．
    if [[ -n "${SUITE_INCLUDE:-}" ]] && ! [[ "$label" =~ $SUITE_INCLUDE ]]; then
        return 0
    fi
    if [[ -n "${SUITE_EXCLUDE:-}" ]] && [[ "$label" =~ $SUITE_EXCLUDE ]]; then
        return 0
    fi

    local done_marker="$STATE_DIR/$label.done"
    if [[ -f "$done_marker" ]]; then
        log "SKIP  $label (already completed)"
        return 0
    fi

    # 実行中マーカー．複数のプロセスで並行実行しても同じベンチマークを
    # 二重に走らせないようにする（このスクリプトを複数起動できる）．
    local running_marker="$STATE_DIR/$label.running"
    if [[ -f "$running_marker" ]]; then
        log "SKIP  $label (another process is running it: $(cat "$running_marker"))"
        return 0
    fi
    printf 'pid=%s since=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$running_marker"
    # このプロセスが終了したら実行中マーカーを消す．
    trap 'rm -f "$running_marker"' RETURN

    local model_args="model=$MODEL_NAME,api_key=$OPENROUTER_API_KEY"
    if [[ -n "$gen_params" ]]; then
        model_args="$model_args,generation_parameters={$gen_params}"
    fi

    local task_log="$LOG_DIR/$label.log"
    log "START $label -> $task_spec"
    local started
    started=$(date +%s)

    if "$LIGHTEVAL" endpoint litellm \
            "$model_args" \
            "$task_spec" \
            --use-chat-template \
            --output-dir "$OUT_DIR" \
            --save-details \
            "$@" >"$task_log" 2>&1; then
        local elapsed=$(( $(date +%s) - started ))
        printf '%s\n' "$task_spec" >"$done_marker"
        rm -f "$STATE_DIR/$label.failed"
        log "OK    $label (${elapsed}s)"
    else
        local elapsed=$(( $(date +%s) - started ))
        log "FAIL  $label (${elapsed}s) -- see $task_log"
        printf '%s\n' "$task_spec" >"$STATE_DIR/$label.failed"
    fi
    rm -f "$running_marker"
}

# 非推論型モデル向けの生成パラメータ（BENCHMARKS.md の推奨設定）．
GREEDY="temperature:0.0,max_new_tokens:4096"
GREEDY_LONG="temperature:0.0,max_new_tokens:8192"
CODE="temperature:0.2,top_p:0.95,max_new_tokens:4096"
SAMPLING="temperature:0.6,top_p:0.95,max_new_tokens:8192"
# MT-Bench はカテゴリごとの temperature が自動適用されるため temperature を指定しない．
MT_BENCH="max_new_tokens:4096"

# 推論型モデルを評価する場合は REASONING=1 を指定する．
# BENCHMARKS.md の「Swallow LLM Leaderboard との関係」に従い，
# 全ベンチマークで確率的デコーディング（temperature=0.6, top_p=0.95）を用いる
# （非推論型と異なり，ベンチマークごとの推奨設定より
#   モデル種別の方針が優先される旨が記載されている）．
#
# 出力トークン数の上限は非推論型より大幅に大きくする．推論型モデルは
# 推論過程にトークンを消費するため，上限が小さいと推論の途中で打ち切られ，
# 最終回答が空のまま不正解と判定されてしまう．
# （実測：簡単な算数1問でも推論だけで1,700〜2,200トークンを消費する）
if [[ "${REASONING:-0}" == "1" ]]; then
    GREEDY="temperature:0.6,top_p:0.95,max_new_tokens:32768"
    GREEDY_LONG="temperature:0.6,top_p:0.95,max_new_tokens:32768"
    CODE="temperature:0.6,top_p:0.95,max_new_tokens:32768"
    SAMPLING="temperature:0.6,top_p:0.95,max_new_tokens:32768"
    MT_BENCH="temperature:0.6,top_p:0.95,max_new_tokens:32768"
fi

# MAX_NEW_TOKENS で全ベンチマークの出力トークン数の上限をまとめて上書きできる．
# 上限に到達して出力が打ち切られると，回答が途中で切れて不正解と判定されるため，
# モデルに合わせて引き上げたい場合に使う．
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
    for _preset in GREEDY GREEDY_LONG CODE SAMPLING MT_BENCH; do
        printf -v "$_preset" '%s' "$(sed -E "s/max_new_tokens:[0-9]+/max_new_tokens:${MAX_NEW_TOKENS}/" <<<"${!_preset}")"
    done
    unset _preset
fi

log "===== suite start: model=$MODEL_NAME judge=$JUDGE_MODEL out=$OUT_DIR reasoning=${REASONING:-0} ====="

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
    run_task_raw mmlu_prox_japanese "$MMLU_PROX_TASKS" "$GREEDY"
else
    log "FAIL  mmlu_prox_japanese -- could not enumerate the subsets"
fi

log "===== suite done ====="
log "completed: $(ls -1 "$STATE_DIR"/*.done 2>/dev/null | wc -l), failed: $(ls -1 "$STATE_DIR"/*.failed 2>/dev/null | wc -l)"
