#!/bin/bash
# ベンチマーク実行プロセスの監視・自動復旧．
#
# 使い方：
#   OPENROUTER_API_KEY=sk-or-... [HF_TOKEN=hf_...] ./scripts/supervisor.sh &
#
# 環境変数：
#   SWALLOW_EVAL_REPO       リポジトリのパス（既定：このスクリプトの親ディレクトリ）
#   SUITE_SCRIPT            使用する suite スクリプト（既定：scripts/run_openrouter_suite.sh）
#   SUPERVISOR_INTERVAL     確認の間隔（秒，既定60）
#   SUPERVISOR_MAX_PROCS    lighteval プロセスの上限（既定8）
#   SUPERVISOR_MEM_FLOOR_MB この空きメモリを下回ったら新規起動を控える（既定3200）
#                           MT-Bench は判定処理のために lighteval 自身を4本ほど
#                           fork するため，1プロセスあたり最大で6GB近く必要に
#                           なることがある．上限と下限はこれを見込んだ値．
#   SUPERVISOR_STALL_TIMEOUT ログがこの秒数更新されないベンチマークを停止する（既定2700）
#   SUPERVISOR_SLOTS_FILE   スロット定義ファイル．1行1スロットで
#                           <モデルID>|<出力ディレクトリ名>|<種類(main|mmlu)>|<必要本数>|<追加生成パラメータ>
#                           指定しない場合はスクリプト内の既定値を使う．
#
# 「出力ディレクトリ×種類（main / mmlu）」ごとに，動いていてほしいドライバ
# （suite スクリプト）の本数を定義し，足りなければ起動し直す．
# 完了済み（state/.complete.* が存在）の種類は起動しない．
#
# ドライバの種類は /proc/<pid>/environ の SUITE_INCLUDE / SUITE_EXCLUDE で
# 判定する．同じ出力ディレクトリに複数のドライバを立てるため，コマンドライン
# だけでは区別できない．本数で管理しているため，この監視プロセスの導入前に
# 手で起動したドライバも正しく数えられる．
#
# 注意：この監視プロセス自体もコンテナが再起動すると死ぬ．コンテナ再起動から
# 復旧するには，サーバ側の定期実行（Routine）から本スクリプトを起動し直す．
# 冪等なので何度実行しても安全．
set -u

REPO="${SWALLOW_EVAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SNAP="${SUITE_SCRIPT:-$REPO/scripts/run_openrouter_suite.sh}"
SNAP_NAME=$(basename "$SNAP")
LOG=$REPO/runs/supervisor.log
LOCK=$REPO/runs/supervisor.pid
INTERVAL="${SUPERVISOR_INTERVAL:-60}"
MAX_PROCS="${SUPERVISOR_MAX_PROCS:-8}"
MEM_FLOOR_MB="${SUPERVISOR_MEM_FLOOR_MB:-3200}"

cd "$REPO" || exit 1

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY を設定してください}"
mkdir -p runs

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }

# 二重起動の防止．
if [[ -f "$LOCK" ]]; then
    old_pid=$(cat "$LOCK" 2>/dev/null)
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        log "supervisor already running (pid $old_pid); exiting"
        exit 0
    fi
    log "removing stale lock (pid $old_pid is gone)"
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

# スロット定義：<モデルID>|<出力ディレクトリ名>|<種類>|<必要本数>|<追加生成パラメータ>
# 上から順に優先して起動する．main を優先するのは，個別のベンチマークを
# 消化してプロセス枠を早く空けるため．
SLOTS=(
  "deepseek/deepseek-v4-flash-0731|deepseek-v4-flash|main|1|"
  "qwen/qwen3.8-27b|qwen3.8-27b|main|1|"
  "moonshotai/kimi-k2.6|kimi-k2.6|main|1|"
  'google/gemma-4-31b-it|gemma-4-31b-it-reasoning|main|1|reasoning_effort:"medium"'
  "deepseek/deepseek-v4-flash-0731|deepseek-v4-flash|mmlu|1|"
  "qwen/qwen3.8-27b|qwen3.8-27b|mmlu|1|"
  "moonshotai/kimi-k2.6|kimi-k2.6|mmlu|1|"
  'google/gemma-4-31b-it|gemma-4-31b-it-reasoning|mmlu|1|reasoning_effort:"medium"'
)

# スロット定義を外部ファイルで上書きできるようにする（空行と # から始まる行は無視）．
if [[ -n "${SUPERVISOR_SLOTS_FILE:-}" ]]; then
    mapfile -t SLOTS < <(grep -vE '^\s*(#|$)' "$SUPERVISOR_SLOTS_FILE")
    log "loaded ${#SLOTS[@]} slots from $SUPERVISOR_SLOTS_FILE"
fi

# 指定した出力ディレクトリ・種類のドライバの本数を数える．
count_drivers() {
    local out_dir="$1" kind="$2" pid environ n=0
    for pid in $(pgrep -f "$SNAP_NAME $out_dir" 2>/dev/null); do
        environ=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null) || continue
        if [[ "$kind" == mmlu ]]; then
            grep -qx 'SUITE_INCLUDE=mmlu_prox' <<<"$environ" && n=$(( n + 1 ))
        else
            grep -qx 'SUITE_EXCLUDE=mmlu_prox' <<<"$environ" && n=$(( n + 1 ))
        fi
    done
    printf '%s' "$n"
}

# その種類が完了しているか（完了マーカーがあるか）．
kind_complete() {
    local out_name="$1" kind="$2"
    if [[ "$kind" == mmlu ]]; then
        compgen -G "runs/$out_name/state/.complete.mmlu*" >/dev/null 2>&1
    else
        [[ -f "runs/$out_name/state/.complete.main" ]]
    fi
}

available_mb() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
# ドライバが起動した lighteval の本数．
# lighteval は判定処理やデータセット取得のために自身を fork するため，
# pgrep でそのまま数えると本数が実際の数倍になり，起動枠の判定を誤る．
# 親が lighteval でないもの（＝ドライバ直下のもの）だけを数える．
n_lighteval() {
    local pid ppid n=0
    for pid in $(pgrep -f 'lighteval endpoint litellm' 2>/dev/null); do
        ppid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null) || continue
        [[ -n "$ppid" ]] || continue
        grep -q 'lighteval' "/proc/$ppid/cmdline" 2>/dev/null && continue
        n=$(( n + 1 ))
    done
    printf '%s' "$n"
}

start_slot() {
    local model="$1" out_name="$2" kind="$3" extra="$4" tag="$5"
    local dir="./runs/$out_name"
    mkdir -p "$dir"
    (
      export OPENROUTER_API_KEY
      export HF_TOKEN="${HF_TOKEN:-}"
      export REASONING=1 MAX_NEW_TOKENS=32768 MMLU_PROX_CHUNK=8
      export LITELLM_CONCURRENT_CALLS=32 JUDGE_CONCURRENT_CALLS=24 JFBENCH_MAX_CONCURRENCY=24
      export MODEL_NAME="openrouter/$model"
      export SUITE_PART="$tag"
      [[ -n "$extra" ]] && export EXTRA_GEN_PARAMS="$extra"
      if [[ "$kind" == main ]]; then
          export SUITE_EXCLUDE='mmlu_prox'
      else
          export SUITE_INCLUDE='mmlu_prox'
      fi
      setsid nohup "$SNAP" "$dir" >>"$dir/driver_$tag.out" 2>&1 </dev/null &
    )
    log "STARTED $out_name/$tag (model=$model kind=$kind)"
}

# 実行中のベンチマークのログが一定時間更新されない場合，そのプロセスを停止する．
# lighteval が内部で使う multiprocessing のワーカーが OOM で殺されると，
# 親プロセスが futex で永久に待ち続け，ログの更新が止まったまま何時間も
# 居座ることがある（実際に MT-Bench で2時間の無音ハングが発生した）．
# ドライバは停止を検知して FAIL を記録し，次のベンチマークに進む．
# 失敗が残っている限り完了マーカーは作られないため，後で再実行される．
STALL_TIMEOUT="${SUPERVISOR_STALL_TIMEOUT:-2700}"

kill_stalled_tasks() {
    local marker label out_name logfile age drv pid child now
    now=$(date +%s)
    for marker in runs/*/state/*.running; do
        [[ -f "$marker" ]] || continue
        label=$(basename "$marker" .running)
        out_name=$(basename "$(dirname "$(dirname "$marker")")")
        logfile="runs/$out_name/logs/$label.log"
        [[ -f "$logfile" ]] || continue
        age=$(( now - $(stat -c %Y "$logfile" 2>/dev/null || echo "$now") ))
        (( age > STALL_TIMEOUT )) || continue

        drv=$(sed -n 's/^pid=\([0-9]*\).*/\1/p' "$marker")
        [[ -n "$drv" ]] && kill -0 "$drv" 2>/dev/null || continue
        for pid in $(pgrep -P "$drv" 2>/dev/null); do
            log "STALLED $out_name/$label (log untouched for ${age}s) -- killing lighteval pid $pid"
            for child in $(pgrep -P "$pid" 2>/dev/null); do kill -9 "$child" 2>/dev/null; done
            kill -9 "$pid" 2>/dev/null
        done
    done
    return 0
}

log "===== supervisor start (pid $$, interval ${INTERVAL}s, max_procs $MAX_PROCS, mem_floor ${MEM_FLOOR_MB}MB) ====="

while true; do
    kill_stalled_tasks
    for slot in "${SLOTS[@]}"; do
        IFS='|' read -r model out_name kind want extra <<<"$slot"

        kind_complete "$out_name" "$kind" && continue

        have=$(count_drivers "./runs/$out_name" "$kind")
        (( have >= want )) && continue

        procs=$(n_lighteval)
        (( procs >= MAX_PROCS )) && continue

        mem=$(available_mb)
        if (( mem < MEM_FLOOR_MB )); then
            log "HOLD  $out_name/$kind (available ${mem}MB < ${MEM_FLOOR_MB}MB, have=$have want=$want)"
            continue
        fi

        log "RECOVER $out_name/$kind (have=$have want=$want procs=$procs mem=${mem}MB)"
        start_slot "$model" "$out_name" "$kind" "$extra" "${kind}$(( have + 1 ))"
        sleep 20   # 起動直後のメモリ確保を待ってから次のスロットを見る
    done
    sleep "$INTERVAL"
done
