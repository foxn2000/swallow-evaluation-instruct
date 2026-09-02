#!/bin/bash
# OpenRouter のモデルを評価対象・ジャッジの双方に使う実行例．
#
# OpenRouter は OpenAI 互換の Chat Completion API を提供しているため，
# LiteLLM バックエンドでそのまま評価できる．
#   * 評価対象モデル：MODEL_ARGS の model に "openrouter/{プロバイダ}/{モデルID}" を指定する．
#   * ジャッジモデル：環境変数 JUDGE_MODEL_NAME / JUDGE_BASE_URL / JUDGE_API_KEY を指定する．
#
# GPUを必要としないため，vLLM を含まない軽量な extra（lighteval_api）で実行できる．
set -euo pipefail

# ---- 設定 ----------------------------------------------------------------
MODEL_NAME="openrouter/google/gemma-4-31b-it"
TASK_ID="${1:-swallow|japanese_mt_bench}"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY を設定してください}"

# LLM-as-a-Judge の設定．OpenRouter のモデルをジャッジに使う．
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_MODEL_NAME="openai/gpt-5.4-mini"

# 推論APIの最大並列数．
export LITELLM_CONCURRENT_CALLS="${LITELLM_CONCURRENT_CALLS:-16}"
export JUDGE_CONCURRENT_CALLS="${JUDGE_CONCURRENT_CALLS:-16}"

# ---- 実行 ----------------------------------------------------------------
uv run --isolated --locked --extra lighteval_api \
    lighteval endpoint litellm \
        "model=$MODEL_NAME,api_key=$OPENROUTER_API_KEY,generation_parameters={temperature:0.0,max_new_tokens:4096}" \
        "${TASK_ID}|0|0" \
        --use-chat-template \
        --output-dir ./lighteval/outputs \
        --save-details
