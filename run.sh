#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results_tafi_final}"
SEEDS="${SEEDS:-0,1,2,3,4}"
LOG_EVERY="${LOG_EVERY:-10}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV:-oodv1}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --seeds "$SEEDS"
  --device "$DEVICE"
  --log-every "$LOG_EVERY"
)

run_default() {
  local name="$1"
  shift
  echo "=== Running default configuration: ${name} ==="
  "${PYTHON_CMD[@]}" run_tafi.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR/default/${name}" \
    "$@"
}

run_default "GOODCBAS" \
  --datasets GOODCBAS \
  --good-shifts concept,covariate

run_default "GOODCora_degree" \
  --datasets GOODCora \
  --good-domain degree \
  --good-shifts concept,covariate

run_default "GOODCora_word_concept" \
  --datasets GOODCora \
  --good-domain word \
  --good-shifts concept

run_default "GOODWebKB_university_concept" \
  --datasets GOODWebKB \
  --good-domain university \
  --good-shifts concept

run_default "other_datasets" \
  --datasets ogbn_arxiv,twitch,Elliptic

echo "=== Running tuned configuration: GOODCora/word/covariate ==="
"${PYTHON_CMD[@]}" run_tafi.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR/tuned/GOODCora_word_covariate" \
  --datasets GOODCora \
  --good-domain word \
  --good-shifts covariate \
  --epochs 50 \
  --patience 10 \
  --lr-predictor 0.003 \
  --lr-augmenter 0.001 \
  --lr-protection-generator 0.003 \
  --predictor-weight-decay 0.0001 \
  --augmenter-weight-decay 0.0001 \
  --protection-generator-weight-decay 0.0001 \
  --inner-steps 1 \
  --alpha-div 0.01 \
  --predictor-hidden-dim 256 \
  --predictor-num-layers 3 \
  --num-envs 3 \
  --perturb-budget-fraction 0.05 \
  --protection-hidden-dim 128 \
  --protection-dropout 0.5 \
  --max-edge-attenuation 1.0 \
  --predictor-dropout 0.7

echo "=== Running tuned configuration: GOODWebKB/university/covariate ==="
"${PYTHON_CMD[@]}" run_tafi.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR/tuned/GOODWebKB_university_covariate" \
  --datasets GOODWebKB \
  --good-domain university \
  --good-shifts covariate \
  --epochs 400 \
  --patience 400 \
  --lr-predictor 0.00005 \
  --lr-augmenter 0.0003 \
  --lr-protection-generator 0.00005 \
  --predictor-weight-decay 0.0002 \
  --augmenter-weight-decay 0.0002 \
  --protection-generator-weight-decay 0.0002 \
  --inner-steps 4 \
  --alpha-div 0.005 \
  --predictor-hidden-dim 384 \
  --predictor-num-layers 2 \
  --num-envs 3 \
  --perturb-budget-fraction 0.02 \
  --protection-hidden-dim 128 \
  --protection-dropout 0.5 \
  --max-edge-attenuation 0.4 \
  --predictor-dropout 0.2
