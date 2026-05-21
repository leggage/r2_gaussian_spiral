#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_R2="${PYTHON_R2:-/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT_ROOT="${OUT_ROOT:-output/spiral_organs_r2}"

cases=(
  "abdomen:data/abdomen/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/abdomen_cone"
  "adrenal:data/adrenal/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/adrenal_cone"
  "aorta:data/aorta/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/aorta_cone"
)

try_free_gpu() {
  pkill -f ollama || true
}

mkdir -p "${OUT_ROOT}"

for item in "${cases[@]}"; do
  name="${item%%:*}"
  src="${item#*:}"
  model_dir="${OUT_ROOT}/${name}_cone"
  try_free_gpu
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_R2}" train.py -s "${src}" -m "${model_dir}"
  try_free_gpu
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_R2}" test.py -m "${model_dir}"
done

"${PYTHON_R2}" scripts/collect_r2_results.py \
  --output_root "${OUT_ROOT}" \
  --cases abdomen_cone adrenal_cone aorta_cone \
  --csv_path "${OUT_ROOT}/benchmark_summary.csv" \
  --md_path "${OUT_ROOT}/benchmark_summary.md"

echo "R2 spiral organ benchmark finished: ${OUT_ROOT}"
