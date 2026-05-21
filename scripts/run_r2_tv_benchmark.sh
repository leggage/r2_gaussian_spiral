#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_R2="${PYTHON_R2:-/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python}"
GPU_ID="${GPU_ID:-0}"
LAMBDA_TV="${LAMBDA_TV:-0.05}"
TAG="tv${LAMBDA_TV}"
OUT_ROOT="${OUT_ROOT:-output/r2_benchmark_${TAG}}"

try_free_gpu() {
  pkill -f ollama || true
}

cases=(
  "ldct001_ref:data/LDCT-C001/ldct001_synthetic_nospiral/ldct_c001_cone"
  "abdomen:data/abdomen/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/abdomen_cone"
  "adrenal:data/adrenal/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/adrenal_cone"
  "aorta:data/aorta/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/aorta_cone"
  "cat_real_dataset:data/LDCT-C001/cat/real_dataset"
)

mkdir -p "$OUT_ROOT"
for item in "${cases[@]}"; do
  name="${item%%:*}"
  src="${item#*:}"
  model_dir="${OUT_ROOT}/${name}_${TAG}"
  echo "[RUN] ${name} -> ${model_dir}"
  try_free_gpu
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON_R2" train.py -s "$src" -m "$model_dir" --lambda_tv "$LAMBDA_TV"
  try_free_gpu
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON_R2" test.py -m "$model_dir"
done

"$PYTHON_R2" scripts/collect_r2_results.py   --output_root "$OUT_ROOT"   --cases ldct001_ref_${TAG} abdomen_${TAG} adrenal_${TAG} aorta_${TAG} cat_real_dataset_${TAG}   --csv_path "$OUT_ROOT/benchmark_${TAG}.csv"   --md_path "$OUT_ROOT/benchmark_${TAG}.md"

"$PYTHON_R2" scripts/export_r2_eval_txt.py   --summary_csv "$OUT_ROOT/benchmark_${TAG}.csv"   --output_txt "$OUT_ROOT/eval_${TAG}.txt"

echo "Done. Outputs in $OUT_ROOT"
