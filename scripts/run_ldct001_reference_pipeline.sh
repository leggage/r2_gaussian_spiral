#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_R2="${PYTHON_R2:-/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python}"
PYTHON_SAX="${PYTHON_SAX:-/home/xielei/miniconda3/envs/sax_nerf/bin/python}"
GPU_ID="${GPU_ID:-0}"

R2_CASE_DIR="data/LDCT-C001/synthetic_ref_nonspiral/ldct_c001_cone"
R2_MODEL_DIR="output/ldct001_reference/r2_gaussian/ldct_c001_cone"
R2_TABLE_DIR="output/ldct001_reference"
NAF_PATH="data/LDCT-C001/synthetic_ref_nonspiral_naf/ldct_c001_cone.pickle"
SAX_DATA_PATH="/home/xielei/3dgs/SAX-NeRF/data/ldct_c001_cone.pickle"
SAX_ROOT="/home/xielei/3dgs/SAX-NeRF"

try_free_gpu() {
  pkill -f ollama || true
}

if [[ ! -f "${R2_CASE_DIR}/meta_data.json" ]]; then
  "${PYTHON_R2}" data_generator/synthetic_dataset/generate_data.py \
    --vol data_generator_usr/volume_gt/ldct_c001.npy \
    --scanner data_generator_usr/synthetic_dataset/scanner/ldct001_nonspiral_cone_beam.yml \
    --output data/LDCT-C001/synthetic_ref_nonspiral \
    --n_train 200 \
    --n_test 300
fi

if [[ ! -f "${R2_CASE_DIR}/init_ldct_c001_cone.npy" ]]; then
  python3 scripts/init_fan_beam_volume_cone.py \
    --case "${R2_CASE_DIR}" \
    --template_init data_generator_usr/volume_gt/univeral_random_init.npy \
    --overwrite
fi

try_free_gpu
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_R2}" train.py -s "${R2_CASE_DIR}" -m "${R2_MODEL_DIR}"
try_free_gpu
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_R2}" test.py -m "${R2_MODEL_DIR}"

"${PYTHON_R2}" scripts/collect_r2_results.py \
  --output_root "output/ldct001_reference/r2_gaussian" \
  --cases ldct_c001_cone \
  --csv_path "${R2_TABLE_DIR}/r2_summary.csv" \
  --md_path "${R2_TABLE_DIR}/r2_summary.md"

"${PYTHON_R2}" scripts/ours_to_naf_format_generic.py \
  --data_path "${R2_CASE_DIR}" \
  --output_path "${NAF_PATH}"
cp "${NAF_PATH}" "${SAX_DATA_PATH}"

cd "${SAX_ROOT}"
try_free_gpu
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_SAX}" train.py --config config/bench_organs/naf_ldct001_ref.yaml --gpu_id "${GPU_ID}"
try_free_gpu
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_SAX}" train.py --config config/bench_organs/intratomo_ldct001_ref.yaml --gpu_id "${GPU_ID}"
try_free_gpu
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_SAX}" train_mlg.py --config config/bench_organs/lineformer_ldct001_ref.yaml --gpu_id "${GPU_ID}"

"${PYTHON_SAX}" scripts/collect_benchmark_results.py \
  --experiments naf_ldct001_ref intratomo_ldct001_ref lineformer_ldct001_ref \
  --csv_path Logs/ldct001_reference_summary.csv \
  --md_path Logs/ldct001_reference_summary.md

echo "LDCT001 reference pipeline finished."
