#!/usr/bin/env bash
# 续跑：spiral_organs_r2 中 adrenal / aorta + 汇总表；ldct001 参考集 R2 + SAX 三方法 + 汇总表。
# 用法：bash scripts/resume_spiral_and_ldct001.sh
# 或：nohup bash scripts/resume_spiral_and_ldct001.sh > output/pipeline_resume.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python
SAX=/home/xielei/miniconda3/envs/sax_nerf/bin/python
GPU=0
free_gpu() { pkill -f ollama 2>/dev/null || true; }

log() { echo "[$(date -Iseconds)] $*"; }

log "=== adrenal train ==="
free_gpu
if [ ! -d output/spiral_organs_r2/adrenal_cone/point_cloud ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" train.py -s data/adrenal/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/adrenal_cone -m output/spiral_organs_r2/adrenal_cone
else
  log "skip adrenal train (already has point_cloud)"
fi
log "=== adrenal test ==="
free_gpu
CUDA_VISIBLE_DEVICES=$GPU "$PY" test.py -m output/spiral_organs_r2/adrenal_cone

log "=== aorta train ==="
free_gpu
if [ ! -d output/spiral_organs_r2/aorta_cone/point_cloud ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" train.py -s data/aorta/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/aorta_cone -m output/spiral_organs_r2/aorta_cone
else
  log "skip aorta train"
fi
log "=== aorta test ==="
free_gpu
CUDA_VISIBLE_DEVICES=$GPU "$PY" test.py -m output/spiral_organs_r2/aorta_cone

log "=== spiral organs table ==="
"$PY" scripts/collect_r2_results.py \
  --output_root output/spiral_organs_r2 \
  --cases abdomen_cone adrenal_cone aorta_cone \
  --csv_path output/spiral_organs_r2/benchmark_summary.csv \
  --md_path output/spiral_organs_r2/benchmark_summary.md

log "=== ldct001 R2 train ==="
mkdir -p output/ldct001_reference/r2_gaussian
free_gpu
if [ ! -d output/ldct001_reference/r2_gaussian/ldct_c001_cone/point_cloud ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" train.py -s data/LDCT-C001/synthetic_ref_nonspiral/ldct_c001_cone -m output/ldct001_reference/r2_gaussian/ldct_c001_cone
else
  log "skip ldct r2 train"
fi
log "=== ldct001 R2 test ==="
free_gpu
CUDA_VISIBLE_DEVICES=$GPU "$PY" test.py -m output/ldct001_reference/r2_gaussian/ldct_c001_cone
"$PY" scripts/collect_r2_results.py \
  --output_root output/ldct001_reference/r2_gaussian \
  --cases ldct_c001_cone \
  --csv_path output/ldct001_reference/r2_summary.csv \
  --md_path output/ldct001_reference/r2_summary.md

log "=== pickle for SAX ==="
"$PY" scripts/ours_to_naf_format_generic.py \
  --data_path data/LDCT-C001/synthetic_ref_nonspiral/ldct_c001_cone \
  --output_path data/LDCT-C001/synthetic_ref_nonspiral_naf/ldct_c001_cone.pickle
cp -f data/LDCT-C001/synthetic_ref_nonspiral_naf/ldct_c001_cone.pickle /home/xielei/3dgs/SAX-NeRF/data/ldct_c001_cone.pickle

log "=== SAX naf / intratomo / lineformer ==="
cd /home/xielei/3dgs/SAX-NeRF
free_gpu
CUDA_VISIBLE_DEVICES=$GPU "$SAX" train.py --config config/bench_organs/naf_ldct001_ref.yaml --gpu_id $GPU
free_gpu
CUDA_VISIBLE_DEVICES=$GPU "$SAX" train.py --config config/bench_organs/intratomo_ldct001_ref.yaml --gpu_id $GPU
free_gpu
CUDA_VISIBLE_DEVICES=$GPU "$SAX" train_mlg.py --config config/bench_organs/lineformer_ldct001_ref.yaml --gpu_id $GPU
"$SAX" scripts/collect_benchmark_results.py \
  --experiments naf_ldct001_ref intratomo_ldct001_ref lineformer_ldct001_ref \
  --csv_path Logs/ldct001_reference_summary.csv \
  --md_path Logs/ldct001_reference_summary.md

log "=== ALL DONE ==="
