#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_R2="${PY_R2:-/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python}"
PY_SAX="${PY_SAX:-/home/xielei/miniconda3/envs/sax_nerf/bin/python}"
GPU="${GPU:-0}"
TAG="spiral200_8circle"

R2_SRC="data/LDCT-C001/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/ldct_c001_cone"
R2_OUT="output/ldct001_spiral_compare/${TAG}/r2gaussian"
SAX_ROOT="/home/xielei/3dgs/SAX-NeRF"
OUT_TXT="output/ldct001_spiral_compare/${TAG}/eval.txt"

free_gpu() { pkill -f ollama || true; }
mkdir -p "output/ldct001_spiral_compare/${TAG}"

if [[ ! -f "${R2_OUT}/test/iter_30000/eval3d.yml" ]]; then
  free_gpu
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_R2" train.py -s "$R2_SRC" -m "$R2_OUT"
  free_gpu
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_R2" test.py -m "$R2_OUT"
fi

"$PY_R2" scripts/collect_r2_results.py \
  --output_root "output/ldct001_spiral_compare/${TAG}" \
  --cases r2gaussian \
  --csv_path "output/ldct001_spiral_compare/${TAG}/r2.csv" \
  --md_path "output/ldct001_spiral_compare/${TAG}/r2.md"

cd "$SAX_ROOT"
export PATH="$(dirname "$PY_SAX"):$PATH"

if [[ ! -f Logs/naf_ldct001_spiral/*/training_time_sec.txt ]]; then
  free_gpu
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_SAX" train.py --config config/bench_organs/naf_ldct001_spiral.yaml --gpu_id "$GPU"
fi
if [[ ! -f Logs/lineformer_ldct001_spiral/*/training_time_sec.txt ]]; then
  free_gpu
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_SAX" train_mlg.py --config config/bench_organs/lineformer_ldct001_spiral.yaml --gpu_id "$GPU"
fi
if [[ ! -f Logs/intratomo_ldct001_spiral/*/training_time_sec.txt ]]; then
  free_gpu
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_SAX" train.py --config config/bench_organs/intratomo_ldct001_spiral.yaml --gpu_id "$GPU"
fi

"$PY_SAX" scripts/collect_benchmark_results.py \
  --experiments naf_ldct001_spiral lineformer_ldct001_spiral intratomo_ldct001_spiral \
  --csv_path "Logs/ldct001_spiral_4way_summary.csv" \
  --md_path "Logs/ldct001_spiral_4way_summary.md"

cd "$ROOT"
python3 - <<'PY2'
import csv
from pathlib import Path

root = Path("output/ldct001_spiral_compare/spiral200_8circle")
r2_csv = root / "r2.csv"
sax_csv = Path("/home/xielei/3dgs/SAX-NeRF/Logs/ldct001_spiral_4way_summary.csv")
out = root / "eval.txt"

lines = []
if r2_csv.exists():
    with r2_csv.open() as f:
        for r in csv.DictReader(f):
            lines += [
                "ldct001_spiral_r2gaussian:",
                f"psnr_2d: {r.get('psnr_2d_test', '')}",
                f"ssim_2d: {r.get('ssim_2d_test', '')}",
                f"psnr_3d: {r.get('psnr_3d', '')}",
                f"ssim_3d: {r.get('ssim_3d', '')}",
                f"time: {r.get('training_time_sec', '')}",
                "",
            ]
if sax_csv.exists():
    with sax_csv.open() as f:
        for r in csv.DictReader(f):
            exp = r["experiment"]
            lines += [
                f"ldct001_spiral_{exp}:",
                f"proj_psnr: {r.get('proj_psnr', '')}",
                f"proj_ssim: {r.get('proj_ssim', '')}",
                f"psnr_3d: {r.get('psnr_3d', '')}",
                f"ssim_3d: {r.get('ssim_3d', '')}",
                f"time: {r.get('training_time_sec', '')}",
                "",
            ]
out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(f"Saved {out}")
PY2

echo "All done. Results: $OUT_TXT"
