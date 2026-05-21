#!/usr/bin/env python3
"""
LDCT two-dataset benchmark (ntrain=200): LDCT-L004 spiral + LDCT-C001 spiral.

- Trains / collects SAX (naf, intratomo, lineformer) on
  ``data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200`` into
  ``output/by_experiment/re_2organs_benchmark/{naf,intratomo,lineformer}/re_ldctl004_spiral_nt200/``.
- R2 Gaussian output lives at ``output/by_experiment/re_2organs_benchmark/r2_ldctl004``.
- Merges metrics with existing
  ``output/by_experiment/re_ldctc001_spiral_different_methods_ntrains`` for
  ``re_ldctc001_spiral_ntrain200`` (four algorithms).

Writes ``benchmark_2organs_ntrain200.{csv,md}`` under the benchmark root.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

R2_ROOT = Path(__file__).resolve().parents[1]
SAX_ROOT = Path(os.environ.get("SAX_NERF_ROOT", "/home/xielei/3dgs/SAX-NeRF"))
PY_R2 = os.environ.get("PY_R2", "/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python")
PY_SAX = os.environ.get("PY_SAX", "/home/xielei/miniconda3/envs/sax_nerf/bin/python")
TEMPLATE_INIT = R2_ROOT / "data_generator_usr/volume_gt/univeral_random_init.npy"

BENCH_ROOT = R2_ROOT / "output/by_experiment/re_2organs_benchmark"
L004_DATA = R2_ROOT / "data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200"
L004_R2_DIR = BENCH_ROOT / "r2_ldctl004"
L004_CASE_TAG = "re_ldctl004_spiral_nt200"
L004_PICKLE_NAME = f"{L004_CASE_TAG}.pickle"

C001_ROOT = R2_ROOT / "output/by_experiment/re_ldctc001_spiral_different_methods_ntrains"
C001_TAG = "re_ldctc001_spiral_ntrain200"

SAX_TEMPLATES = {
    "naf": SAX_ROOT / "config/bench_organs/naf_ldct001_spiral.yaml",
    "intratomo": SAX_ROOT / "config/bench_organs/intratomo_ldct001_spiral.yaml",
    "lineformer": SAX_ROOT / "config/bench_organs/lineformer_ldct001_spiral.yaml",
}
SAX_TRAIN_CMDS: dict[str, list[str]] = {
    "naf": ["train.py"],
    "intratomo": ["train.py"],
    "lineformer": ["train_mlg.py"],
}
SAX_GEN_L004 = SAX_ROOT / "config" / "_generated_ldct_l004_bench"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_vol_gt(case_path: Path) -> None:
    dst = case_path / "vol_gt.npy"
    if dst.is_file():
        return
    alt = case_path.parent / "ldct_l004_spiral_nt200_test" / "vol_gt.npy"
    if alt.is_file():
        shutil.copy2(alt, dst)
        print(f"[ensure_vol_gt] {alt} -> {dst}", flush=True)
        return
    raise FileNotFoundError(f"Missing {dst}")


def ensure_init(case_path: Path) -> None:
    case_name = case_path.name
    init_np = case_path / f"init_{case_name}.npy"
    if init_np.is_file():
        return
    run(
        [
            PY_R2,
            str(R2_ROOT / "scripts/init_fan_beam_volume_cone.py"),
            "--case",
            str(case_path),
            "--template_init",
            str(TEMPLATE_INIT),
            "--overwrite",
        ],
        cwd=R2_ROOT,
    )


def r2_test_metrics_ready(r2_dir: Path) -> bool:
    tr = r2_dir / "test"
    if not tr.is_dir():
        return False
    for p in tr.iterdir():
        if re.match(r"iter_(\d+)$", p.name) and (p / "eval3d.yml").is_file():
            return True
    return False


def _latest_r2_eval(r2_out: Path) -> tuple[dict, dict, str]:
    """Prefer ``test/``; fall back to ``eval/`` (training-time eval)."""
    eval3d: dict = {}
    eval2d: dict = {}
    tt = ""
    ttp = r2_out / "training_time_sec.txt"
    if ttp.is_file():
        tt = ttp.read_text(encoding="utf-8").strip()

    for sub in ("test", "eval"):
        root = r2_out / sub
        if not root.is_dir():
            continue
        iters = []
        for p in root.iterdir():
            m = re.match(r"iter_(\d+)$", p.name)
            if m and (p / "eval3d.yml").is_file():
                iters.append((int(m.group(1)), p))
        if not iters:
            continue
        iters.sort(key=lambda x: x[0])
        test_dir = iters[-1][1]
        ev3 = test_dir / "eval3d.yml"
        ev2 = test_dir / "eval2d_render_test.yml"
        if ev3.is_file():
            eval3d = yaml.safe_load(ev3.read_text(encoding="utf-8")) or {}
        if ev2.is_file():
            eval2d = yaml.safe_load(ev2.read_text(encoding="utf-8")) or {}
        break
    return eval3d, eval2d, tt


def latest_sax_run_dir(log_root: Path, expname: str) -> Path | None:
    pat = str(log_root / expname / "*")
    runs = sorted(glob.glob(pat), key=os.path.basename)
    if not runs:
        return None
    return Path(runs[-1])


def best_sax_eval_stats(run_dir: Path) -> tuple[int | None, dict[str, float] | None]:
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return None, None
    epoch_dirs = [p for p in eval_root.glob("epoch_*") if p.is_dir()]
    if not epoch_dirs:
        return None, None

    def ep_num(p: Path) -> int:
        m = re.match(r"epoch_(\d+)$", p.name)
        return int(m.group(1)) if m else -1

    best = max(epoch_dirs, key=ep_num)
    ep = ep_num(best)
    stats_path = best / "stats.txt"
    if not stats_path.is_file():
        return ep, None
    metrics: dict[str, float] = {}
    for line in stats_path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            metrics[k] = float(v)
        except ValueError:
            continue
    return ep, metrics


def read_training_sec_sax(run_dir: Path) -> float | None:
    p = run_dir / "training_time_sec.txt"
    if not p.is_file():
        return None
    try:
        return float(p.read_text().strip())
    except ValueError:
        return None


def write_sax_sidecar(dest: Path, expname: str, run_dir: Path) -> None:
    ep, metrics = best_sax_eval_stats(run_dir)
    tsec = read_training_sec_sax(run_dir)
    payload = {
        "experiment": expname,
        "run_dir": str(run_dir),
        "eval_epoch": ep,
        "metrics": metrics or {},
        "training_time_sec": tsec,
    }
    (dest / "sax_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_sax_config_l004(method: str, expname: str, pickle_basename: str) -> Path:
    tpl = SAX_TEMPLATES[method]
    cfg = yaml.safe_load(tpl.read_text(encoding="utf-8"))
    cfg["exp"]["expname"] = expname
    cfg["exp"]["datadir"] = f"./data/{pickle_basename}"
    SAX_GEN_L004.mkdir(parents=True, exist_ok=True)
    out_path = SAX_GEN_L004 / f"{expname}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def run_l004_sax_if_needed(gpu: int, methods: tuple[str, ...], env_base: dict) -> None:
    sax_pickle = SAX_ROOT / "data" / L004_PICKLE_NAME
    if not sax_pickle.is_file():
        run(
            [
                PY_R2,
                str(R2_ROOT / "scripts/ours_to_naf_format_generic.py"),
                "--data_path",
                str(L004_DATA),
                "--output_path",
                str(sax_pickle),
            ],
            cwd=R2_ROOT,
        )

    sax_path = str(Path(PY_SAX).parent) + os.pathsep + env_base.get("PATH", "")
    sax_env = env_base.copy()
    sax_env["PATH"] = sax_path
    if not (sax_env.get("TORCH_CUDA_ARCH_LIST") or "").strip():
        sax_env["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
            "SAX_TORCH_CUDA_ARCH_LIST",
            "8.0;8.6;8.9;12.0",
        )

    for method in methods:
        script = SAX_TRAIN_CMDS[method]
        train_cmd = [PY_SAX, *script]
        sub = BENCH_ROOT / method / L004_CASE_TAG
        sub.mkdir(parents=True, exist_ok=True)
        marker = sub / "sax_summary.json"
        if marker.is_file():
            print(f"[skip] SAX {method} {L004_CASE_TAG} (sax_summary.json exists)", flush=True)
            continue
        expname = f"{method}_{L004_CASE_TAG}"
        cfg_path = build_sax_config_l004(method, expname, L004_PICKLE_NAME)
        shutil.copy2(cfg_path, sub / "config.yaml")
        run(
            [
                *train_cmd,
                "--config",
                str(cfg_path.relative_to(SAX_ROOT)),
                "--gpu_id",
                str(gpu),
            ],
            cwd=SAX_ROOT,
            env=sax_env,
        )
        run_dir = latest_sax_run_dir(SAX_ROOT / "Logs", expname)
        if run_dir is None:
            raise RuntimeError(f"No SAX log dir for {expname}")
        write_sax_sidecar(sub, expname, run_dir)


def write_combined_benchmark_summary() -> None:
    rows: list[dict[str, object]] = []

    # LDCT-L004
    ev3, ev2, tt = _latest_r2_eval(L004_R2_DIR)
    rows.append(
        {
            "dataset": "LDCT-L004",
            "algorithm": "r2_gaussian",
            "ntrain": 200,
            "case": L004_CASE_TAG,
            "psnr_3d": ev3.get("psnr_3d", ""),
            "ssim_3d": ev3.get("ssim_3d", ""),
            "proj_psnr": ev2.get("psnr_2d", ""),
            "proj_ssim": ev2.get("ssim_2d", ""),
            "training_time_sec": tt,
        }
    )
    for method in ("naf", "intratomo", "lineformer"):
        js = BENCH_ROOT / method / L004_CASE_TAG / "sax_summary.json"
        if not js.is_file():
            rows.append(
                {
                    "dataset": "LDCT-L004",
                    "algorithm": method,
                    "ntrain": 200,
                    "case": L004_CASE_TAG,
                    "psnr_3d": "",
                    "ssim_3d": "",
                    "proj_psnr": "",
                    "proj_ssim": "",
                    "training_time_sec": "",
                }
            )
            continue
        summ = json.loads(js.read_text(encoding="utf-8"))
        mets = summ.get("metrics") or {}
        rows.append(
            {
                "dataset": "LDCT-L004",
                "algorithm": method,
                "ntrain": 200,
                "case": L004_CASE_TAG,
                "psnr_3d": mets.get("psnr_3d", ""),
                "ssim_3d": mets.get("ssim_3d", ""),
                "proj_psnr": mets.get("proj_psnr", ""),
                "proj_ssim": mets.get("proj_ssim", ""),
                "training_time_sec": summ.get("training_time_sec", ""),
            }
        )

    # LDCT-C001 (existing sweep)
    c1_r2 = C001_ROOT / "r2_gaussian" / C001_TAG
    ev3, ev2, tt = _latest_r2_eval(c1_r2)
    rows.append(
        {
            "dataset": "LDCT-C001",
            "algorithm": "r2_gaussian",
            "ntrain": 200,
            "case": C001_TAG,
            "psnr_3d": ev3.get("psnr_3d", ""),
            "ssim_3d": ev3.get("ssim_3d", ""),
            "proj_psnr": ev2.get("psnr_2d", ""),
            "proj_ssim": ev2.get("ssim_2d", ""),
            "training_time_sec": tt,
        }
    )
    for method in ("naf", "intratomo", "lineformer"):
        js = C001_ROOT / method / C001_TAG / "sax_summary.json"
        if not js.is_file():
            rows.append(
                {
                    "dataset": "LDCT-C001",
                    "algorithm": method,
                    "ntrain": 200,
                    "case": C001_TAG,
                    "psnr_3d": "",
                    "ssim_3d": "",
                    "proj_psnr": "",
                    "proj_ssim": "",
                    "training_time_sec": "",
                }
            )
            continue
        summ = json.loads(js.read_text(encoding="utf-8"))
        mets = summ.get("metrics") or {}
        rows.append(
            {
                "dataset": "LDCT-C001",
                "algorithm": method,
                "ntrain": 200,
                "case": C001_TAG,
                "psnr_3d": mets.get("psnr_3d", ""),
                "ssim_3d": mets.get("ssim_3d", ""),
                "proj_psnr": mets.get("proj_psnr", ""),
                "proj_ssim": mets.get("proj_ssim", ""),
                "training_time_sec": summ.get("training_time_sec", ""),
            }
        )

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = BENCH_ROOT / "benchmark_2organs_ntrain200.csv"
    md_path = BENCH_ROOT / "benchmark_2organs_ntrain200.md"
    fieldnames = [
        "dataset",
        "algorithm",
        "ntrain",
        "case",
        "psnr_3d",
        "ssim_3d",
        "proj_psnr",
        "proj_ssim",
        "training_time_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "| dataset | algorithm | ntrain | case | psnr_3d | ssim_3d | proj_psnr | proj_ssim | time_s |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {algorithm} | {ntrain} | {case} | {psnr_3d} | {ssim_3d} | "
            "{proj_psnr} | {proj_ssim} | {training_time_sec} |".format(**{k: r.get(k, "") for k in fieldnames})
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}\nWrote {md_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", "0")))
    ap.add_argument("--only", choices=("r2", "sax", "all", "summary"), default="all")
    ap.add_argument(
        "--sax_methods",
        nargs="+",
        choices=("naf", "intratomo", "lineformer"),
        default=["naf", "intratomo", "lineformer"],
    )
    args = ap.parse_args()

    if not L004_DATA.is_dir() or not (L004_DATA / "meta_data.json").is_file():
        raise FileNotFoundError(f"Missing L004 dataset: {L004_DATA}")

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.only == "summary":
        write_combined_benchmark_summary()
        return

    ensure_vol_gt(L004_DATA)
    ensure_init(L004_DATA)
    L004_R2_DIR.parent.mkdir(parents=True, exist_ok=True)

    if args.only in ("r2", "all"):
        if not r2_test_metrics_ready(L004_R2_DIR):
            run(
                [PY_R2, "test.py", "-m", str(L004_R2_DIR)],
                cwd=R2_ROOT,
                env=env_base,
            )
        else:
            print(f"[skip] R2 test metrics already present under {L004_R2_DIR / 'test'}", flush=True)

    if args.only in ("sax", "all"):
        run_l004_sax_if_needed(args.gpu, tuple(dict.fromkeys(args.sax_methods)), env_base)

    write_combined_benchmark_summary()


if __name__ == "__main__":
    main()
