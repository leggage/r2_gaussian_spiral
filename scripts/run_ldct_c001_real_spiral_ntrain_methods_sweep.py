#!/usr/bin/env python3
"""
Train / eval r2_gaussian + SAX-NeRF (naf, intratomo, lineformer) on LDCT-C001 *real* spiral
datasets: data/LDCT-C001/real_dataset/re_spiral_ntrain{N}/ for N in {50,100,200,400,800,1000}.

Mirrors the layout of the synthetic sweep (e.g. syn_ldct001_spiral_different_methods_ntrains):

  output/by_experiment/re_ldctc001_spiral_different_methods_ntrains/
    r2_gaussian/re_ldctc001_spiral_ntrain200/
    naf/re_ldctc001_spiral_ntrain200/
    intratomo/...
    lineformer/...

Resume: skips R2 if latest test/*/eval3d.yml exists; skips SAX if sax_summary.json exists.

SAX Lineformer JIT: if the driver is not visible to PyTorch, set ``SAX_TORCH_CUDA_ARCH_LIST``
(e.g. ``8.9`` for Ada) so nvcc gets valid arch flags (see ``sax_env`` in ``main()``).
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

DEFAULT_NTRAINS = (50, 100, 200, 400, 800, 1000)
DEFAULT_SAX_METHODS = ("naf", "intratomo", "lineformer")
SAX_TRAIN_CMDS: dict[str, list[str]] = {
    "naf": ["train.py"],
    "intratomo": ["train.py"],
    "lineformer": ["train_mlg.py"],
}
R2_ROOT = Path(__file__).resolve().parents[1]
SAX_ROOT = Path(os.environ.get("SAX_NERF_ROOT", "/home/xielei/3dgs/SAX-NeRF"))
PY_R2 = os.environ.get("PY_R2", "/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python")
PY_SAX = os.environ.get("PY_SAX", "/home/xielei/miniconda3/envs/sax_nerf/bin/python")
DEFAULT_REAL_BASE = R2_ROOT / "data/LDCT-C001/real_dataset"
FOLDER_PREFIX = "re_spiral_ntrain"
TEMPLATE_INIT = R2_ROOT / "data_generator_usr/volume_gt/univeral_random_init.npy"
SAX_TEMPLATES = {
    "naf": SAX_ROOT / "config/bench_organs/naf_ldct001_spiral.yaml",
    "intratomo": SAX_ROOT / "config/bench_organs/intratomo_ldct001_spiral.yaml",
    "lineformer": SAX_ROOT / "config/bench_organs/lineformer_ldct001_spiral.yaml",
}
SAX_GEN_CONFIG_DIR = SAX_ROOT / "config" / "_generated_ldct001_real_ntrain_sweep"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def dataset_folder(n: int) -> str:
    return f"{FOLDER_PREFIX}{n}"


def exp_case_name(n: int) -> str:
    return f"re_ldctc001_spiral_ntrain{n}"


def case_data_dir(n: int, real_base: Path) -> Path:
    return real_base / dataset_folder(n)


def ensure_vol_gt(case_path: Path) -> None:
    """meta_data.json references vol_gt.npy; data generation may omit it. Copy from a sibling case."""
    dst = case_path / "vol_gt.npy"
    if dst.is_file():
        return
    parent = case_path.parent
    for name in (
        "re_spiral_ntrain50",
        "re_spiral_ntrain200",
        "re_spiral_ntrain1000",
    ):
        src = parent / name / "vol_gt.npy"
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"[ensure_vol_gt] {src} -> {dst}", flush=True)
            return
    raise FileNotFoundError(
        f"Missing {dst} and no reference vol_gt.npy under {parent} "
        "(expected one of re_spiral_ntrain50/200/1000)."
    )


def ensure_init(case_path: Path, py_r2: str) -> None:
    case_name = case_path.name
    init_np = case_path / f"init_{case_name}.npy"
    if init_np.is_file():
        return
    run(
        [
            py_r2,
            str(R2_ROOT / "scripts/init_fan_beam_volume_cone.py"),
            "--case",
            str(case_path),
            "--template_init",
            str(TEMPLATE_INIT),
            "--overwrite",
        ],
        cwd=R2_ROOT,
    )


def r2_eval_done(model_dir: Path) -> bool:
    test_root = model_dir / "test"
    if not test_root.is_dir():
        return False
    best = None
    for p in test_root.iterdir():
        m = re.match(r"iter_(\d+)$", p.name)
        if m:
            it = int(m.group(1))
            if (p / "eval3d.yml").is_file():
                best = it if best is None else max(best, it)
    return best is not None


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


def read_training_sec(run_dir: Path) -> float | None:
    p = run_dir / "training_time_sec.txt"
    if not p.is_file():
        return None
    try:
        return float(p.read_text().strip())
    except ValueError:
        return None


def write_sax_sidecar(dest: Path, expname: str, run_dir: Path) -> None:
    ep, metrics = best_sax_eval_stats(run_dir)
    tsec = read_training_sec(run_dir)
    payload = {
        "experiment": expname,
        "run_dir": str(run_dir),
        "eval_epoch": ep,
        "metrics": metrics or {},
        "training_time_sec": tsec,
    }
    (dest / "sax_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_sax_config(method: str, expname: str, pickle_name: str) -> Path:
    tpl = SAX_TEMPLATES[method]
    cfg = yaml.safe_load(tpl.read_text(encoding="utf-8"))
    cfg["exp"]["expname"] = expname
    cfg["exp"]["datadir"] = f"./data/{pickle_name}"
    SAX_GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAX_GEN_CONFIG_DIR / f"{expname}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def _latest_r2_eval(r2_out: Path) -> tuple[dict, dict, str]:
    eval3d: dict = {}
    eval2d: dict = {}
    tt = ""
    if not r2_out.is_dir():
        return eval3d, eval2d, tt
    tr = r2_out / "test"
    if not tr.is_dir():
        return eval3d, eval2d, tt
    iters = []
    for p in tr.iterdir():
        m = re.match(r"iter_(\d+)$", p.name)
        if m and (p / "eval3d.yml").is_file():
            iters.append((int(m.group(1)), p))
    if not iters:
        return eval3d, eval2d, tt
    iters.sort(key=lambda x: x[0])
    test_dir = iters[-1][1]
    ev3 = test_dir / "eval3d.yml"
    ev2 = test_dir / "eval2d_render_test.yml"
    if ev3.is_file():
        eval3d = yaml.safe_load(ev3.read_text(encoding="utf-8")) or {}
    if ev2.is_file():
        eval2d = yaml.safe_load(ev2.read_text(encoding="utf-8")) or {}
    ttp = r2_out / "training_time_sec.txt"
    if ttp.is_file():
        tt = ttp.read_text(encoding="utf-8").strip()
    return eval3d, eval2d, tt


def write_sweep_summary(output_root: Path, ntrains: tuple[int, ...]) -> None:
    rows: list[dict[str, object]] = []
    for n in ntrains:
        tag = exp_case_name(n)
        r2_out = output_root / "r2_gaussian" / tag
        eval3d, eval2d, tt = _latest_r2_eval(r2_out)
        rows.append(
            {
                "algorithm": "r2_gaussian",
                "ntrain": n,
                "case": tag,
                "psnr_3d": eval3d.get("psnr_3d", ""),
                "ssim_3d": eval3d.get("ssim_3d", ""),
                "proj_psnr": eval2d.get("psnr_2d", ""),
                "proj_ssim": eval2d.get("ssim_2d", ""),
                "training_time_sec": tt,
            }
        )
        for method in ("intratomo", "lineformer", "naf"):
            js = output_root / method / tag / "sax_summary.json"
            if not js.is_file():
                rows.append(
                    {
                        "algorithm": method,
                        "ntrain": n,
                        "case": tag,
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
                    "algorithm": method,
                    "ntrain": n,
                    "case": tag,
                    "psnr_3d": mets.get("psnr_3d", ""),
                    "ssim_3d": mets.get("ssim_3d", ""),
                    "proj_psnr": mets.get("proj_psnr", ""),
                    "proj_ssim": mets.get("proj_ssim", ""),
                    "training_time_sec": summ.get("training_time_sec", ""),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    master_csv = output_root / "summary_metrics.csv"
    master_md = output_root / "summary_metrics.md"
    fieldnames = [
        "algorithm",
        "ntrain",
        "case",
        "psnr_3d",
        "ssim_3d",
        "proj_psnr",
        "proj_ssim",
        "training_time_sec",
    ]
    with master_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "| algorithm | ntrain | case | psnr_3d | ssim_3d | proj_psnr | proj_ssim | time_s |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {algorithm} | {ntrain} | {case} | {psnr_3d} | {ssim_3d} | {proj_psnr} | {proj_ssim} | {training_time_sec} |".format(
                algorithm=r.get("algorithm", ""),
                ntrain=r.get("ntrain", ""),
                case=r.get("case", ""),
                psnr_3d=r.get("psnr_3d", ""),
                ssim_3d=r.get("ssim_3d", ""),
                proj_psnr=r.get("proj_psnr", ""),
                proj_ssim=r.get("proj_ssim", ""),
                training_time_sec=r.get("training_time_sec", ""),
            )
        )
    master_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {master_csv}")
    print(f"Wrote {master_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_root",
        type=str,
        default=str(
            R2_ROOT
            / "output/by_experiment/re_ldctc001_spiral_different_methods_ntrains"
        ),
        help="Top-level folder for this ablation.",
    )
    ap.add_argument(
        "--real_base",
        type=str,
        default=str(DEFAULT_REAL_BASE),
        help="Parent of re_spiral_ntrain* folders.",
    )
    ap.add_argument("--ntrains", type=int, nargs="+", default=list(DEFAULT_NTRAINS))
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", "0")))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only", choices=("r2", "sax", "all"), default="all")
    ap.add_argument(
        "--sax_methods",
        nargs="+",
        choices=("naf", "intratomo", "lineformer"),
        default=list(DEFAULT_SAX_METHODS),
        help="SAX trainers (order preserved).",
    )
    args = ap.parse_args()

    real_base = Path(args.real_base).resolve()

    output_root = Path(args.output_root).resolve()
    gpu = args.gpu
    py_r2 = PY_R2
    py_sax = PY_SAX
    ntrains = tuple(args.ntrains)

    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sax_path = str(Path(py_sax).parent) + os.pathsep + env_base.get("PATH", "")

    if args.dry_run:
        for n in ntrains:
            src = case_data_dir(n, real_base)
            tag = exp_case_name(n)
            ok = (src / "meta_data.json").is_file()
            print(f"[dry] n={n} src={src} tag={tag} exists={ok}")
        print("[dry] done.")
        return

    sax_method_list = tuple(dict.fromkeys(args.sax_methods))

    for n in ntrains:
        src = case_data_dir(n, real_base)
        if not (src / "meta_data.json").is_file():
            raise FileNotFoundError(f"Missing dataset: {src}")

        ensure_vol_gt(src)

        tag = exp_case_name(n)
        pickle_name = f"{tag}.pickle"
        sax_pickle = SAX_ROOT / "data" / pickle_name

        ensure_init(src, py_r2)

        r2_out = output_root / "r2_gaussian" / tag
        r2_out.parent.mkdir(parents=True, exist_ok=True)
        if args.only in ("all", "r2"):
            if not r2_eval_done(r2_out):
                run(
                    [py_r2, "train.py", "-s", str(src), "-m", str(r2_out)],
                    cwd=R2_ROOT,
                    env=env_base,
                )
                run(
                    [py_r2, "test.py", "-m", str(r2_out)],
                    cwd=R2_ROOT,
                    env=env_base,
                )
            else:
                print(f"[skip] R2 already evaluated: {r2_out}")

        if args.only == "r2":
            continue

        if not sax_pickle.is_file():
            run(
                [
                    py_r2,
                    str(R2_ROOT / "scripts/ours_to_naf_format_generic.py"),
                    "--data_path",
                    str(src),
                    "--output_path",
                    str(sax_pickle),
                ],
                cwd=R2_ROOT,
            )

        sax_env = env_base.copy()
        sax_env["PATH"] = sax_path
        # Lineformer JIT-compiles CUDA extensions. If PyTorch sees no GPU (or NVML fails),
        # _get_cuda_arch_flags() can be empty -> IndexError in torch.utils.cpp_extension.
        if not (sax_env.get("TORCH_CUDA_ARCH_LIST") or "").strip():
            sax_env["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
                "SAX_TORCH_CUDA_ARCH_LIST",
                "8.0;8.6;8.9;12.0",
            )

        for method in sax_method_list:
            script = SAX_TRAIN_CMDS[method]
            train_cmd = [py_sax, *script]
            sub = output_root / method / tag
            sub.mkdir(parents=True, exist_ok=True)
            marker = sub / "sax_summary.json"
            if marker.is_file():
                print(f"[skip] SAX {method} {tag} (sax_summary.json exists)")
                continue

            expname = f"{method}_{tag}"
            cfg_path = build_sax_config(method, expname, pickle_name)
            shutil.copy2(cfg_path, sub / "config.yaml")
            run(
                [*train_cmd, "--config", str(cfg_path.relative_to(SAX_ROOT)), "--gpu_id", str(gpu)],
                cwd=SAX_ROOT,
                env=sax_env,
            )
            run_dir = latest_sax_run_dir(SAX_ROOT / "Logs", expname)
            if run_dir is None:
                raise RuntimeError(f"No SAX log dir for {expname}")
            write_sax_sidecar(sub, expname, run_dir)

    write_sweep_summary(output_root, ntrains)


if __name__ == "__main__":
    main()
