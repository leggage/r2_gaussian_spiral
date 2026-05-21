#!/usr/bin/env python3
"""
Build a trainable r2_gaussian case from LDCT-L004 SPIRAL_processed (geometry JSON + .mat per view).

- Subsamples projection indices, symlinks only those ``.mat`` files, runs ``generate_data_usr.py``.
- ``vol_gt.npy`` is built from ``3DSLICES`` DICOM stack using the same pipeline as
  ``data_generator_usr/synthetic_dataset/process_raw_data.py`` (``process_dcm`` → normalize → ``reshape_vol``).
- Optional: write random Gaussian init via ``initialize_pcd.py --recon_method random`` (recommended for real data).

Example:
  python scripts/prepare_ldct_l004_real_dataset.py \\
    --output_case data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200 \\
    --n_total 700 --n_train 200 --n_test 300 \\
    --write_random_init --overwrite_init
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

R2_ROOT = Path(__file__).resolve().parents[1]


def _load_process_raw_data():
    path = R2_ROOT / "data_generator_usr/synthetic_dataset/process_raw_data.py"
    spec = importlib.util.spec_from_file_location("process_raw_data_usr", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_subset_geometry(
    full_geom: dict, indices: list[int], out_json: Path
) -> None:
    projs_full = full_geom["projections"]
    subset = {
        "scanner": full_geom["scanner"],
        "projections": [projs_full[i] for i in indices],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(subset, indent=2), encoding="utf-8")


def symlink_subset_mats(
    proj_src_dir: Path, stems: list[str], symlink_dir: Path
) -> None:
    if symlink_dir.is_dir():
        shutil.rmtree(symlink_dir)
    symlink_dir.mkdir(parents=True)
    for stem in stems:
        src = proj_src_dir / f"{stem}.mat"
        if not src.is_file():
            raise FileNotFoundError(f"Missing projection: {src}")
        dst = symlink_dir / f"{stem}.mat"
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def write_vol_gt_from_3dslices(
    case_dir: Path,
    slices_dir: Path,
    target_size: int,
) -> None:
    prd = _load_process_raw_data()
    if not slices_dir.is_dir():
        raise FileNotFoundError(f"3D slices directory not found: {slices_dir}")
    case_info = {
        "raw_path": str(slices_dir.resolve()),
        "output_name": "ldct_l004_3dslices",
        "file_type": "dcm",
        "thickness": None,
        "xy_invert": False,
    }
    vol = prd.process_dcm(case_info, int(target_size))
    vol = np.asarray(vol, dtype=np.float32)
    out_vol = case_dir / "vol_gt.npy"
    np.save(out_vol, vol)
    print(f"[3DSLICES] wrote {out_vol} shape={vol.shape} (process_dcm / reshape_vol)")


def run_fdk_volume(case_dir: Path, subset_geom_path: Path) -> None:
    import tigre.algorithms as algs

    sys.path.insert(0, str(R2_ROOT))
    from r2_gaussian.utils.ct_utils import get_geometry_tigre

    meta_path = case_dir / "meta_data.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scanner_cfg = meta["scanner"]
    mode = scanner_cfg["mode"]

    subset = json.loads(subset_geom_path.read_text(encoding="utf-8"))
    stems = [str(p["file_stem"]) for p in subset["projections"]]
    angles = np.array(
        [
            float(p["angle_rad"])
            if p.get("angle_rad") is not None
            else float(p["angle_deg"])
            for p in subset["projections"]
        ],
        dtype=np.float64,
    )

    projs_all = case_dir / "proj_all"
    stack = []
    for stem in stems:
        p = projs_all / f"{stem}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"Missing saved projection {p}")
        stack.append(np.load(p))
    projs = np.stack(stack, axis=0).astype(np.float32)

    geo = get_geometry_tigre(scanner_cfg)
    if mode == "fan":
        fdk = None
        for fn_name in ("fdk_fan", "FDK_fan", "FDK_Fan", "fdkFan"):
            fdk = getattr(algs, fn_name, None)
            if fdk is not None:
                vol = fdk(projs[:, ::-1, :], geo, angles)
                break
        if fdk is None:
            raise RuntimeError("TIGRE fan FDK not available")
    else:
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
    vol = np.asarray(vol, dtype=np.float32)
    vol = vol.transpose((2, 1, 0))
    vol = np.maximum(vol, 0.0)
    out_vol = case_dir / "vol_gt.npy"
    np.save(out_vol, vol)
    print(f"[FDK] wrote {out_vol} shape={vol.shape}")


def write_random_init(case_dir: Path, py_exe: str, gpu: int, overwrite: bool) -> None:
    case_name = case_dir.name
    init_path = case_dir / f"init_{case_name}.npy"
    if init_path.is_file():
        if not overwrite:
            print(f"[init] skip existing {init_path} (pass --overwrite_init)")
            return
        init_path.unlink()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        py_exe,
        str(R2_ROOT / "initialize_pcd.py"),
        "--data",
        str(case_dir),
        "--output",
        str(init_path),
        "--recon_method",
        "random",
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(R2_ROOT), env=env, check=True)
    print(f"[init] wrote {init_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--geometry_json",
        type=str,
        default=str(R2_ROOT / "data/LDCT-L004/SPIRAL_processed/scanner_geometry.json"),
    )
    ap.add_argument(
        "--proj_dir",
        type=str,
        default=str(R2_ROOT / "data/LDCT-L004/SPIRAL_processed/proj"),
    )
    ap.add_argument(
        "--slices_dir",
        type=str,
        default=str(R2_ROOT / "data/LDCT-L004/3DSLICES"),
        help="Folder of axial *.dcm for vol_gt (same convention as process_raw_data.process_dcm).",
    )
    ap.add_argument(
        "--output_case",
        type=str,
        default=str(R2_ROOT / "data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200"),
        help="Case directory: meta_data.json, proj_*, vol_gt.npy, init_*.npy.",
    )
    ap.add_argument("--n_total", type=int, default=700, help="Subsampled view count (>= n_train + n_test).")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--object_scale", type=int, default=50)
    ap.add_argument("--proj_rescale", type=float, default=400.0)
    ap.add_argument("--proj_subsample", type=int, default=1)
    ap.add_argument("--nVoxel", nargs=3, type=int, default=[256, 256, 256])
    ap.add_argument("--sVoxel", nargs=3, type=float, default=[20.0, 20.0, 20.0])
    ap.add_argument("--offOrigin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument(
        "--target_vol_size",
        type=int,
        default=None,
        help="Cube side for process_dcm resize; default = first --nVoxel entry.",
    )
    ap.add_argument(
        "--vol_source",
        choices=("slices", "fdk"),
        default="slices",
        help="How to build vol_gt.npy (default: 3DSLICES via process_dcm).",
    )
    ap.add_argument("--skip_symlink_generate", action="store_true")
    ap.add_argument("--skip_vol_gt", action="store_true")
    ap.add_argument(
        "--write_random_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run initialize_pcd.py with --recon_method random (default: on).",
    )
    ap.add_argument(
        "--overwrite_init",
        action="store_true",
        help="Replace existing init_*.npy when writing random init.",
    )
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", "0")))
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python for generate_data_usr / initialize_pcd.",
    )
    args = ap.parse_args()

    if args.n_train + args.n_test > args.n_total:
        raise ValueError("n_train + n_test must be <= n_total")

    target_vol = args.target_vol_size if args.target_vol_size is not None else int(args.nVoxel[0])

    geom_path = Path(args.geometry_json).resolve()
    proj_dir = Path(args.proj_dir).resolve()
    slices_dir = Path(args.slices_dir).resolve()
    case_dir = Path(args.output_case).resolve()
    work_dir = case_dir.parent / f".{case_dir.name}_work"
    subset_json = work_dir / "subset_geometry.json"
    symlink_dir = work_dir / "proj_mat_symlinks"

    full = json.loads(geom_path.read_text(encoding="utf-8"))
    n_full = len(full["projections"])
    if args.n_total > n_full:
        raise ValueError(f"n_total={args.n_total} exceeds available projections ({n_full}).")
    idx = np.unique(np.linspace(0, n_full - 1, args.n_total, dtype=int)).tolist()
    if len(idx) != args.n_total:
        raise RuntimeError(
            f"Subsample produced {len(idx)} unique indices for n_total={args.n_total}; "
            "reduce n_total or adjust subsampling."
        )
    stems = [str(full["projections"][i]["file_stem"]) for i in idx]

    case_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_symlink_generate:
        build_subset_geometry(full, idx, subset_json)
        symlink_subset_mats(proj_dir, stems, symlink_dir)
        gen_py = R2_ROOT / "data_generator_usr/real_dataset/generate_data_usr.py"
        cmd = [
            args.python,
            str(gen_py),
            "--data",
            str(symlink_dir),
            "--geometry_json",
            str(subset_json),
            "--output",
            str(case_dir),
            "--n_train",
            str(args.n_train),
            "--n_test",
            str(args.n_test),
            "--object_scale",
            str(int(args.object_scale)),
            "--proj_rescale",
            str(float(args.proj_rescale)),
            "--proj_subsample",
            str(args.proj_subsample),
            "--nVoxel",
            *[str(x) for x in args.nVoxel],
            "--sVoxel",
            *[str(x) for x in args.sVoxel],
            "--offOrigin",
            *[str(x) for x in args.offOrigin],
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(R2_ROOT), check=True)
    else:
        if not subset_json.is_file():
            raise FileNotFoundError(f"Missing {subset_json}; run without --skip_symlink_generate first")

    vol_path = case_dir / "vol_gt.npy"
    if not args.skip_vol_gt:
        if not (case_dir / "meta_data.json").is_file():
            raise FileNotFoundError(f"No meta_data.json under {case_dir}")
        if args.vol_source == "slices":
            write_vol_gt_from_3dslices(case_dir, slices_dir, target_vol)
        else:
            run_fdk_volume(case_dir, subset_json)
    elif not vol_path.is_file():
        raise FileNotFoundError(f"No {vol_path} and --skip_vol_gt set")

    if args.write_random_init:
        write_random_init(case_dir, args.python, args.gpu, args.overwrite_init)

    print(f"[OK] case ready: {case_dir}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    main()
