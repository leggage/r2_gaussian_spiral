import argparse
import glob
import os
import os.path as osp
import shutil
import subprocess
import sys


DEFAULT_CASES = [
    "data/abdomen/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/abdomen_cone",
    "data/adrenal/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/adrenal_cone",
    "data/aorta/syn_dataset/syn_spiral_ntrain200_8circle_pjzspan24/aorta_cone",
]


def _resolve_cases(args):
    if args.case:
        return [osp.abspath(p) for p in args.case]

    if args.glob_pattern:
        return [osp.abspath(p) for p in sorted(glob.glob(args.glob_pattern))]

    return [osp.abspath(p) for p in DEFAULT_CASES]


def main():
    parser = argparse.ArgumentParser(
        description="Initialize fan-beam/cone datasets with init_<case_name>.npy naming."
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case folder path. Repeat this flag for multiple cases.",
    )
    parser.add_argument(
        "--glob_pattern",
        default="",
        type=str,
        help="Glob pattern to discover case folders (optional).",
    )
    parser.add_argument("--device", default=0, type=int, help="GPU device index.")
    parser.add_argument("--recon_method", default="fdk", type=str)
    parser.add_argument("--n_points", default=50000, type=int)
    parser.add_argument("--density_thresh", default=0.05, type=float)
    parser.add_argument("--density_rescale", default=0.15, type=float)
    parser.add_argument("--random_density_max", default=1.0, type=float)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing init_*.npy files.",
    )
    parser.add_argument(
        "--template_init",
        default="",
        type=str,
        help="If provided, copy this init npy to each case as init_<case_name>.npy.",
    )
    args = parser.parse_args()

    case_paths = _resolve_cases(args)
    if not case_paths:
        raise ValueError("No case folders found.")

    for case_path in case_paths:
        if not osp.isdir(case_path):
            raise FileNotFoundError(f"Case folder does not exist: {case_path}")
        meta_path = osp.join(case_path, "meta_data.json")
        if not osp.exists(meta_path):
            raise FileNotFoundError(f"meta_data.json not found: {meta_path}")

        case_name = osp.basename(case_path)
        output_path = osp.join(case_path, f"init_{case_name}.npy")
        if osp.exists(output_path) and not args.overwrite:
            print(f"[Skip] {output_path} already exists.")
            continue
        if args.template_init:
            template_init = osp.abspath(args.template_init)
            if not osp.exists(template_init):
                raise FileNotFoundError(f"Template init file not found: {template_init}")
            shutil.copy2(template_init, output_path)
            print(f"[Copy] {template_init} -> {output_path}")
            continue

        cmd = [
            sys.executable,
            "initialize_pcd.py",
            "--data",
            case_path,
            "--output",
            output_path,
            "--recon_method",
            str(args.recon_method),
            "--n_points",
            str(args.n_points),
            "--density_thresh",
            str(args.density_thresh),
            "--density_rescale",
            str(args.density_rescale),
            "--random_density_max",
            str(args.random_density_max),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"[Run] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        print(f"[Done] {output_path}")


if __name__ == "__main__":
    main()
