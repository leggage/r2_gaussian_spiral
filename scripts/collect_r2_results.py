import argparse
import csv
import os
import os.path as osp

import yaml


def read_yaml(path):
    if not osp.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def read_time_sec(model_dir):
    p = osp.join(model_dir, "training_time_sec.txt")
    if not osp.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return ""


def latest_iter_test_dir(model_dir):
    test_root = osp.join(model_dir, "test")
    if not osp.isdir(test_root):
        return ""
    cands = []
    for name in os.listdir(test_root):
        if name.startswith("iter_"):
            try:
                cands.append((int(name.split("_")[1]), osp.join(test_root, name)))
            except Exception:
                pass
    if not cands:
        return ""
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--md_path", required=True, type=str)
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        model_dir = osp.join(args.output_root, case)
        test_dir = latest_iter_test_dir(model_dir)
        eval3d = read_yaml(osp.join(test_dir, "eval3d.yml")) if test_dir else {}
        eval2d = read_yaml(osp.join(test_dir, "eval2d_render_test.yml")) if test_dir else {}
        rows.append(
            {
                "case": case,
                "psnr_3d": eval3d.get("psnr_3d", ""),
                "ssim_3d": eval3d.get("ssim_3d", ""),
                "psnr_2d_test": eval2d.get("psnr_2d", ""),
                "ssim_2d_test": eval2d.get("ssim_2d", ""),
                "training_time_sec": read_time_sec(model_dir),
                "model_dir": model_dir,
                "test_dir": test_dir,
            }
        )

    os.makedirs(osp.dirname(args.csv_path), exist_ok=True)
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "psnr_3d", "ssim_3d", "psnr_2d_test", "ssim_2d_test", "training_time_sec", "model_dir", "test_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Case | PSNR3D | SSIM3D | PSNR2D(Test) | SSIM2D(Test) | Time(s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['psnr_3d']} | {r['ssim_3d']} | {r['psnr_2d_test']} | {r['ssim_2d_test']} | {r['training_time_sec']} |"
        )
    with open(args.md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved CSV: {args.csv_path}")
    print(f"Saved MD: {args.md_path}")


if __name__ == "__main__":
    main()
