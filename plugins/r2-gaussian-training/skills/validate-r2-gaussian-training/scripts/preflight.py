#!/usr/bin/env python3
"""Read-only R2-Gaussian training preflight check."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.info.append(message)

    def emit(self) -> int:
        for message in self.info:
            print(f"[INFO] {message}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.failures:
            print(f"[FAIL] {message}")
        if self.failures:
            print(f"RESULT: FAIL ({len(self.failures)} blocking, {len(self.warnings)} warnings)")
            return 2
        if self.warnings:
            print(f"RESULT: PASS WITH WARNINGS ({len(self.warnings)} warnings)")
        else:
            print("RESULT: PASS")
        return 0


def load_yaml(path: Path, repo: Path, report: Report, seen: set[Path] | None = None) -> dict:
    try:
        import yaml
    except Exception as exc:
        report.fail(f"Cannot import PyYAML: {exc}")
        return {}
    seen = seen or set()
    path = path if path.is_absolute() else repo / path
    path = path.resolve()
    if path in seen:
        report.fail(f"Configuration inheritance cycle at {path}")
        return {}
    if not path.is_file():
        report.fail(f"Configuration file does not exist: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        report.fail(f"Cannot parse configuration {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        report.fail(f"Configuration root must be a mapping: {path}")
        return {}
    parent = data.pop("inherit_from", None)
    merged: dict = {}
    if parent:
        merged.update(load_yaml(Path(str(parent)), repo, report, seen | {path}))
    merged.update(data)
    report.note(f"Loaded configuration: {path}")
    return merged


def run_text(command: list[str]) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output if output else None


def check_positive(cfg: dict, report: Report, names: list[str]) -> None:
    for name in names:
        value = cfg.get(name)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            report.fail(f"{name} must be positive, got {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="R2-Gaussian repository root")
    parser.add_argument("--config", help="Training YAML path")
    parser.add_argument("--source-path", help="Override source_path for validation")
    parser.add_argument("--model-path", help="Override model_path for validation")
    parser.add_argument("--checkpoint", help="Override start_checkpoint for validation")
    parser.add_argument("--gpu", type=int, default=0, help="Visible CUDA device index")
    args = parser.parse_args()

    report = Report()
    repo = Path(args.repo).expanduser().resolve()
    report.note(f"Python: {sys.executable} ({sys.version.split()[0]})")
    report.note(f"Repository: {repo}")
    for marker in ("train.py", "r2_gaussian/arguments/__init__.py"):
        if not (repo / marker).is_file():
            report.fail(f"Missing repository marker: {marker}")

    cfg: dict = {}
    if args.config:
        cfg.update(load_yaml(Path(args.config), repo, report))
    # These represent the intended effective values for preflight. Note that the
    # current train.py lets YAML override CLI; callers should avoid conflicting values.
    if args.source_path:
        cfg["source_path"] = args.source_path
    if args.model_path:
        cfg["model_path"] = args.model_path
    if args.checkpoint:
        cfg["start_checkpoint"] = args.checkpoint

    for key in ("source_path", "model_path"):
        if not cfg.get(key):
            report.fail(f"Missing required effective parameter: {key}")

    source = Path(str(cfg.get("source_path", ""))).expanduser()
    if source and not source.is_absolute():
        source = repo / source
    if cfg.get("source_path"):
        if not source.is_dir():
            report.fail(f"source_path is not a directory: {source}")
        elif not (source / "meta_data.json").is_file():
            report.warn(f"Dataset metadata not found: {source / 'meta_data.json'}")
        else:
            try:
                metadata = json.loads((source / "meta_data.json").read_text(encoding="utf-8"))
                report.note(f"Dataset metadata is valid JSON with keys: {', '.join(sorted(metadata)[:12])}")
            except Exception as exc:
                report.fail(f"Invalid dataset metadata JSON: {exc}")

    model = Path(str(cfg.get("model_path", ""))).expanduser()
    if model and not model.is_absolute():
        model = repo / model
    if cfg.get("model_path") and model.exists():
        if not model.is_dir():
            report.fail(f"model_path exists but is not a directory: {model}")
        else:
            try:
                if any(model.iterdir()):
                    report.warn(f"model_path already exists and is non-empty: {model}")
            except OSError as exc:
                report.fail(f"Cannot inspect model_path {model}: {exc}")

    checkpoint = cfg.get("start_checkpoint")
    if checkpoint:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = repo / checkpoint_path
        if not checkpoint_path.is_file() or not os.access(checkpoint_path, os.R_OK):
            report.fail(f"Checkpoint is not a readable file: {checkpoint_path}")

    check_positive(cfg, report, [
        "iterations", "position_lr_init", "position_lr_final", "density_lr_init",
        "density_lr_final", "scaling_lr_init", "scaling_lr_final", "rotation_lr_init",
        "rotation_lr_final", "tv_vol_size", "densification_interval",
        "densify_grad_threshold", "density_min_threshold", "max_num_gaussians",
    ])
    for name in ("lambda_dssim", "lambda_tv"):
        value = cfg.get(name)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            report.fail(f"{name} must be non-negative, got {value!r}")
    if isinstance(cfg.get("lambda_dssim"), (int, float)) and cfg["lambda_dssim"] > 1:
        report.warn(f"lambda_dssim is unusually greater than 1: {cfg['lambda_dssim']}")

    scale_min, scale_max = cfg.get("scale_min"), cfg.get("scale_max")
    if isinstance(scale_min, (int, float)) and isinstance(scale_max, (int, float)):
        if scale_min <= 0 or scale_max <= 0 or scale_min >= scale_max:
            report.fail(f"Require 0 < scale_min < scale_max, got {scale_min}, {scale_max}")
    iterations = cfg.get("iterations")
    densify_from, densify_until = cfg.get("densify_from_iter"), cfg.get("densify_until_iter")
    if all(isinstance(v, int) for v in (iterations, densify_from, densify_until)):
        if not (0 <= densify_from < densify_until <= iterations):
            report.fail("Require 0 <= densify_from_iter < densify_until_iter <= iterations")
    for prefix in ("position", "density", "scaling", "rotation"):
        initial, final = cfg.get(f"{prefix}_lr_init"), cfg.get(f"{prefix}_lr_final")
        if isinstance(initial, (int, float)) and isinstance(final, (int, float)) and final > initial:
            report.warn(f"{prefix}_lr_final ({final}) is greater than {prefix}_lr_init ({initial})")
    if isinstance(iterations, int):
        for key in ("test_iterations", "save_iterations", "checkpoint_iterations"):
            values = cfg.get(key)
            if isinstance(values, list):
                outside = [v for v in values if not isinstance(v, int) or v < 1 or v > iterations]
                if outside:
                    report.warn(f"{key} contains values outside 1..{iterations}: {outside}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        report.note(f"CUDA_VISIBLE_DEVICES={visible}")
    smi = shutil.which("nvidia-smi")
    if smi:
        succeeded, output = run_text([smi, "--query-gpu=index,name,driver_version,memory.total,memory.free", "--format=csv,noheader"])
        if succeeded and output:
            report.note("nvidia-smi GPUs: " + output.replace("\n", " | "))
        else:
            detail = f": {output.replace(chr(10), ' | ')}" if output else ""
            report.warn("nvidia-smi could not query the NVIDIA driver" + detail)
    else:
        report.warn("nvidia-smi is not available on PATH")
    nvcc = shutil.which("nvcc")
    if nvcc:
        succeeded, output = run_text([nvcc, "--version"])
        if succeeded and output:
            report.note("NVCC: " + output.splitlines()[-1])
        else:
            report.warn(f"nvcc --version failed: {output or 'no output'}")
    else:
        report.warn("nvcc is not available on PATH; this blocks extension rebuilds, not imports of existing builds")

    try:
        import torch
        report.note(f"PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
        if not torch.cuda.is_available():
            report.fail("torch.cuda.is_available() is false")
        else:
            count = torch.cuda.device_count()
            report.note(f"PyTorch visible CUDA devices: {count}")
            if args.gpu < 0 or args.gpu >= count:
                report.fail(f"GPU index {args.gpu} is outside the visible range 0..{count - 1}")
            else:
                props = torch.cuda.get_device_properties(args.gpu)
                report.note(f"Selected GPU {args.gpu}: {props.name}; capability {props.major}.{props.minor}; {props.total_memory / 2**30:.1f} GiB")
                try:
                    tensor = torch.ones(1, device=f"cuda:{args.gpu}")
                    if tensor.item() != 1:
                        report.fail("Minimal CUDA tensor result was incorrect")
                    else:
                        report.note("Minimal CUDA tensor check passed")
                except Exception as exc:
                    report.fail(f"Minimal CUDA tensor check failed: {exc}")
    except Exception as exc:
        report.fail(f"Cannot import PyTorch: {exc}")

    sys.path.insert(0, str(repo))
    for module in ("simple_knn._C", "xray_gaussian_rasterization_voxelization._C"):
        try:
            importlib.import_module(module)
            report.note(f"CUDA extension import passed: {module}")
        except Exception as exc:
            report.fail(f"CUDA extension import failed for {module}: {type(exc).__name__}: {exc}")

    if cfg:
        selected = {key: cfg.get(key) for key in (
            "source_path", "model_path", "iterations", "scale_min", "scale_max",
            "densify_from_iter", "densify_until_iter", "max_num_gaussians",
        ) if key in cfg}
        report.note("Selected parameters: " + json.dumps(selected, ensure_ascii=False, sort_keys=True))
    return report.emit()


if __name__ == "__main__":
    raise SystemExit(main())
