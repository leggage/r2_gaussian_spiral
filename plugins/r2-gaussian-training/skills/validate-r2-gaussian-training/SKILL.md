---
name: validate-r2-gaussian-training
description: Run a safe preflight validation before R2-Gaussian training. Use when Codex is asked to start, prepare, debug, or verify an R2-Gaussian training run; inspect a training command or YAML configuration; check CUDA, GPU, PyTorch, NVCC, compiled CUDA extensions, dataset paths, checkpoints, output paths, or hyperparameter consistency; or diagnose an environment failure before launching train.py.
---

# Validate R2-Gaussian Training

## Workflow

1. Locate the repository root by finding `train.py` and `r2_gaussian/arguments/__init__.py`.
2. Identify the intended command, YAML configuration, source dataset, output directory, checkpoint, and GPU. Do not start training during preflight.
3. Run the bundled validator from the target repository root:

   ```bash
   python <skill-dir>/scripts/preflight.py --repo . [--config CONFIG] [--source-path DATA] [--model-path OUTPUT] [--checkpoint FILE] [--gpu INDEX]
   ```

4. Treat an exit code of 2 as a blocking failure. Resolve every `[FAIL]` before training.
5. Review `[WARN]` findings individually. Do not claim warnings are harmless without evidence from the intended run.
6. If the check passes, show the exact proposed training command and summarize the selected GPU, CUDA/PyTorch versions, configuration, source path, output path, and warnings.
7. Start training only when the user asked to start it. A request to check or diagnose does not authorize a training run.

## Safety rules

- Keep preflight read-only. Do not compile extensions, install packages, overwrite outputs, or allocate a large CUDA tensor unless the user explicitly requests remediation or a stress test.
- Never hide a CUDA mismatch by changing `PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`, compiler flags, or symlinks silently.
- Prefer the Python interpreter from the intended training command. A successful check in another environment is not sufficient.
- Inspect inherited YAML files when `inherit_from` is used. Resolve relative inheritance paths from the repository working directory, matching the current project loader.
- Flag an existing non-empty model directory because training may mix or overwrite experiment artifacts.
- For environment or build remediation, read `references/r2-gaussian-contract.md` before proposing changes.

## Parameter interpretation

- Require `source_path` and `model_path` after merging CLI overrides with YAML.
- Require positive `iterations`, learning rates, `tv_vol_size`, densification interval, Gaussian limit, and scale bounds.
- Require `scale_min < scale_max` when both bounds are enabled.
- Require `densify_from_iter < densify_until_iter <= iterations` when densification is active.
- Require `position_lr_final <= position_lr_init`, `density_lr_final <= density_lr_init`, and `scaling_lr_final <= scaling_lr_init` unless the user explicitly intends an increasing schedule.
- Warn when testing, saving, or checkpoint iterations fall outside the training range.
- Verify the source directory and expected dataset metadata. Verify checkpoint readability when resuming.

## Reporting

Report in this order:

1. `PASS`, `PASS WITH WARNINGS`, or `FAIL`
2. Blocking failures
3. Warnings
4. Confirmed environment and resolved parameters
5. Exact next command, if safe

Distinguish facts observed by the script from inferred recommendations.
