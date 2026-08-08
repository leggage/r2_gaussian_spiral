---
name: collect-r2-gaussian-results
description: Collect and validate experiment results for R2-Gaussian and comparison methods. Use after or during experiments to extract 2D projection PSNR/SSIM, 3D PSNR/SSIM, training time in minutes, model/acquisition labels, final reconstruction profile figures, and metric-versus-time convergence curves from output JSON, logs, TensorBoard events, and reconstructed volumes.
---

# Collect R2-Gaussian Results

## Required record

For each run, produce one structured record containing:

- dataset type and organ/case
- acquisition (`spiral` or `stitch`), `ntrain`, model, and label `<model>-<acquisition>`
- PSNR 3D, SSIM 3D, projection PSNR 2D, projection SSIM 2D
- training time in minutes
- loss definition when known
- source paths for every value and status (`observed`, `derived`, or `missing`)

Never substitute training loss for an evaluation metric. Preserve missing real-world 3D metrics as `/` when no trustworthy volume ground truth exists.

## Collection order

1. Inspect the model output, evaluation JSON files, config snapshots, logs, checkpoints, reconstructed volumes, and TensorBoard event files.
2. Prefer repository collectors such as `scripts/collect_r2_results.py` when compatible. Cross-check keys: `psnr_3d`, `ssim_3d`, `psnr_2d`, `ssim_2d`, and `training_time_sec`. Convert seconds to minutes once.
3. For convergence curves, read TensorBoard scalars (including reconstruction metrics and iteration time). Plot each requested metric against cumulative elapsed minutes, not merely iteration, when timing data exists. Clearly label an iteration-based fallback.
4. Locate the final reconstructed volume. Use `scripts/plot_volume.py` for the profile/cutaway figure, but inspect it first because the current repository version may contain hard-coded input/output paths. Patch or parameterize it only with user authorization; preserve the original source and report the generated image path.
5. Validate numeric finiteness, units, evaluation split, final iteration, and correspondence between model output and dataset. Flag inconsistent or ambiguous evidence.

Write machine-readable CSV or JSON plus a concise human summary in the experiment results directory. Do not fabricate unavailable values or silently recompute metrics with different normalization.

