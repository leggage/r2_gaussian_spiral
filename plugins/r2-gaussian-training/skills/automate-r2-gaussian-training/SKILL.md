---
name: automate-r2-gaussian-training
description: Automate the complete R2-Gaussian experiment workflow. Use whenever the user says they want to train, start/run an R2-Gaussian experiment, train one or many datasets/models, resume training, or asks for an end-to-end training pipeline. Coordinate explicit Conda environment selection, dataset preparation, visualization and confirmation, GPU-aware scheduling, mandatory preflight validation, live training progress, result collection, and optional LaTeX aggregation.
---

# Automate R2-Gaussian Training

## Workflow

1. Run `conda env list` or inspect available environment paths, then ask the user which Conda environment to use. Show an inferred default but require an explicit choice. Resolve its exact Python executable and use that same interpreter for preprocessing, preflight, visualization, training, evaluation, and result collection. In non-interactive shells, prefer the absolute `<conda-env>/bin/python` or `conda run -n <env>` over relying on `conda activate`.
2. Invoke `prepare-r2-gaussian-dataset`. Reuse an existing valid dataset or interactively generate it.
3. Offer dataset visualization with `<python> scripts/visualize_scene.py -s <dataset>`. This is a GUI process: request permission when needed, wait for the user to exit it, then explicitly ask whether the dataset is correct.
4. If the user rejects the dataset, return to preparation, inspect the reported problem, correct the configuration or inputs with approval, regenerate, and confirm again.
5. Discover GPUs with `nvidia-smi`. Ask whether to use multiple GPUs. For multiple independent tasks, default to one task per GPU and run tasks concurrently. Do not claim that a single `train.py` process supports distributed multi-GPU training without repository evidence.
6. Resolve each task's dataset, model, output, checkpoint, selected Python interpreter, and GPU. Use this output hierarchy:
   `output/<real|syn>/<organ>/<spiral|stitch>/ntrain<N>/<model>/`
   Preserve a user-specified output path when explicitly requested.
7. Invoke `validate-r2-gaussian-training` with the selected interpreter for every distinct training task. Do not start any task with `[FAIL]`; review every warning with the user when it may affect the run.
8. Show the exact commands and ask for final confirmation immediately before launching training. A typical R2GS command is:
   `CUDA_VISIBLE_DEVICES=<physical-gpu> <selected-python> train.py -s <dataset> -m <output>`
   When `CUDA_VISIBLE_DEVICES` contains one GPU, pass local CUDA device 0 to programs that also accept a device index unless repository behavior proves otherwise.
9. Start training only after confirmation. Keep a persistent execution session and a durable log. Do not use a detached process unless its survival has been verified. Call `scripts/progress.py --model-path <output> --total-iterations <N>` at least every 30-60 seconds while training is active and show its live progress line. If TensorBoard is unavailable, parse the durable log equivalently. For example:
   `[████░░░░░░░░░░░░░░░░] 6000/30000 20.0% | loss 0.031 | points 85000 | ETA 18m`
   Include current/total iteration, percentage, loss, Gaussian count, elapsed time, and ETA when observable. Recalculate ETA from recent throughput; label it approximate. Immediately report evaluation metrics, NaN/Inf, CUDA errors, OOM, or unexpected process exit. Do not end the active turn while the user asked to monitor until completion.
10. After each completed task, invoke `collect-r2-gaussian-results`.
11. For multiple datasets or models, ask whether to invoke `summarize-r2-gaussian-results` to produce one comparative LaTeX table.

## Interaction and safety

- Ask only for unresolved choices; inspect repository files first and offer inferred defaults.
- Treat preprocessing, visualization, training, and remediation as state-changing or interactive phases requiring the confirmations above.
- Never overwrite or mix artifacts in a non-empty output directory. Offer a new run name, explicit resume checkpoint, or user-approved reuse.
- Keep task names `model-spiral` or `model-stitch` in reports.
