---
name: summarize-r2-gaussian-results
description: Aggregate results from multiple datasets, acquisition modes, or reconstruction models into a publication-ready LaTeX comparison table. Use when users want an experiment table, benchmark summary, cross-model comparison, best-value highlighting, or LaTeX output from collected R2-Gaussian/NAF/FDK/INR metrics.
---

# Summarize R2-Gaussian Results

## Input validation

Read structured outputs produced by `collect-r2-gaussian-results` when available. Reconcile duplicate runs, metric definitions, evaluation splits, normalization, time units, and dataset labels before comparing them. Keep missing values as `/`; never convert them to zero.

## Table rules

- Group rows by dataset and organ/case; label groups clearly.
- Use method labels `<model>-<spiral|stitch>`.
- Default columns: `METHOD`, `PSNR$_{3D}$`, `SSIM$_{3D}$`, `PROJ_PSNR`, `PROJ_SSIM`, `TIME(min)`, and `LOSS`.
- Escape LaTeX special characters in labels and use valid math syntax.
- Highlight the best available value independently within each dataset group and metric column using `\best{...}`. Maximize PSNR/SSIM; minimize time. Do not highlight `/`, missing, invalid, or incomparable values.
- Preserve ties by marking every tied best value. Do not compare values from incompatible splits or definitions.
- State rounding precision and perform best selection on unrounded values when available.
- Include required package/macro notes: `booktabs`, `graphicx`, and a definition such as `\newcommand{\best}[1]{\textbf{#1}}` if the document does not already define it.

Start from `assets/results-table.tex`, adapt caption and grouping, and save both the `.tex` file and a source CSV/JSON beside it. Validate brace balance and, when a LaTeX engine is available, compile a minimal non-destructive preview and report any errors.
