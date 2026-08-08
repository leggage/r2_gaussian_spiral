---
name: prepare-r2-gaussian-dataset
description: Interactively select, configure, generate, validate, visualize, and troubleshoot normalized datasets for R2-Gaussian and comparison models. Use when preparing synthetic or real CT data, choosing organ/spiral/stitch/ntrain/model parameters, creating a norm_pipeline YAML, running data_preprocess/norm_pipeline.py, or checking whether an existing dataset is usable.
---

# Prepare R2-Gaussian Dataset

## Gather inputs

Inspect existing datasets and configs first, then resolve:

- Conda environment and its exact Python executable. If the parent training workflow has not already resolved one, list available environments and ask the user to choose before running validation or generation.
- Dataset type: `syn` or `real`. Set `scanner.coord_left: false` for synthetic and `true` for real unless the user explicitly supplies a justified override.
- Target model: `r2gs`, `naf`, `fdk`, or another supported model.
- Existing dataset path, or raw inputs. Synthetic requires `raw_gt`; real requires `raw_gt` and `raw_proj`.
- Organ/case name, acquisition type (`spiral`, `stitch`, or both), `n_train`, `n_test`, seed, scanner geometry, spiral settings, and initialization settings.
- For stitch, collect its independent `n_train` and `n_test` and set `stitch.enabled: true`.

Ask in small groups and propose values inferred from repository examples. Never invent scanner geometry.

## Naming and configuration

Store every generated dataset in this mandatory hierarchy:

`data/<real|syn>/<organ>/<spiral|stitch>/ntrain<N>/<model>/`

For example:

`data/real/chest/spiral/ntrain400/r2gs/`

Set `output_root: data`; the pipeline appends dataset type, organ, acquisition, ntrain, and model. Do not create new flat dataset directories. Preserve existing legacy datasets in place unless the user explicitly requests migration.

Create every run-specific YAML under `data_preprocess/configs/`, based on `data_preprocess/configs/norm_pipeline.example.yml`. Create that directory when missing. Use a descriptive file name such as `real_chest_spiral_ntrain400_r2gs.yml`. Do not place experiment YAML files directly under `data_preprocess/`.

## Generate and validate

1. For an existing dataset, verify directory readability plus `meta_data.json`, initialization `*.npy`, and expected train/test projections. Check `vol_gt.npy` when the dataset is expected to have ground truth.
2. For a new configuration, first run:
   `<selected-python> data_preprocess/norm_pipeline.py --config data_preprocess/configs/<config.yml> --validate-only`
3. Show validation findings and the exact generation command. Obtain confirmation before the potentially expensive generation step.
4. Run:
   `<selected-python> data_preprocess/norm_pipeline.py --config data_preprocess/configs/<config.yml>`
5. Verify outputs and record the resolved dataset path and parameters. Do not overwrite an existing dataset without explicit approval.

## Visualization handoff

Offer:

`python scripts/visualize_scene.py -s <dataset>`

The command is interactive and requires a display. Request the required permission, wait for exit, and ask whether the scene is correct. If rejected, gather the observed symptom, inspect configuration/data, and return to the relevant preparation step.
