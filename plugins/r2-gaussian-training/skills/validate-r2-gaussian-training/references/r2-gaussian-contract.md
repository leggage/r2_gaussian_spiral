# R2-Gaussian training contract

## Repository markers

The expected repository contains:

- `train.py`
- `r2_gaussian/arguments/__init__.py`
- `r2_gaussian/submodules/simple-knn`
- `r2_gaussian/submodules/xray-gaussian-rasterization-voxelization`

## Runtime contract

Training uses CUDA directly through `torch.cuda.Event`, CUDA tensors, and custom PyTorch extensions. A CPU-only PyTorch installation is therefore not a valid training environment.

The two importable extension modules are:

- `simple_knn._C`
- `xray_gaussian_rasterization_voxelization._C`

Validate both using the same Python executable that will launch `train.py`.

The repository environment file currently describes a legacy baseline of Python 3.9, PyTorch 1.12.1, torchvision 0.13.1, and CUDA Toolkit 11.6. Treat this as repository history, not proof that a different installed combination is invalid. Compatibility depends on the active PyTorch build, NVIDIA driver, toolkit used to compile extensions, host compiler, and extension ABI.

## Configuration behavior

`train.py` constructs argparse defaults first, then loads YAML and assigns every top-level YAML key into the parsed namespace. YAML therefore overrides CLI values in the current implementation, including values explicitly supplied on the command line.

The YAML loader supports `inherit_from`. The project implementation passes the inherited path directly to the loader, so its practical resolution depends on the process working directory.

Important parameters include:

- Paths: `source_path`, `model_path`, `ply_path`, `start_checkpoint`
- Run length: `iterations`
- Scale bounds: `scale_min`, `scale_max`
- Learning rates: position, density, scaling, and rotation initial/final/max-step values
- Loss: `lambda_dssim`, `lambda_tv`, `tv_vol_size`
- Adaptive control: `densify_from_iter`, `densify_until_iter`, `densification_interval`, `densify_grad_threshold`, `density_min_threshold`, `densify_scale_threshold`, `max_num_gaussians`
- Scheduling: `test_iterations`, `save_iterations`, `checkpoint_iterations`

## Dataset indicators

R2-Gaussian datasets normally expose `meta_data.json` in the source directory and may contain initialization arrays matching `init_*.npy` or a point cloud supplied through `ply_path`. Do not infer dataset validity merely from the directory existing.

## Remediation boundaries

Preflight is read-only. Installing packages, rebuilding CUDA extensions, changing CUDA symlinks, editing shell profiles, or deleting build artifacts requires an explicit fix request. Before rebuilding, record the active Python path, PyTorch version, `torch.version.cuda`, NVCC version, compiler version, GPU model, driver version, and the exact failing import traceback.
