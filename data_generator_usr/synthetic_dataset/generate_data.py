import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import plotly.graph_objects as go
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import json
import matplotlib.pyplot as plt
import tigre.algorithms as algs
import argparse
import open3d as o3d
import cv2
import pickle
import copy

import sys

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume


def _build_spiral_offsets(num_views, spiral_cfg):
    """
    Compute per-view z offsets for a helical/spiral trajectory.
    
    Modified version: Z offsets are determined solely by:
      - z_start: initial z offset
      - z_end: final z offset
      - num_views: total number of projections
      
    The z step is automatically computed as: z_delta_per_view = (z_end - z_start) / (num_views - 1)
    This ensures that z offsets are linearly distributed between z_start and z_end,
    keeping all projections within the volume bounds.
    
    If z_delta_per_view is explicitly provided (for backward compatibility), it will be used instead.
    """
    if not spiral_cfg:
        return None
    if spiral_cfg.get("enabled", True) is False:
        return None

    if num_views == 0:
        return None

    start = float(spiral_cfg.get("z_start", 0.0))
    
    # Check if z_end is provided (new way) or z_delta_per_view (old way)
    if "z_end" in spiral_cfg:
        end = float(spiral_cfg.get("z_end", 0.0))
        # Compute step to ensure endpoints match exactly
        step = (end - start) / max(1, num_views - 1) if num_views > 1 else 0.0
    else:
        # Backward compatibility: use z_delta_per_view if provided
        step = float(spiral_cfg.get("z_delta_per_view", 0.0))

    # Z offsets determined only by projection index, not by angle
    increments = np.arange(num_views, dtype=np.float32)
    return start + step * increments


# def _geo_with_spiral_offsets(base_geo, z_offsets):
#     """
#     Create a copy of the TIGRE geometry whose offOrigin follows the provided z offsets.
#     TIGRE expects the offOrigin ordering to be [z, y, x] and shape (3, N) where each column
#     corresponds to one projection.
#     """
#     if z_offsets is None:
#         return base_geo

#     spiral_geo = copy.deepcopy(base_geo)
#     base_origin = np.asarray(base_geo.offOrigin)
#     # Normalize to shape (3, N)
#     if base_origin.ndim == 1:
#         # [z, y, x] -> (3, 1)
#         base_origin = base_origin.reshape(-1, 1)
#     elif base_origin.shape[0] != 3 and base_origin.shape[1] == 3:
#         # (N, 3) -> transpose to (3, N)
#         base_origin = base_origin.T
#     # base_origin now (3, M); use first column as template
#     template = base_origin[:, :1]
#     repeated = np.repeat(template, len(z_offsets), axis=1).astype(np.float32)
#     repeated[0, :] += z_offsets.astype(np.float32)  # adjust z row
#     spiral_geo.offOrigin = repeated
#     return spiral_geo


def _geo_with_spiral_offsets(base_geo, z_offsets): 
    """ Create a copy of the TIGRE geometry whose offOrigin follows the provided z offsets.
     TIGRE expects the offOrigin ordering to be [z, y, x], so we only perturb the first column. """ 
    if z_offsets is None: 
        return base_geo 
    spiral_geo = copy.deepcopy(base_geo) 
    base_origin = np.asarray(base_geo.offOrigin) 
    if base_origin.ndim == 1: 
        base_origin = np.expand_dims(base_origin, axis=0) 
    repeated = np.repeat(base_origin[:1, :], len(z_offsets), axis=0).astype(np.float32) 
    repeated[:, 0] += z_offsets.astype(np.float32) 
    spiral_geo.offOrigin = repeated 
    return spiral_geo


def main(args):
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    vol_path = args.vol
    scanner_cfg_path = args.scanner
    n_train = args.n_train
    n_test = args.n_test
    vol_name = osp.basename(vol_path)[:-4]
    output_path = args.output

    # Load configuration
    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)

    case_name = f"{vol_name}_{scanner_cfg['mode']}"
    print(f"Generate data for case {case_name}")
    geo = get_geometry_tigre(scanner_cfg)
    spiral_cfg = scanner_cfg.get("spiral")

    # Load volume
    vol = np.load(vol_path).astype(np.float32)

    # Generate training projections with multi-cycle support
    # Allow totalAngle to exceed 360 degrees for multiple rotations
    total_angle_deg = scanner_cfg.get("totalAngle", 360.0)
    start_angle_deg = scanner_cfg.get("startAngle", 0.0)
    
    # Generate angles that can span multiple full rotations
    projs_train_angles_full = (
        np.linspace(0, total_angle_deg / 180 * np.pi, n_train + 1)[:-1]
        + start_angle_deg / 180 * np.pi
    )
    
    # Convert to 0-2π range (modulo 2π) for TIGRE, but keep full angle for metadata
    projs_train_angles_for_tigre = np.mod(projs_train_angles_full, 2 * np.pi)
    
    # Generate testing projections (random angles within the total angle range)
    projs_test_angles_full = (
        np.sort(np.random.rand(n_test) * total_angle_deg / 180 * np.pi)
        + start_angle_deg / 180 * np.pi
    )
    projs_test_angles_for_tigre = np.mod(projs_test_angles_full, 2 * np.pi)
    
    # Build spiral offsets based only on number of views and z_delta_per_view
    spiral_train_offsets = _build_spiral_offsets(n_train, spiral_cfg)
    spiral_test_offsets = _build_spiral_offsets(n_test, spiral_cfg)

    # Apply spiral offsets per split so each projector sees the correct z displacement.
    geo_train = _geo_with_spiral_offsets(geo, spiral_train_offsets)
    geo_test = _geo_with_spiral_offsets(geo, spiral_test_offsets)

    # Generate projections using modulo angles (0-2π range)
    projs_train = tigre.Ax(
        np.transpose(vol, (2, 1, 0)).copy(), geo_train, projs_train_angles_for_tigre
    )[:, ::-1, :]
    if scanner_cfg["noise"]:
        projs_train = CTnoise.add(
            projs_train,
            Poisson=float(scanner_cfg["possion_noise"]),
            Gaussian=np.array(scanner_cfg["gaussian_noise"]),
        )  #
        projs_train[projs_train < 0.0] = 0.0

    projs_test = tigre.Ax(np.transpose(vol, (2, 1, 0)).copy(), geo_test, projs_test_angles_for_tigre)[
        :, ::-1, :
    ]

    # Save
    case_save_path = osp.join(output_path, case_name)
    os.makedirs(case_save_path, exist_ok=True)
    np.save(osp.join(case_save_path, "vol_gt.npy"), vol)
    file_path_dict = {}
    # Keep per-view offOrigin used in forward projection for downstream reconstruction.
    train_offorigin = getattr(geo_train, "offOrigin", None)
    test_offorigin = getattr(geo_test, "offOrigin", None)
    split_payloads = [
        ("proj_train", projs_train, projs_train_angles_for_tigre, spiral_train_offsets),
        ("proj_test", projs_test, projs_test_angles_for_tigre, spiral_test_offsets),
    ]
    for split, projs, angles, z_shifts in split_payloads:
        os.makedirs(osp.join(case_save_path, split), exist_ok=True)
        file_path_dict[split] = []
        for i_proj in range(projs.shape[0]):
            proj = projs[i_proj]
            frame_save_name = osp.join(split, f"{split}_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), proj)
            meta_entry = {
                "file_path": frame_save_name,
                "angle": float(angles[i_proj]),  # Store full angle (can be > 2π)
            }
            # Store camera z offset relative to volume (negative of volume offset)
            # In TIGRE forward projection, we shift the volume via offOrigin.
            # For camera pose in rendering, we need the camera's z offset relative
            # to the static volume, which is the negative of the volume's z offset.
            if z_shifts is not None:
                # z_shifts[i_proj] = volume's z offset relative to origin
                # camera_z_shift = camera's z offset relative to volume
                camera_z_shift = -z_shifts[i_proj]
                meta_entry["z_shift"] = float(camera_z_shift)
            file_path_dict[split].append(meta_entry)
    meta = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "proj_train": file_path_dict["proj_train"],
        "proj_test": file_path_dict["proj_test"],
    }
    if spiral_cfg:
        spiral_meta = copy.deepcopy(spiral_cfg)
        # Also store camera offsets (negative of volume offsets) for reference
        spiral_meta["train_z_shifts"] = (
            (-spiral_train_offsets).tolist() if spiral_train_offsets is not None else None
        )
        spiral_meta["test_z_shifts"] = (
            (-spiral_test_offsets).tolist() if spiral_test_offsets is not None else None
        )
        # Persist per-view offOrigin (volume frame, TIGRE [z, y, x]) used in forward projection.
        if train_offorigin is not None:
            spiral_meta["train_offOrigin"] = np.array(train_offorigin).tolist()
        if test_offorigin is not None:
            spiral_meta["test_offOrigin"] = np.array(test_offorigin).tolist()
        meta["spiral"] = spiral_meta
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="data_generator_usr/volume_gt/fan_beam_volume.npy", type=str, help="Path to volume.")
    parser.add_argument("--scanner", default="data_generator_usr/synthetic_dataset/scanner/spiral_cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=1000, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=1500, type=int, help="Number of projections for evaluation.")
    # fmt: on

    args = parser.parse_args()
    main(args)
