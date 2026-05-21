import os
import os.path as osp
import shutil
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


def _collimation_width_mm(dsd, dso, s_detector_v):
    """Axial collimation at isocenter (mm): 2 * tan( sDetector_v / (2*DSD) ) * DSO."""
    return 2.0 * float(dso) * np.tan(float(s_detector_v) / (2.0 * float(dsd)))


def _physical_spiral_enabled(spiral_cfg):
    if not spiral_cfg or spiral_cfg.get("enabled", True) is False:
        return False
    if spiral_cfg.get("physical_spiral") is False:
        return False
    return "pitch" in spiral_cfg and "sample_per_rotation" in spiral_cfg


def _build_physical_spiral_train(spiral_cfg, geo, scanner_cfg):
    """
    Helical table feed: table_feed_per_rotation = pitch * collimation_width.
    Per view: z_gap = table_feed_per_rotation / sample_per_rotation,
              angle_gap = 2*pi / sample_per_rotation.
    First view at (z_start, angle_start). Append views while z <= z_end (strict).
    """
    dsd = float(scanner_cfg["DSD"])
    dso = float(scanner_cfg["DSO"])
    s_det_v = float(np.asarray(geo.sDetector).reshape(-1)[0])
    pitch = float(spiral_cfg["pitch"])
    spr = int(spiral_cfg["sample_per_rotation"])
    if spr < 1:
        raise ValueError("sample_per_rotation must be >= 1")

    coll_w = _collimation_width_mm(dsd, dso, s_det_v)
    table_feed = pitch * coll_w
    z_gap = table_feed / spr
    angle_gap = (2.0 * np.pi) / spr

    z0 = float(spiral_cfg.get("z_start", 0.0))
    z1 = float(spiral_cfg.get("z_end", spiral_cfg.get("zend", 0.0)))
    angle_start_deg = float(
        spiral_cfg.get("angle_start", spiral_cfg.get("angle_start_deg", scanner_cfg.get("startAngle", 0.0)))
    )
    a0 = angle_start_deg / 180.0 * np.pi

    z_list = []
    ang_list = []
    z = z0
    ang = a0
    # First sample at (z0, a0); stop before taking any view with z > z_end
    while z <= z1 + 1e-9:
        z_list.append(z)
        ang_list.append(ang)
        z_next = z + z_gap
        if z_next > z1 + 1e-9:
            break
        z = z_next
        ang = ang + angle_gap

    z_arr = np.asarray(z_list, dtype=np.float32)
    ang_full = np.asarray(ang_list, dtype=np.float64)
    ang_tigre = np.mod(ang_full, 2.0 * np.pi).astype(np.float64)

    dbg = {
        "collimation_width_mm": float(coll_w),
        "table_feed_per_rotation_mm": float(table_feed),
        "z_gap_mm": float(z_gap),
        "angle_gap_rad": float(angle_gap),
        "sample_per_rotation": spr,
        "pitch": pitch,
        "num_train_views": int(len(z_arr)),
        "num_raw_helical_views": int(len(z_arr)),
    }
    return z_arr, ang_full, ang_tigre, dbg


def _resample_trajectory(z_seq, angle_full_seq, n_target):
    """
    Resample along acquisition index with linear interpolation in z and angle (unwrapped).
    n_target can be larger than len(z_seq) (extrapolate along chord in index space).
    """
    z_seq = np.asarray(z_seq, dtype=np.float64).reshape(-1)
    a_seq = np.asarray(angle_full_seq, dtype=np.float64).reshape(-1)
    n = int(z_seq.size)
    if n == 0:
        raise ValueError("Cannot resample empty trajectory")
    if n_target < 1:
        raise ValueError("n_target must be >= 1")
    if n == 1:
        return (
            np.full(n_target, z_seq[0], dtype=np.float32),
            np.full(n_target, a_seq[0], dtype=np.float64),
        )
    idx_new = np.linspace(0, n - 1, n_target)
    z_new = np.interp(idx_new, np.arange(n), z_seq).astype(np.float32)
    ang_new = np.interp(idx_new, np.arange(n), a_seq)
    return z_new, ang_new


def _linspace_sample_indices(n_source, n_target):
    """Round indices in [0, n_source-1] for subsampling; length n_target (duplicates allowed)."""
    if n_source < 1 or n_target < 1:
        raise ValueError
    idx = np.linspace(0, n_source - 1, n_target)
    return np.clip(np.round(idx).astype(np.int64), 0, n_source - 1)


def cat_m_like_z_stitch(
    projs,
    z_mm,
    angles_rad,
    pitch,
    sample_per_rotation,
    width_row_mm,
    row,
    col,
    collimation_width_mm=None,
):
    """
    Mimic CAT.m: sort by z, stack detector rows on a tall canvas by z→index mapping,
    then crop to the common region where all views have coverage.
    projs: (N, row, col) numpy (v, u); z_mm, angles_rad: (N,) per projection (acquisition).
    Returns cropped (crop_rows, col, spr), angle_rad_by_view (spr,), stitch_meta dict.
    """
    num_proj, rv, cv = projs.shape
    if rv != row or cv != col:
        raise ValueError(f"projs shape mismatch: got ({num_proj},{rv},{cv}), expect row={row}, col={col}")
    spr = int(sample_per_rotation)
    if spr < 1:
        raise ValueError("sample_per_rotation must be >= 1")

    order = np.argsort(z_mm, kind="stable")
    z_sorted = z_mm[order]
    ang_sorted = angles_rad[order]
    proj_sorted = projs[order]

    num_rot_est = int(np.ceil(num_proj / spr))
    n_canvas_rot = num_rot_est + 1
    canvas_rows = n_canvas_rot * row

    z_min = float(z_sorted[0])
    if collimation_width_mm is not None and collimation_width_mm > 1e-12:
        coll_w = float(collimation_width_mm)
    elif num_proj > spr:
        coll_w = float((z_sorted[spr] - z_sorted[0]) / pitch)
    else:
        coll_w = float(width_row_mm * row)
    if abs(coll_w) < 1e-12:
        raise ValueError("Invalid collimation_width for z-stitch.")

    def z_to_index_1based(zpos):
        return int(np.round((float(zpos) - z_min) / coll_w * row)) + 1

    stitched = np.zeros((canvas_rows, col, spr), dtype=np.float32)
    angle_rad_by_view = np.zeros(spr, dtype=np.float64)
    angle_slot_filled = np.zeros(spr, dtype=bool)
    start_idx_all = np.zeros(num_proj, dtype=np.int32)

    for i in range(num_proj):
        view_id = i % spr
        if not angle_slot_filled[view_id]:
            angle_rad_by_view[view_id] = float(ang_sorted[i])
            angle_slot_filled[view_id] = True

        start_idx = z_to_index_1based(float(z_sorted[i]))
        start_idx = max(1, min(start_idx, canvas_rows - row + 1))
        end_idx = start_idx + row - 1
        start_idx_all[i] = start_idx
        # Stitched layout: (canvas_rows, col, view). Each projection is (row, col),
        # so we write directly without transpose.
        slab = proj_sorted[i].astype(np.float32, copy=False)
        stitched[start_idx - 1 : end_idx, :, view_id] = slab

    if num_proj >= spr:
        z1 = float(z_sorted[spr - 1])
        idx1 = z_to_index_1based(z1)
    else:
        idx1 = 1

    full_rotations = num_proj // spr
    if full_rotations >= 2:
        second_last_rot_start_1based = (full_rotations - 1) * spr + 1
        z2 = float(z_sorted[second_last_rot_start_1based - 1])
        idx2 = z_to_index_1based(z2)
    else:
        idx2 = canvas_rows - row + 1

    crop_start = max(1, min(idx1, canvas_rows))
    crop_end = max(crop_start, min(idx2 + row - 1, canvas_rows))

    cropped = stitched[crop_start - 1 : crop_end, :, :].copy()
    center_idx_all = start_idx_all.astype(np.float64) + np.floor(row / 2.0)
    is_kept = (center_idx_all >= crop_start) & (center_idx_all <= crop_end)

    meta = {
        "collimation_width_mm": coll_w,
        "canvas_rows": canvas_rows,
        "crop_start_1based": int(crop_start),
        "crop_end_1based": int(crop_end),
        "num_proj": int(num_proj),
        "samples_per_rotation": spr,
        "is_kept_by_sorted_index": is_kept.tolist(),
    }
    return cropped, angle_rad_by_view, meta


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

    total_angle_deg = scanner_cfg.get("totalAngle", 360.0)
    start_angle_deg = scanner_cfg.get("startAngle", 0.0)
    physical = _physical_spiral_enabled(spiral_cfg)
    phys_dbg = None
    cat_stitch = getattr(args, "cat_stitch", False)

    if physical:
        z_tr_raw, ang_tr_full_raw, _, phys_dbg = _build_physical_spiral_train(
            spiral_cfg, geo, scanner_cfg
        )
    else:
        z_tr_raw = _build_spiral_offsets(n_train, spiral_cfg)
        ang_tr_full_raw = (
            np.linspace(0, total_angle_deg / 180 * np.pi, n_train + 1)[:-1]
            + start_angle_deg / 180 * np.pi
        )
        if z_tr_raw is None:
            z_tr_raw = np.zeros(len(ang_tr_full_raw), dtype=np.float32)
        z_tr_raw = np.asarray(z_tr_raw, dtype=np.float32)
        ang_tr_full_raw = np.asarray(ang_tr_full_raw, dtype=np.float64)

    n_tr_raw = int(len(z_tr_raw))
    if n_tr_raw < 1:
        raise ValueError("螺旋轨迹视角数为 0：请检查 spiral 的 z_start / z_end 与 pitch、sample_per_rotation。")
    ang0 = float(ang_tr_full_raw[0]) if n_tr_raw > 0 else start_angle_deg / 180.0 * np.pi
    if n_tr_raw >= 2:
        span = max(float(ang_tr_full_raw[-1] - ang_tr_full_raw[0]), total_angle_deg / 180.0 * np.pi)
    else:
        span = total_angle_deg / 180.0 * np.pi

    n_test_raw = max(n_test * 8, 200)
    if spiral_cfg and spiral_cfg.get("enabled", True) is not False:
        z_te_raw = _build_spiral_offsets(n_test_raw, spiral_cfg)
        z_te_raw = np.asarray(z_te_raw, dtype=np.float32)
    else:
        z_te_raw = np.zeros(n_test_raw, dtype=np.float32)
    ang_te_raw = np.sort(np.random.rand(n_test_raw) * span) + ang0

    vol_ax = np.transpose(vol, (2, 1, 0)).copy()
    use_dense_stitch = cat_stitch and physical and (n_tr_raw > n_train) and (n_tr_raw > 0)

    if use_dense_stitch:
        geo_dense = _geo_with_spiral_offsets(geo, z_tr_raw)
        ang_dense_tigre = np.mod(ang_tr_full_raw, 2.0 * np.pi).astype(np.float64)
        projs_dense = tigre.Ax(vol_ax, geo_dense, ang_dense_tigre)[:, ::-1, :]
        if scanner_cfg["noise"]:
            projs_dense = CTnoise.add(
                projs_dense,
                Poisson=float(scanner_cfg["possion_noise"]),
                Gaussian=np.array(scanner_cfg["gaussian_noise"]),
            )
            projs_dense[projs_dense < 0.0] = 0.0
        ix_tr = _linspace_sample_indices(n_tr_raw, n_train)
        spiral_train_offsets = z_tr_raw[ix_tr].astype(np.float32)
        projs_train_angles_full = ang_tr_full_raw[ix_tr].astype(np.float64)
        projs_train_angles_for_tigre = np.mod(projs_train_angles_full, 2.0 * np.pi)
        projs_train = np.asarray(projs_dense[ix_tr], dtype=np.float32)
        projs_for_stitch = projs_dense
        z_for_stitch = np.asarray(z_tr_raw, dtype=np.float64)
        ang_for_stitch = np.asarray(ang_tr_full_raw, dtype=np.float64)
    else:
        if n_tr_raw >= n_train:
            ix_tr = _linspace_sample_indices(n_tr_raw, n_train)
            spiral_train_offsets = z_tr_raw[ix_tr].astype(np.float32)
            projs_train_angles_full = ang_tr_full_raw[ix_tr].astype(np.float64)
        else:
            spiral_train_offsets, projs_train_angles_full = _resample_trajectory(
                z_tr_raw, ang_tr_full_raw, n_train
            )
        projs_train_angles_for_tigre = np.mod(projs_train_angles_full, 2.0 * np.pi)
        geo_train = _geo_with_spiral_offsets(geo, spiral_train_offsets)
        projs_train = tigre.Ax(vol_ax, geo_train, projs_train_angles_for_tigre)[:, ::-1, :]
        if scanner_cfg["noise"]:
            projs_train = CTnoise.add(
                projs_train,
                Poisson=float(scanner_cfg["possion_noise"]),
                Gaussian=np.array(scanner_cfg["gaussian_noise"]),
            )
            projs_train[projs_train < 0.0] = 0.0
        projs_for_stitch = projs_train
        z_for_stitch = np.asarray(spiral_train_offsets, dtype=np.float64)
        ang_for_stitch = np.asarray(projs_train_angles_full, dtype=np.float64)

    if len(z_te_raw) >= n_test:
        ix_te = _linspace_sample_indices(len(z_te_raw), n_test)
        spiral_test_offsets = z_te_raw[ix_te].astype(np.float32)
        projs_test_angles_full = ang_te_raw[ix_te].astype(np.float64)
    else:
        spiral_test_offsets, projs_test_angles_full = _resample_trajectory(
            z_te_raw.astype(np.float64), ang_te_raw.astype(np.float64), n_test
        )
        spiral_test_offsets = spiral_test_offsets.astype(np.float32)
    projs_test_angles_for_tigre = np.mod(projs_test_angles_full, 2.0 * np.pi)

    geo_train = _geo_with_spiral_offsets(geo, spiral_train_offsets)
    geo_test = _geo_with_spiral_offsets(geo, spiral_test_offsets)

    projs_test = tigre.Ax(vol_ax, geo_test, projs_test_angles_for_tigre)[:, ::-1, :]

    # Save
    case_save_path = osp.join(output_path, case_name)
    os.makedirs(case_save_path, exist_ok=True)

    cone_cat_meta = None
    cone_case_path = None
    cone_entries = []
    st_meta = None
    spr = None
    if cat_stitch:
        if spiral_cfg is None:
            raise ValueError("--cat_stitch 需要 YAML 中的 spiral 配置")
        pch = float(spiral_cfg.get("pitch", 0.0))
        spr = int(spiral_cfg.get("sample_per_rotation", 0))
        if spr <= 0 or pch == 0.0:
            raise ValueError("--cat_stitch 需要 spiral.pitch 与 spiral.sample_per_rotation")
        s_det_v = float(np.asarray(geo.sDetector).reshape(-1)[0])
        dsd = float(scanner_cfg["DSD"])
        dso = float(scanner_cfg["DSO"])
        coll_w_stitch = (
            float(phys_dbg["collimation_width_mm"])
            if phys_dbg is not None
            else _collimation_width_mm(dsd, dso, s_det_v)
        )
        row = int(geo.nDetector[0])
        col = int(geo.nDetector[1])
        width_row = float(geo.dDetector[0])
        cropped, angle_by_view, st_meta = cat_m_like_z_stitch(
            projs_for_stitch.astype(np.float32, copy=False),
            z_for_stitch,
            ang_for_stitch,
            pch,
            spr,
            width_row,
            row,
            col,
            collimation_width_mm=coll_w_stitch,
        )
        cone_case_path = osp.join(output_path, f"{case_name}_cone_cat")
        os.makedirs(cone_case_path, exist_ok=True)
        cone_train_dir = osp.join(cone_case_path, "proj_train")
        os.makedirs(cone_train_dir, exist_ok=True)
        n_train_cat = int(args.n_train_cat) if args.n_train_cat is not None else int(spr)
        if n_train_cat < 1:
            raise ValueError("--n_train_cat 必须 >= 1")
        cat_train_idx = _linspace_sample_indices(spr, n_train_cat)
        cone_cat_rotate_k = int(args.cone_cat_rotate_k) % 4
        for j_out, j_src in enumerate(cat_train_idx):
            img = np.rot90(cropped[:, :, int(j_src)], k=cone_cat_rotate_k).astype(
                np.float32, copy=False
            )
            fn = osp.join("proj_train", f"proj_train_{j_out:04d}.npy")
            np.save(osp.join(cone_case_path, fn), img)
            cone_entries.append(
                {
                    "file_path": fn,
                    "angle": float(np.mod(angle_by_view[int(j_src)], 2.0 * np.pi)),
                    "angle_rad_full": float(angle_by_view[int(j_src)]),
                }
            )
        cone_cat_meta = {
            "cone_dataset_dir": osp.basename(cone_case_path),
            "stitch": st_meta,
            "proj_train_count_raw": int(spr),
            "proj_train_count_resampled": int(n_train_cat),
        }
    np.save(osp.join(case_save_path, "vol_gt.npy"), vol)
    file_path_dict = {}
    # Keep per-view offOrigin used in forward projection for downstream reconstruction.
    train_offorigin = getattr(geo_train, "offOrigin", None)
    test_offorigin = getattr(geo_test, "offOrigin", None)
    split_payloads = [
        (
            "proj_train",
            projs_train,
            projs_train_angles_for_tigre,
            spiral_train_offsets,
            projs_train_angles_full,
        ),
        (
            "proj_test",
            projs_test,
            projs_test_angles_for_tigre,
            spiral_test_offsets,
            projs_test_angles_full,
        ),
    ]
    for split, projs, angles_tigre, z_shifts, angles_meta in split_payloads:
        os.makedirs(osp.join(case_save_path, split), exist_ok=True)
        file_path_dict[split] = []
        for i_proj in range(projs.shape[0]):
            proj = projs[i_proj]
            frame_save_name = osp.join(split, f"{split}_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), proj)
            meta_entry = {
                "file_path": frame_save_name,
                "angle": float(angles_meta[i_proj]),
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
    if cone_cat_meta is not None:
        meta["cone_cat_stitch"] = cone_cat_meta
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
        spiral_meta["train_resample"] = {
            "n_source_views": n_tr_raw,
            "n_train": n_train,
            "dense_forward_for_cat_stitch": bool(use_dense_stitch),
        }
        spiral_meta["test_resample"] = {
            "n_source_views": int(len(z_te_raw)),
            "n_test": n_test,
        }
        if phys_dbg is not None:
            spiral_meta["physical_spiral"] = {**phys_dbg, "n_train_after_resample": n_train}
        meta["spiral"] = spiral_meta
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    if cone_case_path is not None:
        shutil.copy2(osp.join(case_save_path, "vol_gt.npy"), osp.join(cone_case_path, "vol_gt.npy"))
        os.makedirs(osp.join(cone_case_path, "proj_test"), exist_ok=True)
        n_test_cat = int(args.n_test_cat) if args.n_test_cat is not None else int(n_test)
        if n_test_cat < 1:
            raise ValueError("--n_test_cat 必须 >= 1")
        cat_test_idx = _linspace_sample_indices(int(spr), n_test_cat)
        cone_test_entries = []
        for i_out, i_src in enumerate(cat_test_idx):
            dst_rel = osp.join("proj_test", f"proj_test_{i_out:04d}.npy")
            img = np.rot90(cropped[:, :, int(i_src)], k=cone_cat_rotate_k).astype(
                np.float32, copy=False
            )
            np.save(osp.join(cone_case_path, dst_rel), img)
            cone_test_entries.append(
                {
                    "file_path": dst_rel,
                    "angle": float(np.mod(angle_by_view[int(i_src)], 2.0 * np.pi)),
                    "angle_rad_full": float(angle_by_view[int(i_src)]),
                }
            )
        cone_h, cone_w = np.rot90(cropped[:, :, 0], k=cone_cat_rotate_k).shape
        cone_scanner = copy.deepcopy(scanner_cfg)
        cone_scanner["nDetector"] = [int(cone_h), int(cone_w)]
        cone_scanner["sDetector"] = [
            float(cone_h * float(geo.dDetector[0])),
            float(geo.sDetector[1]),
        ]
        cone_meta = {
            "dataset_type": "cone_cat_stitch",
            "scanner": cone_scanner,
            "vol": "vol_gt.npy",
            "bbox": [[-1, -1, -1], [1, 1, 1]],
            "proj_train": cone_entries,
            "proj_test": cone_test_entries,
            "source_case_name": case_name,
            "cone_cat_stitch": {
                "stitch": st_meta,
                "samples_per_rotation": int(spr),
                "n_train_cat": int(len(cone_entries)),
                "n_test_cat": int(len(cone_test_entries)),
                "rotate_k": int(cone_cat_rotate_k),
            },
        }
        if spiral_cfg is not None:
            cone_meta["spiral"] = copy.deepcopy(meta.get("spiral", {}))
        with open(osp.join(cone_case_path, "meta_data.json"), "w", encoding="utf-8") as f:
            json.dump(cone_meta, f, indent=4)
        print(f"[cone_cat] 独立数据集已写入: {cone_case_path}")

    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="data_generator_usr/volume_gt/ldct_c001.npy", type=str, help="Path to volume.")
    parser.add_argument("--scanner", default="data_generator_usr/synthetic_dataset/scanner/spiral_cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/LDCT-C001/syn_dataset/syn_spiral_ntrain100", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=100, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=200, type=int, help="Number of projections for evaluation.")
    parser.add_argument(
        "--cat_stitch",
        action="store_true",
        help="CAT.m 风格 z 向拼接裁切；在 output 下额外生成 {case}_cone_cat/ 完整数据集（vol_gt、proj_train、proj_test、meta_data.json）。",
    )
    parser.add_argument(
        "--n_train_cat",
        type=int,
        default=50,
        help="cone-cat 数据集训练投影数量；默认使用 sample_per_rotation。",
    )
    parser.add_argument(
        "--n_test_cat",
        type=int,
        default=75,
        help="cone-cat 数据集测试投影数量；默认等于主数据集 n_test。",
    )
    parser.add_argument(
        "--cone_cat_rotate_k",
        type=int,
        default=0,
        help="cone-cat 导出投影旋转次数（每次90度，按 mod 4 生效）。默认0（不旋转）。",
    )
    # fmt: on

    args = parser.parse_args()
    main(args)
