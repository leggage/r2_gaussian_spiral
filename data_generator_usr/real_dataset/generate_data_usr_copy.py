import os
import os.path as osp
import sys
import argparse
import glob
import numpy as np
from tqdm import trange
import tigre.algorithms as algs
import scipy
import cv2
import random
import json
import h5py

random.seed(0)

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre


def _mm_to_scene(value_mm, object_scale):
    if value_mm is None:
        return None
    value = np.asarray(value_mm, dtype=float)
    scaled = value / 1000.0 * object_scale
    if scaled.size == 1:
        return float(scaled)
    return scaled.tolist()


def _run_fdk_with_mode(projs, geo, angles, mode):
    if mode == "fan":
        for fn_name in ("fdk_fan", "FDK_fan", "FDK_Fan", "fdkFan"):
            fan_fn = getattr(algs, fn_name, None)
            if fan_fn is not None:
                return fan_fn(projs, geo, angles)
        raise RuntimeError(
            "Fan-beam FDK requested but TIGRE fan-beam solver is unavailable."
        )
    return algs.fdk(projs, geo, angles)


def _load_geometry_json(json_path, object_scale, proj_subsample):
    with open(json_path, "r", encoding="utf-8") as f:
        geometry = json.load(f)

    projections = geometry.get("projections", [])
    if len(projections) == 0:
        raise ValueError(f"No projections listed in {json_path}.")

    angles = []
    order = []
    projection_meta = {}
    for proj in projections:
        stem = str(proj.get("file_stem"))
        if stem is None:
            raise ValueError("Each projection needs a 'file_stem' entry.")
        angle_rad = proj.get("angle_rad")
        if angle_rad is None:
            angle_deg = proj.get("angle_deg")
            if angle_deg is None:
                raise ValueError(f"Projection {stem} is missing angle info.")
            angle_rad = float(angle_deg)
        order.append(stem)
        angles.append(angle_rad)
        extra = {"angle": angle_rad}
        if "table_z_mm" in proj:
            extra["table_z"] = _mm_to_scene(proj["table_z_mm"], object_scale)
        projection_meta[stem] = extra


    scanner = geometry.get("scanner", {})
    scanner_cfg = {}
    scanner_mode = scanner.get("mode")
    if scanner_mode and scanner_mode not in ("cone", "fan"):
        raise ValueError(f"Unsupported scanner mode '{scanner_mode}' in {json_path}")
    if "DSD_mm" in scanner:
        scanner_cfg["DSD"] = _mm_to_scene(scanner["DSD_mm"], object_scale)
    if "DSO_mm" in scanner:
        scanner_cfg["DSO"] = _mm_to_scene(scanner["DSO_mm"], object_scale)
    if "collimation_width" in scanner:
        scanner_cfg["collimation_width"] = _mm_to_scene(
            scanner["collimation_width"], object_scale
        )
    ###x,y方向像素大小（不一样）
    if "detector_pixel_size_mm" in scanner:
        detector_spacing = np.array(scanner["detector_pixel_size_mm"], dtype=float).reshape(-1)
        if detector_spacing.size == 1:
            detector_spacing = np.repeat(detector_spacing, 2)
        detector_spacing = detector_spacing * proj_subsample / 1000.0 * object_scale
        scanner_cfg["detector_spacing"] = detector_spacing.tolist()

    return {
        "angles": np.array(angles, dtype=np.float32),
        "order": order,
        "projection_meta": projection_meta,
        "scanner_overrides": scanner_cfg,
        "scanner_mode": scanner_mode,
        "angle_start_deg": angles[0] * 180.0 / np.pi,
        "angle_last_deg": angles[-1] * 180.0 / np.pi,
    }


def _z_shift_monotonicity_fix_applies(ordered_z_shifts):
    """
    Along acquisition order: if table-z offset is **increasing** (median Δz > 0), we apply
    the helical convention fix: negate stored z_shift **and** flip each detector image
    vertically (``np.flip(..., axis=0)`` — rows / 上下翻).

    If already non-increasing (decreasing or flat), return ``(False, median_Δz)``.

    Returns
    -------
    apply_fix : bool
        True if z_shift should be negated and projections flipped UD.
    median_dz : float
        ``nan`` if undetermined; else median of successive differences.
    """
    z = np.asarray(ordered_z_shifts, dtype=float)
    if z.size < 2:
        return False, float("nan")
    d = np.diff(z)
    if not np.any(np.isfinite(d)):
        return False, float("nan")
    if np.allclose(d, 0.0, rtol=0.0, atol=1e-12):
        print("[z_shift] table_z flat along projection order; no convention fix.", flush=True)
        return False, 0.0
    med = float(np.nanmedian(d))
    if med > 0:
        print(
            "[z_shift] table_z increases along projection order (median Δz=%.6f); "
            "applied z_shift *= -1 and vertical flip (axis=0) on projections." % med,
            flush=True,
        )
        return True, med
    print(
        "[z_shift] non-increasing along projection order (median Δz=%.6f); "
        "no z sign change, no projection flip." % med,
        flush=True,
    )
    return False, med


def main(args):
    input_data_path = args.data
    proj_subsample = args.proj_subsample
    proj_rescale = args.proj_rescale
    object_scale = args.object_scale

    geometry_info = None
    scanner_mode = args.scanner_mode
    if args.geometry_json:
        geometry_info = _load_geometry_json(
            args.geometry_json, object_scale, proj_subsample
        )
        angles = geometry_info["angles"]
        n_proj = len(angles)
        angle_start = geometry_info["angle_start_deg"]
        angle_last = geometry_info["angle_last_deg"]
        scanner_overrides = geometry_info["scanner_overrides"]
        if geometry_info["scanner_mode"]:
            scanner_mode = geometry_info["scanner_mode"]
        DSD = scanner_overrides.get("DSD")
        DSO = scanner_overrides.get("DSO")
        dDetector = scanner_overrides.get("detector_spacing")
        if any(val is None for val in (DSD, DSO, dDetector)):
            raise ValueError(
                "geometry_json must specify DSD_mm, DSO_mm, and detector_pixel_size_mm."
            )
    else:
        # Read configuration
        config_file_path = osp.join(input_data_path, "config.txt")
        with open(config_file_path, "r") as f:
            for config_line in f.readlines():
                if "NumberImages" in config_line:
                    n_proj = int(config_line.split("=")[-1])
                elif "AngleInterval" in config_line:
                    angle_interval = float(config_line.split("=")[-1])
                elif "AngleFirst" in config_line:
                    angle_start = float(config_line.split("=")[-1])
                elif "AngleLast" in config_line:
                    angle_last = float(config_line.split("=")[-1])
                elif "DistanceSourceDetector" in config_line:
                    DSD = float(config_line.split("=")[-1]) / 1000 * object_scale
                elif "DistanceSourceOrigin" in config_line:
                    DSO = float(config_line.split("=")[-1]) / 1000 * object_scale
                elif "PixelSize" in config_line and "PixelSizeUnit" not in config_line:
                    dDetector = (
                        float(config_line.split("=")[-1])
                        * proj_subsample
                        / 1000
                        * object_scale
                    )
        angles = np.concatenate(
            [np.arange(angle_start, angle_last, angle_interval), [angle_last]]
        )
        angles = angles / 180.0 * np.pi
        dDetector = [dDetector, dDetector]

    # Read and save projections
    output_path = args.output
    all_save_path = osp.join(output_path, "proj_all")
    train_save_path = osp.join(output_path, "proj_train")
    test_save_path = osp.join(output_path, "proj_test")
    os.makedirs(all_save_path, exist_ok=True)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    proj_mat_paths = sorted(glob.glob(osp.join(input_data_path, "*.mat")))
    if geometry_info:
        stem_to_path = {
            osp.basename(path).split(".")[0]: path for path in proj_mat_paths
        }
        ordered_paths = []
        for stem in geometry_info["order"]:
            if stem not in stem_to_path:
                raise FileNotFoundError(
                    f"Projection file with stem {stem} not found in {input_data_path}."
                )
            ordered_paths.append(stem_to_path[stem])
        proj_mat_paths = ordered_paths

    # Raw table_z along the same order as proj_mat_paths (before sign / projection flip).
    if geometry_info:
        ordered_z_preview = []
        for stem in geometry_info["order"]:
            meta = geometry_info["projection_meta"].get(stem, {})
            tz = meta.get("table_z")
            ordered_z_preview.append(float(tz) if tz is not None else 0.0)
    else:
        ordered_z_preview = [0.0] * len(proj_mat_paths)

    z_shift_sign_flipped, _ = _z_shift_monotonicity_fix_applies(ordered_z_preview)

    projection_train_list = []
    projection_test_list = []
    train_ids = np.linspace(0, n_proj - 1, args.n_train).astype(int)
    test_ids = sorted(
        random.sample(np.setdiff1d(np.arange(n_proj), train_ids).tolist(), args.n_test)
    )
    extra_meta = geometry_info["projection_meta"] if geometry_info else {}
    count = 0
    for i_proj in trange(len(proj_mat_paths), desc=osp.basename(output_path)):
        count += 1
        # if count > 4000:
        #     break
        proj_mat_path = proj_mat_paths[i_proj]
        proj_save_name = osp.basename(proj_mat_path).split(".")[0]
        proj_meta = extra_meta.get(proj_save_name, {})
        angle_value = proj_meta.get("angle", angles[i_proj])
        z_shift = 0.0
        if proj_meta and "table_z" in proj_meta and proj_meta["table_z"] is not None:
            z_shift = float(proj_meta["table_z"])
        if z_shift_sign_flipped:
            z_shift = -z_shift
        if i_proj in train_ids:
            entry = {
                "file_path": osp.join(
                    osp.basename(train_save_path), proj_save_name + ".npy"
                ),
                "angle": angle_value,
                "z_shift": z_shift,
            }
            projection_train_list.append(entry)
        elif i_proj in test_ids:
            entry = {
                "file_path": osp.join(
                    osp.basename(test_save_path), proj_save_name + ".npy"
                ),
                "angle": angle_value,
                "z_shift": z_shift,
            }
            projection_test_list.append(entry)

        img = None
        try:
            mat = scipy.io.loadmat(proj_mat_path)
            img = mat.get("img")
        except NotImplementedError:
            pass
        if img is None:
            with h5py.File(proj_mat_path, "r") as f:
                img = f["img"][()]
        
        # Check if dimensions are swapped (MATLAB row×col vs NumPy height×width)


        proj = img
        # print(proj.shape)
        # proj = np.rot90(proj, k=-1, axes=(0, 1)).copy()
        proj = proj.astype(np.float32) / proj_rescale * object_scale
        proj[proj < 0] = 0

        if proj_subsample != 1.0:
            h_ori, w_ori = proj.shape
            h_new, w_new = int(h_ori / proj_subsample), int(w_ori / proj_subsample)
            proj = cv2.resize(proj, (w_new, h_new))  # cv2.resize expects (width, height)
            # # crop to rectangle
            # dim_x, dim_y = proj.shape
            # if dim_x > dim_y:
            #     dim_offset = int((dim_x - dim_y) / 2)
            #     proj = proj[dim_offset:-dim_offset, :]
            # elif dim_x < dim_y:
            #     dim_offset = int((dim_y - dim_x) / 2)
            #     proj = proj[:, dim_offset:-dim_offset]

        if z_shift_sign_flipped:
            # Vertical flip (row axis); paired with z_shift sign fix for increasing table_z.
            proj = np.flip(proj, axis=0).copy()

        np.save(osp.join(all_save_path, proj_save_name + ".npy"), proj)
        if i_proj in train_ids:
            np.save(osp.join(train_save_path, proj_save_name + ".npy"), proj)
        elif i_proj in test_ids:
            np.save(osp.join(test_save_path, proj_save_name + ".npy"), proj)

    # Scanner config
    proj = np.load(osp.join(output_path, projection_train_list[0]["file_path"]))
    nDetector = [proj.shape[0], proj.shape[1]]
    detector_spacing = np.array(dDetector, dtype=float)
    if detector_spacing.size == 1:
        detector_spacing = np.repeat(detector_spacing, 2)
    sDetector = np.array(nDetector) * detector_spacing
    nVoxel = args.nVoxel
    sVoxel = list(args.sVoxel)
    offOrigin = list(args.offOrigin)

    # For helical datasets with z_shift metadata:
    # infer z bounds in scene units from min/max z_shift and auto-adjust volume span.
    all_z_shifts = [
        float(entry["z_shift"])
        for entry in (projection_train_list + projection_test_list)
        if "z_shift" in entry and entry["z_shift"] is not None
    ]
    if args.auto_svoxel_from_zshift and len(all_z_shifts) > 0:
        z_lower = float(np.min(all_z_shifts))
        z_upper = float(np.max(all_z_shifts))
        collimation_width = None
        if geometry_info is not None:
            collimation_width = geometry_info["scanner_overrides"].get(
                "collimation_width"
            )
        # if collimation_width is not None:
        #     cw = float(collimation_width)
        #     # print(f"collimation_width: {cw}")
        #     z_lower -= 0.5 * cw
        #     z_upper += 0.5 * cw
        z_span = max(float(z_upper - z_lower), float(args.min_svoxel_span))
        z_center = 0.5 * (z_lower + z_upper)

        if args.equal_xyz_span:
            sVoxel = [z_span, z_span, z_span]
        else:
            sVoxel[2] = z_span
        offOrigin[2] = z_center

        print(
            f"[auto_svoxel_from_zshift] z_shift range=({z_lower:.6f}, {z_upper:.6f}), "
            f"sVoxel={sVoxel}, offOrigin={offOrigin}"
        )
    bbox = np.array(
        [
            np.array(offOrigin) - np.array(sVoxel) / 2,
            np.array(offOrigin) + np.array(sVoxel) / 2,
        ]
    ).tolist()
    scanner_cfg = {
        "mode": scanner_mode,
        "DSD": DSD,
        "DSO": DSO,
        "nDetector": nDetector,
        "sDetector": sDetector.tolist(),
        "nVoxel": nVoxel,
        "sVoxel": sVoxel,
        "offOrigin": offOrigin,
        "offDetector": args.offDetector,
        "accuracy": args.accuracy,
        "coord_left": True,
        "totalAngle": angle_last - angle_start,
        "startAngle": angle_start,
        "noise": True,
        "filter": None,
    }
    if geometry_info:
        scanner_cfg["geometry_source"] = osp.basename(args.geometry_json)
    else:
        scanner_cfg["geometry_source"] = "config.txt"

    # Reconstruct with FDK as gt
    # ct_gt_save_path = osp.join(output_path, "vol_gt.npy")
    # if not osp.exists(ct_gt_save_path):
    #     projs = []
    #     skip = 1
    #     proj_paths = sorted(glob.glob(osp.join(all_save_path, "*.npy")))
    #     for proj_path in proj_paths[::skip]:
    #         proj = np.load(proj_path)
    #         nDetector = proj.shape
    #         projs.append(proj)
    #     projs = np.stack(projs, axis=0)
    #     print("reconstruct with FDK")
    #     geo = get_geometry_tigre(scanner_cfg)
    #     ct_gt = _run_fdk_with_mode(
    #         projs[:, ::-1, :], geo, angles[::skip], scanner_cfg["mode"]
    #     )
    #     ct_gt = ct_gt.transpose((2, 1, 0))
    #     ct_gt[ct_gt < 0] = 0
    #     np.save(ct_gt_save_path, ct_gt)

    # Build train_offOrigin and test_offOrigin from z_shift
    # Format: [[z, y, x], [z, y, x], ...] where z = -z_shift, y = 0, x = 0
    train_offOrigin = [
        [-entry["z_shift"], 0.0, 0.0] for entry in projection_train_list
    ]
    test_offOrigin = [
        [-entry["z_shift"], 0.0, 0.0] for entry in projection_test_list
    ]
    
    # Save
    meta_data = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
        "spiral": {
            "enabled": True,
            "train_offOrigin": train_offOrigin,
            "test_offOrigin": test_offOrigin,
            "z_shift_sign_flipped_to_decreasing": bool(z_shift_sign_flipped),
            "projection_flipped_ud_axis0": bool(z_shift_sign_flipped),
        },
    }
    if len(all_z_shifts) > 0:
        meta_data["spiral"]["z_shift_range"] = [
            float(np.min(all_z_shifts)),
            float(np.max(all_z_shifts)),
        ]
        meta_data["spiral"]["auto_svoxel_from_zshift"] = bool(
            args.auto_svoxel_from_zshift
        )
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Data saved in {output_path}")


if __name__ == "__main__":
    # fmt: off
#######scene_rescale:ori/1000*object_scale
#######proj_rescale:ori/proj_rescale*object_scale

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/LDCT-L004/SPIRAL_processed_transposed/proj", type=str, help="Path to  data.")
    parser.add_argument("--output", default="data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200_projd20", type=str, help="Path to output.")
    parser.add_argument("--proj_subsample", default=1, type=int, help="subsample projections pixels")
    parser.add_argument("--proj_rescale", default=1000.0, type=float, help="rescale projection values to fit density to around [0,1]")
    parser.add_argument("--object_scale", default=50, type=int, help="Rescale the whole scene to similar scales as the synthetic data")
    parser.add_argument("--n_test", default=300, type=int, help="number of test")
    parser.add_argument("--n_train", default=200, type=int, help="number of train")
    parser.add_argument("--geometry_json", default="data/LDCT-L004/SPIRAL_processed_transposed/scanner_geometry.json", type=str, help="Optional scanner geometry JSON exported from dicom_spiral_process.m")
    parser.add_argument("--scanner_mode", default="cone", choices=["cone", "fan"], help="Beam model used by the scanner")
    
    parser.add_argument("--nVoxel", nargs="+", default=[256, 256, 256], type=int, help="voxel dimension")
    parser.add_argument("--sVoxel", nargs="+", default=[20, 20, 20], type=float, help="volume size")
    parser.add_argument("--offOrigin", nargs="+", default=[0.0, 0.0, 0], type=float, help="offOrigin")
    parser.add_argument("--auto_svoxel_from_zshift", action="store_true", help="Use min/max z_shift (after _mm_to_scene) to auto set z-range; enabled by default for helical projections.")
    parser.add_argument("--equal_xyz_span", action="store_true", help="When auto z range is enabled, force sVoxel x/y/z to the same span (default behavior).")
    parser.add_argument("--min_svoxel_span", default=1e-6, type=float, help="Minimum allowed voxel span when auto-estimating from z_shift.")
    parser.add_argument("--offDetector", nargs="+", default=[0.0, 0.0], type=float, help="offDetector")
    parser.add_argument("--accuracy", default=0.5, type=float, help="accuracy")
    
    
    parser.set_defaults(auto_svoxel_from_zshift=True, equal_xyz_span=True)
    args = parser.parse_args()
    main(args)
    # fmt: on
