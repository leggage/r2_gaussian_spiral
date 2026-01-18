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
        # If MATLAB saved 888×64 (rows×cols), NumPy should have (888, 64)
        # If we get (64, 888), we need to transpose
        #if img.shape[0] < img.shape[1] and img.shape[0] == 64 and img.shape[1] == 888:
            # Likely transposed: transpose back
        #img = img.T

        proj = img
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
    sVoxel = args.sVoxel
    offOrigin = args.offOrigin
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

    # Save
    meta_data = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
    }
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Data saved in {output_path}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to  data.")
    parser.add_argument("--output", type=str, help="Path to output.")
    parser.add_argument("--proj_subsample", default=1, type=int, help="subsample projections pixels")
    parser.add_argument("--proj_rescale", default=400.0, type=float, help="rescale projection values to fit density to around [0,1]")
    parser.add_argument("--object_scale", default=50, type=int, help="Rescale the whole scene to similar scales as the synthetic data")
    parser.add_argument("--n_test", default=4500, type=int, help="number of test")
    parser.add_argument("--n_train", default=4000, type=int, help="number of train")
    parser.add_argument("--geometry_json", default=None, type=str, help="Optional scanner geometry JSON exported from dicom_spiral_process.m")
    parser.add_argument("--scanner_mode", default="cone", choices=["cone", "fan"], help="Beam model used by the scanner")
    
    parser.add_argument("--nVoxel", nargs="+", default=[256, 256, 256], type=int, help="voxel dimension")
    parser.add_argument("--sVoxel", nargs="+", default=[16.0, 16.0, 24.0], type=float, help="volume size")
    parser.add_argument("--offOrigin", nargs="+", default=[0.0, 0.0, -8.0], type=float, help="offOrigin")
    parser.add_argument("--offDetector", nargs="+", default=[0.0, 0.0], type=float, help="offDetector")
    parser.add_argument("--accuracy", default=0.5, type=float, help="accuracy")
    
    
    args = parser.parse_args()
    main(args)
    # fmt: on
