import os
import os.path as osp
import sys
import argparse
import glob
import numpy as np
from tqdm import trange
import tigre.algorithms as algs
import scipy
import scipy.io
import cv2
import random
import json
import h5py

random.seed(0)

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre


def _mm_to_scene(value_mm, object_scale):
    """Convert millimeter values to scene scale."""
    if value_mm is None:
        return None
    value = np.asarray(value_mm, dtype=float)
    scaled = value / 1000.0 * object_scale
    if scaled.size == 1:
        return float(scaled)
    return scaled.tolist()


def _run_fdk_with_mode(projs, geo, angles, mode):
    """Run FDK reconstruction with appropriate mode (fan or cone)."""
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
    """Load scanner geometry and projection information from JSON file."""
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
        
        # Get angle (prefer rad, fallback to deg)
        angle_rad = proj.get("angle_rad")
        if angle_rad is None:
            angle_deg = proj.get("angle_deg")
            if angle_deg is None:
                raise ValueError(f"Projection {stem} is missing angle info.")
            angle_rad = float(angle_deg) * np.pi / 180.0
        else:
            angle_rad = float(angle_rad)
        
        order.append(stem)
        angles.append(angle_rad)
        
        extra = {"angle": angle_rad}
        if "source_z_mm" in proj:
            extra["source_z"] = _mm_to_scene(proj["source_z_mm"], object_scale)
        if "timestamp" in proj:
            extra["timestamp"] = proj["timestamp"]
        if "mat_file" in proj:
            extra["mat_file"] = proj["mat_file"]
        projection_meta[stem] = extra

    scanner = geometry.get("scanner", {})
    scanner_cfg = {}
    scanner_mode = scanner.get("mode", "cone")
    if scanner_mode not in ("cone", "fan"):
        raise ValueError(f"Unsupported scanner mode '{scanner_mode}' in {json_path}")
    
    if "DSD_mm" in scanner:
        scanner_cfg["DSD"] = _mm_to_scene(scanner["DSD_mm"], object_scale)
    if "DSO_mm" in scanner:
        scanner_cfg["DSO"] = _mm_to_scene(scanner["DSO_mm"], object_scale)
    if "detector_pixel_size_mm" in scanner:
        detector_spacing = np.array(scanner["detector_pixel_size_mm"], dtype=float).reshape(-1)
        if detector_spacing.size == 1:
            detector_spacing = np.repeat(detector_spacing, 2)
        detector_spacing = detector_spacing * proj_subsample / 1000.0 * object_scale
        scanner_cfg["detector_spacing"] = detector_spacing.tolist()
    
    if "detector_pixels" in scanner:
        detector_pixels = np.array(scanner["detector_pixels"], dtype=int)
        scanner_cfg["detector_pixels"] = detector_pixels.tolist()

    return {
        "angles": np.array(angles, dtype=np.float32),
        "order": order,
        "projection_meta": projection_meta,
        "scanner_overrides": scanner_cfg,
        "scanner_mode": scanner_mode,
        "angle_start_deg": angles[0] * 180.0 / np.pi,
        "angle_last_deg": angles[-1] * 180.0 / np.pi,
    }


def _load_mat_file(mat_path):
    """Load projection data from .mat file."""
    img = None
    try:
        mat = scipy.io.loadmat(mat_path)
        # Try common variable names
        for var_name in ["img", "pixel", "proj", "proj_blender", "data"]:
            if var_name in mat:
                img = mat[var_name]
                break
        # If no common name found, try to find the largest array
        if img is None:
            for key in mat.keys():
                if not key.startswith("__") and isinstance(mat[key], np.ndarray):
                    if img is None or mat[key].size > img.size:
                        img = mat[key]
    except NotImplementedError:
        pass
    
    if img is None:
        # Try HDF5 format
        try:
            with h5py.File(mat_path, "r") as f:
                # Try common keys
                for key in ["img", "pixel", "proj", "data"]:
                    if key in f:
                        img = f[key][()]
                        break
                # If not found, get first dataset
                if img is None and len(f.keys()) > 0:
                    first_key = list(f.keys())[0]
                    if not first_key.startswith("__"):
                        img = f[first_key][()]
        except Exception as e:
            raise ValueError(f"Failed to load {mat_path}: {e}")
    
    if img is None:
        raise ValueError(f"Could not find projection data in {mat_path}")
    
    return img


def main(args):
    """Main function to process real dataset projections and generate pseudo-GT."""
    input_data_path = args.data
    proj_subsample = args.proj_subsample
    proj_rescale = args.proj_rescale
    object_scale = args.object_scale

    # Load geometry from JSON
    if not args.geometry_json:
        raise ValueError("--geometry_json is required. Please provide path to scanner_geometry.json")
    
    geometry_info = _load_geometry_json(
        args.geometry_json, object_scale, proj_subsample
    )
    angles = geometry_info["angles"]
    n_proj = len(angles)
    angle_start = geometry_info["angle_start_deg"]
    angle_last = geometry_info["angle_last_deg"]
    scanner_overrides = geometry_info["scanner_overrides"]
    scanner_mode = geometry_info["scanner_mode"]
    
    DSD = scanner_overrides.get("DSD")
    DSO = scanner_overrides.get("DSO")
    dDetector = scanner_overrides.get("detector_spacing")
    
    if any(val is None for val in (DSD, DSO, dDetector)):
        raise ValueError(
            "geometry_json must specify DSD_mm, DSO_mm, and detector_pixel_size_mm."
        )

    # Create output directories
    output_path = args.output
    all_save_path = osp.join(output_path, "proj_all")
    train_save_path = osp.join(output_path, "proj_train")
    test_save_path = osp.join(output_path, "proj_test")
    os.makedirs(all_save_path, exist_ok=True)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    # Get projection file paths
    # First try to use mat_file from JSON, otherwise search for .mat files
    projection_meta = geometry_info["projection_meta"]
    proj_mat_paths = []
    
    for stem in geometry_info["order"]:
        meta = projection_meta.get(stem, {})
        mat_file = meta.get("mat_file")
        
        if mat_file:
            # Use path from JSON (relative to input_data_path or absolute)
            if osp.isabs(mat_file):
                mat_path = mat_file
            else:
                mat_path = osp.join(input_data_path, mat_file)
        else:
            # Fallback: search for .mat file with matching stem
            mat_path = osp.join(input_data_path, "proj", stem + ".mat")
            if not osp.exists(mat_path):
                # Try in root of input_data_path
                mat_path = osp.join(input_data_path, stem + ".mat")
        
        if not osp.exists(mat_path):
            raise FileNotFoundError(
                f"Projection file for stem {stem} not found. Tried: {mat_path}"
            )
        proj_mat_paths.append(mat_path)

    # Split into train and test sets
    projection_train_list = []
    projection_test_list = []
    train_ids = np.linspace(0, n_proj - 1, args.n_train).astype(int)
    test_ids = sorted(
        random.sample(np.setdiff1d(np.arange(n_proj), train_ids).tolist(), args.n_test)
    )

    # Process and save projections
    print(f"Processing {n_proj} projections...")
    for i_proj in trange(len(proj_mat_paths), desc="Processing projections"):
        proj_mat_path = proj_mat_paths[i_proj]
        proj_save_name = geometry_info["order"][i_proj]
        proj_meta = projection_meta.get(proj_save_name, {})
        angle_value = proj_meta.get("angle", angles[i_proj])

        # Load projection from .mat file
        proj = _load_mat_file(proj_mat_path)
        
        # Ensure 2D array
        if proj.ndim > 2:
            proj = proj.squeeze()
        if proj.ndim != 2:
            raise ValueError(f"Projection {proj_mat_path} is not 2D after squeezing")
        h_ori, w_ori = proj.shape
        
        # Store original dimensions for sDetector calculation
        if i_proj == 0:
            original_proj_shape = (h_ori, w_ori)
        
        if w_ori!=576:
            proj = cv2.resize(proj,[576,888],interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and rescale
        proj = proj.astype(np.float32) / proj_rescale * object_scale
        proj[proj < 0] = 0

        # Subsample if needed
        if proj_subsample != 1.0:
            h_ori, w_ori = proj.shape
            h_new, w_new = int(h_ori / proj_subsample), int(w_ori / proj_subsample)
            proj = cv2.resize(proj, [w_new, h_new], interpolation=cv2.INTER_LINEAR)
            # Crop to square if needed
            dim_x, dim_y = proj.shape
            if dim_x > dim_y:
                dim_offset = int((dim_x - dim_y) / 2)
                proj = proj[dim_offset:-dim_offset, :]
            elif dim_x < dim_y:
                dim_offset = int((dim_y - dim_x) / 2)
                proj = proj[:, dim_offset:-dim_offset]

        # Save projection
        np.save(osp.join(all_save_path, proj_save_name + ".npy"), proj)
        
        # Add to train or test list
        if i_proj in train_ids:
            np.save(osp.join(train_save_path, proj_save_name + ".npy"), proj)
            entry = {
                "file_path": osp.join(
                    osp.basename(train_save_path), proj_save_name + ".npy"
                ),
                "angle": float(angle_value),
            }
            projection_train_list.append(entry)
        elif i_proj in test_ids:
            np.save(osp.join(test_save_path, proj_save_name + ".npy"), proj)
            entry = {
                "file_path": osp.join(
                    osp.basename(test_save_path), proj_save_name + ".npy"
                ),
                "angle": float(angle_value),
            }
            projection_test_list.append(entry)

    # Get detector dimensions from final projection (after all processing)
    proj_sample = np.load(osp.join(output_path, projection_train_list[0]["file_path"]))
    nDetector = [proj_sample.shape[0], proj_sample.shape[1]]
    
    # Calculate sDetector based on original detector physical size
    # IMPORTANT: When projection is resized, physical size should remain constant
    # So we use original detector pixels * pixel_spacing, not resized pixels * pixel_spacing
    detector_spacing = np.array(dDetector, dtype=float)
    if detector_spacing.size == 1:
        detector_spacing = np.repeat(detector_spacing, 2)
    
    # Get original detector pixels from JSON if available
    original_detector_pixels = scanner_overrides.get("detector_pixels")
    if original_detector_pixels is not None:
        # Use original physical size: original_pixels * pixel_spacing
        # This preserves the physical detector size regardless of resizing
        sDetector = np.array(original_detector_pixels) * detector_spacing
    else:
        # Fallback: use current pixel count * pixel spacing
        # This assumes no resizing happened, or resizing preserved physical size
        sDetector = np.array(nDetector) * detector_spacing
    
    # Scanner configuration
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
        "geometry_source": osp.basename(args.geometry_json),
    }

    # Generate pseudo-GT using TIGRE FDK
    ct_gt_save_path = osp.join(output_path, "vol_gt.npy")
    if not osp.exists(ct_gt_save_path) or args.force_recon:
        print("Reconstructing pseudo-GT volume with FDK...")
        projs = []
        skip = 1
        proj_paths = sorted(glob.glob(osp.join(all_save_path, "*.npy")))
        angles_for_recon = []
        
        for i, proj_path in enumerate(proj_paths[::skip]):
            proj = np.load(proj_path)
            projs.append(proj)
            # Get corresponding angle
            proj_name = osp.basename(proj_path).split(".")[0]
            angle_idx = geometry_info["order"].index(proj_name)
            angles_for_recon.append(angles[angle_idx])
        
        projs = np.stack(projs, axis=0)
        angles_for_recon = np.array(angles_for_recon)
        
        print(f"Reconstructing from {len(projs)} projections...")
        geo = get_geometry_tigre(scanner_cfg)
        ct_gt = _run_fdk_with_mode(
            projs[:, ::-1, :], geo, angles_for_recon, scanner_cfg["mode"]
        )
        ct_gt = ct_gt.transpose((2, 1, 0))
        ct_gt[ct_gt < 0] = 0
        np.save(ct_gt_save_path, ct_gt)
        print(f"Pseudo-GT volume saved to {ct_gt_save_path}")
        print(f"Volume shape: {ct_gt.shape}, range: [{ct_gt.min():.4f}, {ct_gt.max():.4f}]")
    else:
        print(f"Pseudo-GT volume already exists at {ct_gt_save_path}, skipping reconstruction.")

    # Save metadata
    meta_data = {
        "scanner": scanner_cfg,
        "vol": "vol_gt1.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
    }
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"\nData processing complete!")
    print(f"Output directory: {output_path}")
    print(f"Training projections: {len(projection_train_list)}")
    print(f"Test projections: {len(projection_test_list)}")
    print(f"Total projections: {n_proj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process real dataset projections and generate pseudo-GT volume"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input data directory containing .mat projection files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--geometry_json",
        type=str,
        required=True,
        help="Path to scanner_geometry.json file generated by blender.m",
    )
    parser.add_argument(
        "--proj_subsample",
        default=1,
        type=int,
        help="Subsample projection pixels by this factor",
    )
    parser.add_argument(
        "--proj_rescale",
        default=400.0,
        type=float,
        help="Rescale projection values to fit density to around [0,1]",
    )
    parser.add_argument(
        "--object_scale",
        default=50,
        type=int,
        help="Rescale the whole scene to similar scales as the synthetic data",
    )
    parser.add_argument(
        "--n_test",
        default=100,
        type=int,
        help="Number of test projections",
    )
    parser.add_argument(
        "--n_train",
        default=75,
        type=int,
        help="Number of training projections",
    )
    parser.add_argument(
        "--nVoxel",
        nargs="+",
        default=[256, 256, 256],
        type=int,
        help="Voxel dimensions for reconstruction",
    )
    parser.add_argument(
        "--sVoxel",
        nargs="+",
        default=[4.0, 4.0, 4.0],
        type=float,
        help="Volume size in scene units",
    )
    parser.add_argument(
        "--offOrigin",
        nargs="+",
        default=[0.0, 0.0, 0.0],
        type=float,
        help="Offset of volume origin",
    )
    parser.add_argument(
        "--offDetector",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="Offset of detector",
    )
    parser.add_argument(
        "--accuracy",
        default=0.5,
        type=float,
        help="Accuracy parameter for TIGRE forward projection",
    )
    parser.add_argument(
        "--force_recon",
        action="store_true",
        help="Force reconstruction even if vol_gt.npy exists",
    )

    args = parser.parse_args()
    main(args)
