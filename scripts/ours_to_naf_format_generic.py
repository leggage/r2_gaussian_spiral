import argparse
import copy
import json
import os
import os.path as osp
import pickle

import numpy as np


def load_split(case_path, items, coord_left):
    projs = []
    angles = []
    z_shifts = []
    for item in items:
        arr = np.load(osp.join(case_path, item["file_path"]))
        if coord_left:
            arr = arr[:, ::-1]
        projs.append(arr)
        angles.append(item["angle"])
        if "z_shift" in item:
            z_shifts.append(item["z_shift"])
        else:
            z_shifts.append(0.0)
    out = {
        "projections": np.stack(projs, axis=0),
        "angles": np.asarray(angles),
        "z_shift": np.asarray(z_shifts),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert r2_gaussian dataset to SAX-NeRF/NAF pickle format.")
    parser.add_argument("--data_path", required=True, type=str, help="Case folder containing meta_data.json.")
    parser.add_argument("--output_path", required=True, type=str, help="Output .pickle path.")
    args = parser.parse_args()

    case_path = osp.abspath(args.data_path)
    output_path = osp.abspath(args.output_path)
    meta_data_path = osp.join(case_path, "meta_data.json")
    if not osp.exists(meta_data_path):
        raise FileNotFoundError(f"meta_data.json not found: {meta_data_path}")

    with open(meta_data_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    scanner_cfg = meta_data["scanner"]
    coord_left = bool(scanner_cfg.get("coord_left", False))
    train = load_split(case_path, meta_data["proj_train"], coord_left)
    val = load_split(case_path, meta_data["proj_test"], coord_left)
    img = np.load(osp.join(case_path, meta_data["vol"]))

    pkl_dict = copy.deepcopy(scanner_cfg)
    pkl_dict.update(
        {
            "numTrain": int(train["angles"].shape[0]),
            "coord_left": coord_left,
            "numVal": int(val["angles"].shape[0]),
            "dDetector": (np.array(scanner_cfg["sDetector"]) / np.array(scanner_cfg["nDetector"]) * 1000).tolist(),
            "dVoxel": (np.array(scanner_cfg["sVoxel"]) / np.array(scanner_cfg["nVoxel"]) * 1000).tolist(),
            "train": train,
            "val": val,
            "image": img,
        }
    )
    pkl_dict["DSD"] = (np.array(pkl_dict["DSD"]) * 1000).tolist()
    pkl_dict["DSO"] = (np.array(pkl_dict["DSO"]) * 1000).tolist()
    pkl_dict["sDetector"] = (np.array(pkl_dict["sDetector"]) * 1000).tolist()
    pkl_dict["sVoxel"] = (np.array(pkl_dict["sVoxel"]) * 1000).tolist()
    pkl_dict["offOrigin"] = (np.array(pkl_dict["offOrigin"]) * 1000).tolist()
    pkl_dict["offDetector"] = (np.array(pkl_dict["offDetector"]) * 1000).tolist()

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pkl_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
