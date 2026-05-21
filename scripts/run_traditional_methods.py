# There is a np.int bug in TIGREv2.3. You need to change np.int to int manually.
# Usage example: python scripts/run_traditional_methods.py -s 0_chest_cone -m output/0_chest_cone_trad

import os
import os.path as osp
import numpy as np
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import tigre
import yaml
import copy
import json

sys.path.append("./")
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a, safe_state
from r2_gaussian.utils.ct_utils import get_geometry_tigre, run_ct_recon_algs
from r2_gaussian.utils.image_utils import metric_proj
from r2_gaussian.arguments import ModelParams


def main(dataset: ModelParams):

    # Set up dataset
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg
    scene_scale = scene.scene_scale

    # Load meta to check optional per-view offOrigin (e.g., spiral trajectories)
    meta_path = osp.join(dataset.source_path, "meta_data.json")
    spiral_offorigin_train = None
    spiral_offorigin_test = None
    if osp.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        spiral_meta = meta.get("spiral")
        if spiral_meta:
            def _parse_offorigin(key):
                arr = spiral_meta.get(key)
                if arr is None:
                    return None
                arr = np.asarray(arr, dtype=np.float32)
                # Normalize shape to (3, N)
                # if arr.ndim == 2 and arr.shape[0] != 3 and arr.shape[1] == 3:
                #     arr = arr.T
                return arr

            spiral_offorigin_train = _parse_offorigin("train_offOrigin")
            spiral_offorigin_test = _parse_offorigin("test_offOrigin")

    base_geo = get_geometry_tigre(scanner_cfg)
    geo_train = copy.deepcopy(base_geo)
    geo_test = copy.deepcopy(base_geo)
    if spiral_offorigin_train is not None:
        geo_train.offOrigin = (spiral_offorigin_train)*scene_scale+base_geo.offOrigin

    if spiral_offorigin_test is not None:
        geo_test.offOrigin = (spiral_offorigin_test)*scene_scale+base_geo.offOrigin

    projs_train = np.concatenate(
        [t2a(c.original_image) for c in scene.getTrainCameras()],
        axis=0,
    )
    projs_test = np.concatenate(
        [t2a(c.original_image) for c in scene.getTestCameras()],
        axis=0,
    )
    train_angles = np.stack([c.angle for c in scene.getTrainCameras()], axis=0)
    test_angles = np.stack([c.angle for c in scene.getTestCameras()], axis=0)

    vol_gt = t2a(scene.vol_gt)
    save_path = dataset.model_path

    out_dict = {}

    data_name = osp.basename(dataset.source_path)

    print("Run traditional algorithms on {}".format(data_name))
    methods = ["fdk", "sart", "asd_pocs"]
    # methods = ["fdk"]

    for method in methods:
        out_dict[method], ct_pred, _ = run_ct_recon_algs(
            projs_train, train_angles, copy.deepcopy(geo_train), vol_gt, save_path, method
        )
        # Render projections in test
        projs_test_pred = tigre.Ax(
            np.transpose(ct_pred, (2, 1, 0)).copy(), copy.deepcopy(geo_test), test_angles
        )[:, ::-1, :].copy()  # .copy() to avoid negative stride issue
        
        # Calculate 2D projection metrics
        # projs_test_pred and projs_test are in shape [N_proj, H, W]
        # Use axis=0 to compute metrics along projection dimension
        proj_psnr_mean, proj_psnr_list = metric_proj(
            projs_test, projs_test_pred, metric="psnr", axis=0, pixel_max=1.0
        )
        proj_ssim_mean, proj_ssim_list = metric_proj(
            projs_test, projs_test_pred, metric="ssim", axis=0
        )
        
        # Add projection metrics to output dict
        out_dict[method]["proj_psnr"] = float(proj_psnr_mean)
        out_dict[method]["proj_ssim"] = float(proj_ssim_mean)
        # out_dict[method]["proj_psnr_list"] = proj_psnr_list
        # out_dict[method]["proj_ssim_list"] = proj_ssim_list
        
        proj_save_path = osp.join(save_path, method, "projs")
        os.makedirs(proj_save_path, exist_ok=True)
        for i_proj in range(projs_test_pred.shape[0]):
            np.save(
                osp.join(proj_save_path, "{0:05d}_render.npy".format(i_proj)),
                projs_test_pred[i_proj],
            )
            np.save(
                osp.join(proj_save_path, "{0:05d}_gt.npy".format(i_proj)),
                projs_test[i_proj],
            )
            plt.imsave(
                osp.join(proj_save_path, "{0:05d}_render.png".format(i_proj)),
                projs_test_pred[i_proj],
                cmap="gray",
            )
            plt.imsave(
                osp.join(proj_save_path, "{0:05d}_gt.png".format(i_proj)),
                projs_test[i_proj],
                cmap="gray",
            )

    with open(osp.join(save_path, "eval.yml"), "w") as f:
        yaml.dump(out_dict, f, default_flow_style=False, sort_keys=False)

    print("Run traditional algorithms on {} complete".format(data_name))


if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser(description="Traditional method script parameters")
    model = ModelParams(parser)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    # fmt: on
    safe_state(args.quiet)

    main(model.extract(args))
