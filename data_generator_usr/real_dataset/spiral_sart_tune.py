"""
螺旋 CT 的 TIGRE 迭代伪 GT（helical_per_view + OS-SART / TV）。

流程
----
1. 与 ``spiral_fdk_tune`` 相同：几何、z 符号、``proj_all``、探测器翻转
2. 逐视角 ``geo.offOrigin`` → ``sart`` / ``ossart`` / ``ossart_tv``（可选 FDK 初值）
3. 评估与 ``spiral_fdk_tune`` 相同：``eval/slices``、``eval/reproj_compare``、PSNR/SSIM

默认配方对齐 ``generate_data_usr`` 的 ``ossart_tv``。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import tigre.algorithms as algs

from r2_gaussian.utils.ct_utils import get_geometry_tigre

from data_generator_usr.real_dataset.spiral_fdk_tune import (
    FILTER_CHOICES,
    SpiralViewList,
    evaluate_pseudo_gt,
    evaluate_reprojection,
    forward_project_helical,
    geo_with_per_view_offorigin,
    load_geometry_json,
    load_projections,
    load_spiral_views,
    normalize_volume,
    prepare_projs_for_tigre,
    resolve_geometry_json,
    run_fdk,
    save_volume_preview,
    save_volume_slice_profiles,
    validate_volume,
    vol_to_storage,
)

RECON_METHOD_CHOICES = ("sart", "ossart", "ossart_tv")
INIT_CHOICES = ("helical_fdk", "tigre_fdk", "none")


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass
class SartTuneResult:
    """迭代重建输出：存盘布局体数据 + 可序列化报告字段。"""

    volume: np.ndarray
    report: Dict[str, Any]


# -----------------------------------------------------------------------------
# TIGRE：螺旋迭代重建
# -----------------------------------------------------------------------------


def _scalarize_geo_dsd_dso(geo) -> None:
    """OS-SART / SART 与 FDK 初值路径一致：DSD/DSO 压成标量。"""
    for attr in ("DSD", "DSO"):
        val = getattr(geo, attr, None)
        if val is not None:
            scalar = float(np.asarray(val, dtype=np.float64).reshape(-1)[0])
            setattr(geo, attr, np.array([scalar], dtype=np.float32))


def _resolve_init_volume(
    projs: np.ndarray,
    geo,
    angles: np.ndarray,
    init: str,
    fdk_filter: Optional[str],
    verbose: bool,
) -> Optional[Union[str, np.ndarray]]:
    """
    TIGRE ``init`` 参数。

    - ``helical_fdk``：本仓库逐视角 FDK（与 fdk_tune 一致，推荐）
    - ``tigre_fdk``：字符串 ``"FDK"``，交给 TIGRE 内置 FDK
    - ``none``：``None``，由算法默认初始化
    """
    if init == "none":
        return None
    if init == "tigre_fdk":
        return "FDK"
    if init == "helical_fdk":
        vol_tigre = run_fdk(projs, geo, angles, filter_name=fdk_filter, verbose=verbose)
        return np.asarray(vol_tigre, dtype=np.float32)
    raise ValueError(f"Unknown init: {init!r}. Choose from {INIT_CHOICES}.")


def run_helical_iterative(
    projs: np.ndarray,
    geo,
    angles: np.ndarray,
    *,
    method: str,
    niter: int,
    blocksize: int,
    init: str,
    fdk_filter: Optional[str],
    lmbda: float,
    lmbda_red: float,
    tvlambda: float,
    tviter: int,
    verbose: bool,
) -> np.ndarray:
    """逐视角 offOrigin 的 SART / OS-SART / OS-SART+TV（TIGRE 布局体数据）。"""
    geo = copy.deepcopy(geo)
    _scalarize_geo_dsd_dso(geo)

    angles = np.asarray(angles, dtype=np.float32).reshape(-1)
    geo.check_geo(angles)
    geo.checknans()
    geo.angles = angles

    init_arg = _resolve_init_volume(
        projs, geo, angles, init=init, fdk_filter=fdk_filter, verbose=verbose
    )

    common = dict(
        lmbda=float(lmbda),
        lmbda_red=float(lmbda_red),
        verbose=verbose,
        OrderStrategy="ordered",
    )

    if method == "sart":
        vol, _ = algs.sart(
            projs,
            geo,
            angles,
            int(niter),
            init=init_arg,
            **common,
        )
    elif method == "ossart":
        vol = algs.ossart(
            projs,
            geo,
            angles,
            int(niter),
            init=init_arg,
            blocksize=int(blocksize),
            **common,
        )
    elif method == "ossart_tv":
        vol = algs.ossart_tv(
            projs,
            geo,
            angles,
            int(niter),
            init=init_arg,
            blocksize=int(blocksize),
            tvlambda=float(tvlambda),
            tviter=int(tviter),
            **common,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {RECON_METHOD_CHOICES}."
        )

    return np.asarray(vol, dtype=np.float32)


def reconstruct_helical_sart(
    scanner_cfg: Dict[str, Any],
    views: SpiralViewList,
    projs: np.ndarray,
    *,
    method: str = "ossart_tv",
    niter: int = 8,
    blocksize: int = 20,
    init: str = "helical_fdk",
    fdk_filter: Optional[str] = None,
    lmbda: float = 1.0,
    lmbda_red: float = 0.999,
    tvlambda: float = 30.0,
    tviter: int = 25,
    flip_u: bool = True,
    flip_v: bool = True,
    verbose: bool = False,
) -> SartTuneResult:
    """螺旋迭代伪 GT 主路径（默认 ``ossart_tv`` + ``helical_fdk`` 初值）。"""
    angles = views.angles_rad.astype(np.float32)
    projs_tigre = prepare_projs_for_tigre(projs, flip_u=flip_u, flip_v=flip_v)
    geo = geo_with_per_view_offorigin(get_geometry_tigre(scanner_cfg), views.z_scene)

    vol_tigre = run_helical_iterative(
        projs_tigre,
        geo,
        angles,
        method=method,
        niter=niter,
        blocksize=blocksize,
        init=init,
        fdk_filter=fdk_filter,
        lmbda=lmbda,
        lmbda_red=lmbda_red,
        tvlambda=tvlambda,
        tviter=tviter,
        verbose=verbose,
    )
    vol = vol_to_storage(vol_tigre, bool(scanner_cfg.get("coord_left", False)))
    report = {
        "method": f"helical_per_view_{method}",
        "recon_method": method,
        "n_views": len(views.stems),
        "niter": int(niter),
        "blocksize": int(blocksize),
        "init": init,
        "fdk_filter": fdk_filter,
        "lmbda": float(lmbda),
        "lmbda_red": float(lmbda_red),
        "tvlambda": float(tvlambda),
        "tviter": int(tviter),
        "flip_u": flip_u,
        "flip_v": flip_v,
        "z_sign_flipped": views.z_sign_flipped,
        "angle_range_rad": [float(angles.min()), float(angles.max())],
        "z_scene_range": [float(views.z_scene.min()), float(views.z_scene.max())],
    }
    return SartTuneResult(volume=vol, report=report)


reconstruct_spiral_sart = reconstruct_helical_sart  # 兼容旧名


# -----------------------------------------------------------------------------
# 评估：与 spiral_fdk_tune 相同（剖面图 + Ax 重投影）
# -----------------------------------------------------------------------------
#
# ``evaluate_pseudo_gt`` 等由 ``spiral_fdk_tune`` 实现并在此 re-export，
# 保证 SART / FDK tune 输出目录结构一致：
#
#   <out_dir>/eval/slices/
#   <out_dir>/eval/reproj_compare/
#   <out_dir>/eval/eval_report.json
