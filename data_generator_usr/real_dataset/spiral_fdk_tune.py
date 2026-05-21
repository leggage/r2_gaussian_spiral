"""
螺旋 CT 的 TIGRE FDK 伪 GT（iter08：helical_per_view）。

流程
----
1. ``scanner_geometry.json`` → 角度、床台 z；``meta_data`` 对齐 z 符号
2. ``proj_all/*.npy`` → 可选行/列翻转后送入 TIGRE
3. 逐视角 ``geo.offOrigin``（``-z_scene``）→ 单次 FDK
4. 可选：中心剖面图、Ax 重投影 PSNR/SSIM

探测器翻转（与 ``dataset_readers`` 一致）
------------------------------------
| 参数     | 操作                  | 轴   |
|----------|-----------------------|------|
| flip_v   | ``projs[:, ::-1, :]`` | 行   |
| flip_u   | ``projs[:, :, ::-1]`` | 列   |
"""

from __future__ import annotations

import copy
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tigre
from tigre.utilities.Atb import Atb
from tigre.utilities.filtering import filtering

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from r2_gaussian.utils.ct_utils import get_geometry_tigre

# PyVista 预览（与训练可视化习惯一致）
PLOT_VOLUME_CPOS = [
    (-458.0015547298666, -207.26124611865254, 324.4699978427509),
    (129.02644270914504, 111.50694084289574, 98.55158287937994),
    (0.0, 0.0, 79.59633400474613),
]
PLOT_VOLUME_WINDOW = [800, 1000]
PLOT_VOLUME_CMAP = "viridis"

FILTER_CHOICES = ("ram_lak", "shepp_logan", "cosine", "hamming", "hann")


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass
class SpiralViewList:
    """一个病例的螺旋视角列表（已与 meta 对齐 z 符号）。"""

    stems: List[str]           # proj_all 文件名（无 .npy）
    angles_rad: np.ndarray     # 与 stems 等长
    z_scene: np.ndarray        # 床台 z，场景单位；用于 offOrigin
    z_sign_flipped: bool       # 是否对 geometry 的 table_z 取反


@dataclass
class FdkTuneResult:
    """FDK 输出：存盘布局体数据 + 可序列化报告字段。"""

    volume: np.ndarray
    report: Dict[str, Any]


# -----------------------------------------------------------------------------
# 几何与投影加载
# -----------------------------------------------------------------------------


def mm_to_scene(value_mm: float, object_scale: float) -> float:
    """毫米 → 场景坐标（与 object_scale 一致）。"""
    return float(value_mm) / 1000.0 * object_scale


def load_geometry_json(
    geometry_json: str, object_scale: float, proj_subsample: int
) -> Dict[str, Any]:
    """解析螺旋几何 JSON，返回视角顺序、角度、z 及 scanner 覆盖项。"""
    with open(geometry_json, encoding="utf-8") as f:
        geometry = json.load(f)

    projections = geometry.get("projections", [])
    if not projections:
        raise ValueError(f"No projections in {geometry_json}")

    order: List[str] = []
    angles: List[float] = []
    z_scene: List[float] = []
    scanner = geometry.get("scanner", {})
    overrides: Dict[str, Any] = {}

    for proj in projections:
        stem = str(proj["file_stem"])
        angle_rad = proj.get("angle_rad")
        if angle_rad is None:
            angle_deg = proj.get("angle_deg")
            if angle_deg is None:
                raise ValueError(f"Projection {stem} missing angle.")
            angle_rad = float(angle_deg)
        order.append(stem)
        angles.append(float(angle_rad))
        if "table_z_mm" in proj:
            z_scene.append(mm_to_scene(proj["table_z_mm"], object_scale))
        else:
            z_scene.append(0.0)

    if "DSD_mm" in scanner:
        overrides["DSD"] = mm_to_scene(scanner["DSD_mm"], object_scale)
    if "DSO_mm" in scanner:
        overrides["DSO"] = mm_to_scene(scanner["DSO_mm"], object_scale)
    if "detector_pixel_size_mm" in scanner:
        spacing = np.array(scanner["detector_pixel_size_mm"], dtype=float).reshape(-1)
        if spacing.size == 1:
            spacing = np.repeat(spacing, 2)
        overrides["detector_spacing"] = (
            spacing * proj_subsample / 1000.0 * object_scale
        ).tolist()

    return {
        "order": order,
        "angles_rad": np.asarray(angles, dtype=np.float64),
        "z_scene": np.asarray(z_scene, dtype=np.float64),
        "scanner_overrides": overrides,
    }


def resolve_geometry_json(dataset_dir: str, explicit: Optional[str]) -> str:
    """显式路径优先，否则读 meta ``scanner.geometry_source``。"""
    if explicit:
        path = osp.abspath(explicit)
        if not osp.isfile(path):
            raise FileNotFoundError(path)
        return path

    meta_path = osp.join(dataset_dir, "meta_data.json")
    if osp.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        src = meta.get("scanner", {}).get("geometry_source")
        if src:
            for base in (
                dataset_dir,
                osp.dirname(dataset_dir),
                osp.join(osp.dirname(dataset_dir), ".."),
            ):
                cand = osp.join(base, src)
                if osp.isfile(cand):
                    return osp.abspath(cand)

    raise FileNotFoundError(
        "Pass --geometry_json or set meta_data scanner.geometry_source."
    )


def _z_sign_flip_from_meta(dataset_dir: str, z_scene: np.ndarray) -> bool:
    """是否与 generate_data_usr 一致地对 table_z 取反（见 meta spiral 段）。"""
    meta_path = osp.join(dataset_dir, "meta_data.json")
    if not osp.isfile(meta_path):
        return False
    with open(meta_path, encoding="utf-8") as f:
        spiral = json.load(f).get("spiral", {})
    if "z_shift_sign_flipped_to_decreasing" in spiral:
        return bool(spiral["z_shift_sign_flipped_to_decreasing"])
    d = np.diff(z_scene)
    return bool(d.size and np.nanmedian(d) > 0)


def load_spiral_views(
    dataset_dir: str,
    geometry_json: str,
    object_scale: float,
    proj_subsample: int,
    view_skip: int = 1,
) -> SpiralViewList:
    """几何 + z 符号 + ``view_skip`` 子采样；检查 ``proj_all`` 文件存在。"""
    geom = load_geometry_json(geometry_json, object_scale, proj_subsample)
    z_scene = geom["z_scene"].copy()

    z_sign_flipped = _z_sign_flip_from_meta(dataset_dir, z_scene)
    if z_sign_flipped:
        z_scene = -z_scene

    stems = geom["order"][::view_skip]
    angles = geom["angles_rad"][::view_skip]
    z_scene = z_scene[::view_skip]

    proj_all = osp.join(dataset_dir, "proj_all")
    for stem in stems:
        if not osp.isfile(osp.join(proj_all, stem + ".npy")):
            raise FileNotFoundError(f"Missing {proj_all}/{stem}.npy")

    return SpiralViewList(
        stems=stems,
        angles_rad=angles,
        z_scene=z_scene,
        z_sign_flipped=z_sign_flipped,
    )


def load_projections(dataset_dir: str, views: SpiralViewList) -> np.ndarray:
    """堆叠为 ``(n_views, n_rows, n_cols)`` float32。"""
    proj_all = osp.join(dataset_dir, "proj_all")
    stack = [
        np.load(osp.join(proj_all, s + ".npy")).astype(np.float32) for s in views.stems
    ]
    return np.stack(stack, axis=0)


# -----------------------------------------------------------------------------
# TIGRE：投影预处理 + 逐视角 FDK
# -----------------------------------------------------------------------------


def geo_with_per_view_offorigin(base_geo, z_scene: np.ndarray):
    """``train_offOrigin`` 为 ``[-z_shift, 0, 0]`` → TIGRE 加 ``-z_scene``。"""
    z = np.asarray(z_scene, dtype=np.float32).reshape(-1)
    offsets = np.stack([-z, np.zeros_like(z), np.zeros_like(z)], axis=1)
    geo = copy.deepcopy(base_geo)
    base_origin = np.asarray(base_geo.offOrigin, dtype=np.float32)
    geo.offOrigin = offsets + base_origin.reshape(1, 3)
    return geo


def prepare_projs_for_tigre(projs: np.ndarray, flip_u: bool, flip_v: bool) -> np.ndarray:
    """磁盘投影 → TIGRE 使用的朝向（默认 flip_v 行、可选 flip_u 列）。"""
    out = projs.astype(np.float32, copy=True)
    if flip_v:
        out = out[:, ::-1, :]
    if flip_u:
        out = out[:, :, ::-1]
    return out


def run_fdk(
    projs: np.ndarray,
    geo,
    angles: np.ndarray,
    filter_name: Optional[str],
    verbose: bool,
) -> np.ndarray:
    """TIGRE FDK：cos 权重 → 滤波 → ``Atb``（体数据为 TIGRE 布局）。"""
    geo = copy.deepcopy(geo)
    # 螺旋 geo 上 DSD/DSO 可能为 per-view，FDK 路径需标量
    for attr in ("DSD", "DSO"):
        val = getattr(geo, attr, None)
        if val is not None:
            scalar = float(np.asarray(val, dtype=np.float64).reshape(-1)[0])
            setattr(geo, attr, np.array([scalar], dtype=np.float32))

    angles = np.asarray(angles, dtype=np.float32).reshape(-1)
    geo.check_geo(angles)
    geo.checknans()
    geo.filter = filter_name
    geo.angles = angles

    proj_filt = np.zeros(projs.shape, dtype=np.float32)
    xv = np.arange(
        (-geo.nDetector[1] / 2) + 0.5, 1 + (geo.nDetector[1] / 2) - 0.5
    ) * geo.dDetector[1]
    yv = np.arange(
        (-geo.nDetector[0] / 2) + 0.5, 1 + (geo.nDetector[0] / 2) - 0.5
    ) * geo.dDetector[0]
    yy, xx = np.meshgrid(xv, yv)
    dsd0 = float(np.asarray(geo.DSD).reshape(-1)[0])
    w = dsd0 / np.sqrt(dsd0**2 + xx**2 + yy**2)  # 锥束 cos 权重
    np.multiply(projs, w, out=proj_filt)
    # proj_filt = filtering(projs, geo, geo.angles, parker=False, verbose=verbose)
    proj_filt = filtering(proj_filt, geo, geo.angles, parker=False, verbose=verbose)
    return Atb(proj_filt, geo, geo.angles, "FDK")


def reconstruct_helical_fdk(
    scanner_cfg: Dict[str, Any],
    views: SpiralViewList,
    projs: np.ndarray,
    *,
    filter_name: Optional[str],
    flip_u: bool,
    flip_v: bool,
    verbose: bool,
) -> FdkTuneResult:
    """iter08 主路径：逐视角 offOrigin + 单次 FDK。"""
    angles = views.angles_rad.astype(np.float32)
    projs_tigre = prepare_projs_for_tigre(projs, flip_u=flip_u, flip_v=flip_v)
    geo = geo_with_per_view_offorigin(get_geometry_tigre(scanner_cfg), views.z_scene)
    vol_tigre = run_fdk(projs_tigre, geo, angles, filter_name=filter_name, verbose=verbose)
    vol = vol_to_storage(vol_tigre, bool(scanner_cfg.get("coord_left", False)))
    report = {
        "method": "helical_per_view",
        "n_views": len(views.stems),
        "filter": filter_name,
        "flip_u": flip_u,
        "flip_v": flip_v,
        "z_sign_flipped": views.z_sign_flipped,
        "angle_range_rad": [float(angles.min()), float(angles.max())],
        "z_scene_range": [float(views.z_scene.min()), float(views.z_scene.max())],
    }
    return FdkTuneResult(volume=vol, report=report)


reconstruct_spiral_fdk = reconstruct_helical_fdk  # 兼容旧名


# -----------------------------------------------------------------------------
# 体数据：TIGRE 布局 ↔ 存盘布局，归一化
# -----------------------------------------------------------------------------


def vol_to_storage(vol: np.ndarray, coord_left: bool) -> np.ndarray:
    """TIGRE (z,y,x) → 训练存盘 (x,y,z)；``coord_left`` 时翻转 x。"""
    vol = np.asarray(vol, dtype=np.float32).transpose(2, 1, 0)
    if coord_left:
        vol = vol[::-1, :, :].copy()
    return np.maximum(vol, 0.0)


def storage_to_tigre(vol: np.ndarray, coord_left: bool) -> np.ndarray:
    """``vol_to_storage`` 的逆变换，供 Ax 重投影使用。"""
    v = np.asarray(vol, dtype=np.float32)
    if coord_left:
        v = v[::-1, :, :].copy()
    return np.transpose(v, (2, 1, 0)).copy()


def normalize_volume(
    vol: np.ndarray,
    mode: str,
    percentile_hi: float = 99.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """将体数据缩放到 [0, 1]；返回 (vol, 统计字段)。"""
    stats = {
        "raw_min": float(vol.min()),
        "raw_max": float(vol.max()),
        "raw_mean": float(vol.mean()),
    }
    if mode == "none":
        return vol, stats
    positive = vol[vol > 0]
    if positive.size == 0:
        stats.update({"norm_min": 0.0, "norm_max": 0.0})
        return vol, stats
    if mode == "percentile":
        hi = float(np.percentile(positive, percentile_hi))
    elif mode == "max":
        hi = float(positive.max())
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    out = np.clip(vol / max(hi, 1e-8), 0.0, 1.0).astype(np.float32)
    stats.update({"norm_hi": hi, "norm_lo": 0.0, "norm_max": float(out.max())})
    return out, stats


def validate_volume(vol: np.ndarray, max_val: float = 100.0) -> None:
    """拒绝 NaN/Inf 或异常大的重建（几何/归一化错误时常见）。"""
    if not np.isfinite(vol).all():
        raise RuntimeError("Reconstruction contains NaN/Inf.")
    if float(vol.max()) > max_val:
        raise RuntimeError(
            f"Volume max={float(vol.max()):.4g} exceeds {max_val} — check geometry or normalization."
        )


def save_volume_preview(
    volume: np.ndarray, save_path: str, clim: Tuple[float, float] = (0.0, 1.0)
) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("[preview] pyvista not installed, skip.", flush=True)
        return
    vol_show = volume.copy()
    vol_show[: vol_show.shape[0] // 2, :, :] = 0
    plotter = pv.Plotter(window_size=PLOT_VOLUME_WINDOW, line_smoothing=True, off_screen=True)
    plotter.add_volume(vol_show, cmap=PLOT_VOLUME_CMAP, opacity="linear", clim=list(clim))
    plotter.add_axes()
    plotter.camera_position = PLOT_VOLUME_CPOS
    plotter.show(screenshot=save_path)
    plotter.close()
    print(f"[preview] {save_path}", flush=True)


def _volume_display_range(vol: np.ndarray) -> Tuple[float, float]:
    positive = vol[vol > 0]
    if positive.size == 0:
        return 0.0, 1.0
    return 0.0, float(np.percentile(positive, 99.5))


# -----------------------------------------------------------------------------
# 评估：剖面图 + 重投影
# -----------------------------------------------------------------------------


def save_volume_slice_profiles(
    vol: np.ndarray, out_dir: str, n_slices: int = 9
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    vmin, vmax = _volume_display_range(vol)
    nx, ny, nz = vol.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    report: Dict[str, Any] = {"vmin": vmin, "vmax": vmax, "shape": list(vol.shape)}

    center_slices = {
        "axial_xy": vol[:, :, cz],
        "coronal_xz": vol[:, cy, :],
        "sagittal_yz": vol[cx, :, :],
    }
    for name, slc in center_slices.items():
        plt.imsave(
            osp.join(out_dir, f"center_{name}.png"),
            slc.T,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        report[f"center_{name}"] = f"center_{name}.png"

    fig, axes_plt = plt.subplots(1, 3, figsize=(12, 4))
    for ax_plt, (name, slc) in zip(axes_plt, center_slices.items()):
        ax_plt.imshow(slc.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax_plt.set_title(name)
        ax_plt.axis("off")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "center_slices_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    report["center_slices_combined"] = "center_slices_combined.png"
    print(f"[slices] saved under {out_dir}", flush=True)
    return report


def forward_project_helical(
    vol: np.ndarray,
    scanner_cfg: Dict[str, Any],
    views: SpiralViewList,
    view_indices: np.ndarray,
    flip_u: bool,
    flip_v: bool,
) -> np.ndarray:
    """逐视角 offOrigin 的 ``tigre.Ax``；输出与 ``prepare_projs_for_tigre`` 同朝向。"""
    geo = geo_with_per_view_offorigin(
        get_geometry_tigre(scanner_cfg), views.z_scene[view_indices]
    )
    vol_t = storage_to_tigre(vol, scanner_cfg.get("coord_left", False))
    angles = views.angles_rad[view_indices].astype(np.float32)
    projs_pred = np.asarray(tigre.Ax(vol_t, geo, angles), dtype=np.float32)
    # if flip_v:
    #     projs_pred = projs_pred[:, ::-1, :]
    # if flip_u:
    #     projs_pred = projs_pred[:, :, ::-1]
    return projs_pred


def evaluate_reprojection(
    vol: np.ndarray,
    scanner_cfg: Dict[str, Any],
    views: SpiralViewList,
    projs_gt: np.ndarray,
    *,
    flip_u: bool,
    flip_v: bool,
    max_views: int = 100,
    save_compare: int = 8,
    out_dir: str,
) -> Dict[str, Any]:
    from r2_gaussian.utils.image_utils import metric_proj

    n = len(views.stems)
    eval_idx = np.linspace(0, n - 1, min(max(max_views, 1), n)).astype(int)
    projs_gt_t = prepare_projs_for_tigre(projs_gt[eval_idx], flip_u=flip_u, flip_v=flip_v)
    print(f"[reproj] Ax {len(eval_idx)} / {n} views ...", flush=True)
    projs_pred = forward_project_helical(
        vol, scanner_cfg, views, eval_idx, flip_u=flip_u, flip_v=flip_v
    )

    gt = projs_gt_t.astype(np.float64)
    pred = projs_pred.astype(np.float64)

    # 逐视角最小二乘尺度，再算物理 PSNR（与训练投影幅值可比）
    psnr_phys, scales = [], []
    for i in range(gt.shape[0]):
        g, p = gt[i].ravel(), pred[i].ravel()
        denom = float(np.dot(p, p))
        scale = float(np.dot(g, p)) / denom if denom > 1e-12 else 0.0
        scales.append(scale)
        mse = float(np.mean((gt[i] - pred[i] * scale) ** 2))
        peak = max(float(gt[i].max()), 1e-8)
        psnr_phys.append(100.0 if mse < 1e-12 else 10.0 * np.log10(peak * peak / mse))

    psnr_sn, psnr_list = metric_proj(
        gt.astype(np.float32), pred.astype(np.float32), metric="psnr", axis=0, pixel_max=1.0
    )
    ssim_sn, ssim_list = metric_proj(
        gt.astype(np.float32), pred.astype(np.float32), metric="ssim", axis=0
    )

    report = {
        "n_total_views": n,
        "n_eval_views": int(len(eval_idx)),
        "eval_indices": eval_idx.tolist(),
        "proj_psnr_mean": float(np.mean(psnr_phys)),
        "proj_psnr_per_view": psnr_phys,
        "proj_ssim_mean": float(ssim_sn),
        "proj_ssim_per_view": [float(x) for x in ssim_list],
        "proj_psnr_shape_norm": float(psnr_sn),
        "ls_scale_per_view": scales,
    }
    print(
        f"[reproj] PSNR={report['proj_psnr_mean']:.2f} dB, SSIM={report['proj_ssim_mean']:.3f}",
        flush=True,
    )

    if save_compare > 0:
        import matplotlib.pyplot as plt

        cmp_dir = osp.join(out_dir, "reproj_compare")
        os.makedirs(cmp_dir, exist_ok=True)
        for j in np.linspace(0, len(eval_idx) - 1, min(save_compare, len(eval_idx))).astype(int):
            gt_i, pred_i = projs_gt_t[j], projs_pred[j]
            stem = views.stems[int(eval_idx[j])]
            vmax = max(float(gt_i.max()), float(pred_i.max()), 1e-6)
            for tag, arr in (("gt", gt_i), ("pred", pred_i)):
                plt.imsave(
                    osp.join(cmp_dir, f"{stem}_{tag}.png"),
                    arr,
                    cmap="gray",
                    vmin=0,
                    vmax=vmax,
                )
            diff = np.abs(pred_i - gt_i)
            plt.imsave(
                osp.join(cmp_dir, f"{stem}_diff.png"),
                diff,
                cmap="hot",
                vmin=0,
                vmax=float(diff.max()) or 1.0,
            )
        report["reproj_compare_dir"] = osp.relpath(cmp_dir, out_dir)

    return report


def evaluate_pseudo_gt(
    vol: np.ndarray,
    scanner_cfg: Dict[str, Any],
    views: SpiralViewList,
    projs_gt: np.ndarray,
    out_dir: str,
    *,
    flip_u: bool,
    flip_v: bool,
    n_slice_montage: int = 9,
    max_reproj_views: int = 100,
    save_reproj_compare: int = 8,
) -> Dict[str, Any]:
    """写 ``<out_dir>/eval/``：剖面图 + 重投影报告 + ``eval_report.json``。"""
    eval_root = osp.join(out_dir, "eval")
    os.makedirs(eval_root, exist_ok=True)
    slices = save_volume_slice_profiles(
        vol, osp.join(eval_root, "slices"), n_slices=n_slice_montage
    )
    reproj = evaluate_reprojection(
        vol,
        scanner_cfg,
        views,
        projs_gt,
        flip_u=flip_u,
        flip_v=flip_v,
        max_views=max_reproj_views,
        save_compare=save_reproj_compare,
        out_dir=eval_root,
    )
    report = {"slices": slices, "reprojection": reproj}
    with open(osp.join(eval_root, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
