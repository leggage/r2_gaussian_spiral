#!/usr/bin/env python3
"""
螺旋 CT 伪 GT：逐视角床台 z 的 TIGRE FDK（iter08 配方）。

最简用法（LDCT-L004 默认路径）::

    python scripts/tune_spiral_fdk.py

换数据集：改 ``Iter08Defaults`` 中的相对路径，或传 ``--dataset`` / ``--geometry_json``。
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

_REPO = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data_generator_usr.real_dataset.spiral_fdk_tune import (
    FILTER_CHOICES,
    FdkTuneResult,
    SpiralViewList,
    evaluate_pseudo_gt,
    load_projections,
    load_spiral_views,
    normalize_volume,
    reconstruct_helical_fdk,
    resolve_geometry_json,
    save_volume_preview,
    validate_volume,
)


# -----------------------------------------------------------------------------
# 默认配置（iter08）— 换项目时主要改这里
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Iter08Defaults:
    """LDCT-L004 螺旋数据集 + iter08 重建参数的默认值。"""

    # 路径（相对仓库根目录）
    dataset_relpath: str = "data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200_projno_rescale"
    geometry_relpath: str = "data/LDCT-L004/SPIRAL_processed_rect/scanner_geometry.json"
    run_name: str = "iter08_helical_skip2_flipu"

    # FDK
    object_scale: float = 50.0
    proj_subsample: int = 1
    view_skip: int = 2
    filter: str = "shepp_logan"

    # 探测器翻转（须与 dataset_readers / 训练一致）
    flip_u: bool = True   # 列：projs[:, :, ::-1]，coord_left 数据集
    flip_v: bool = True   # 行：projs[:, ::-1, :]

    # 体数据后处理
    normalize: str = "percentile"
    percentile_hi: float = 99.5

    # 评估与预览
    preview: bool = True
    eval_max_views: int = 100
    eval_reproj_compare: int = 8

    def paths(self, repo_root: str = _REPO) -> Tuple[str, str, str]:
        """返回 ``(dataset_dir, geometry_json, out_dir)`` 绝对路径。"""
        dataset = osp.join(repo_root, self.dataset_relpath)
        geometry = osp.join(repo_root, self.geometry_relpath)
        out_dir = osp.join(dataset, "fdk_tune", self.run_name)
        return dataset, geometry, out_dir


def _tigre_filter_name(cli_filter: str) -> Optional[str]:
    """CLI 滤波名 → TIGRE：``ram_lak`` 对应 ``None``（默认 Ram-Lak）。"""
    return None if cli_filter == "ram_lak" else cli_filter


# -----------------------------------------------------------------------------
# 命令行
# -----------------------------------------------------------------------------


def build_parser(defaults: Optional[Iter08Defaults] = None) -> argparse.ArgumentParser:
    d = defaults or Iter08Defaults()
    dataset_def, geometry_def, _ = d.paths()

    parser = argparse.ArgumentParser(
        description="螺旋 FDK 伪 GT（helical_per_view，iter08 默认参数）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="无参数运行 ≈ iter08：view_skip=5, shepp_logan, flip_u, preview。",
    )

    g_paths = parser.add_argument_group("数据路径")
    g_paths.add_argument(
        "--dataset",
        default=dataset_def,
        metavar="DIR",
        help="含 meta_data.json 与 proj_all/ 的病例目录",
    )
    g_paths.add_argument(
        "--geometry_json",
        default=geometry_def,
        metavar="JSON",
        help="螺旋几何（projections + table_z_mm）",
    )
    g_paths.add_argument(
        "--out_dir",
        default=None,
        metavar="DIR",
        help="输出目录；默认 <dataset>/fdk_tune/<run_name>",
    )
    g_paths.add_argument("--run_name", default=d.run_name, help="fdk_tune 子目录名")

    g_fdk = parser.add_argument_group("FDK 重建")
    g_fdk.add_argument("--object_scale", type=float, default=d.object_scale)
    g_fdk.add_argument("--proj_subsample", type=int, default=d.proj_subsample)
    g_fdk.add_argument(
        "--view_skip",
        type=int,
        default=d.view_skip,
        help="每隔 N 张取 1 张投影",
    )
    g_fdk.add_argument("--filter", default=d.filter, choices=FILTER_CHOICES)
    g_fdk.add_argument(
        "--sanity_max",
        type=float,
        default=100.0,
        help="重建后体素最大值上限",
    )
    g_fdk.add_argument("--verbose", action="store_true", help="TIGRE 详细日志")

    g_det = parser.add_argument_group("探测器朝向（与训练一致）")
    g_det.add_argument(
        "--flip_u",
        action=argparse.BooleanOptionalAction,
        default=d.flip_u,
        help="列翻转 projs[:, :, ::-1]（coord_left）",
    )
    g_det.add_argument(
        "--no_flip_v",
        action="store_true",
        help="关闭行翻转 projs[:, ::-1, :]（默认开启）",
    )

    g_out = parser.add_argument_group("输出与归一化")
    g_out.add_argument(
        "--normalize",
        choices=["percentile", "max", "none"],
        default=d.normalize,
    )
    g_out.add_argument("--percentile_hi", type=float, default=d.percentile_hi)
    g_out.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=d.preview,
        help="保存 vol_fdk_preview.png",
    )
    g_out.add_argument(
        "--write_vol_gt",
        action="store_true",
        help="写入 <dataset>/vol_gt.npy（原文件备份为 vol_gt.bak.npy）",
    )

    g_ev = parser.add_argument_group("评估")
    g_ev.add_argument("--skip_eval", action="store_true", help="跳过剖面图与重投影")
    g_ev.add_argument("--eval_only", action="store_true", help="不重建，仅评估已有体数据")
    g_ev.add_argument(
        "--vol_fdk",
        default=None,
        metavar="NPY",
        help="--eval_only 时使用的体数据路径",
    )
    g_ev.add_argument("--eval_max_views", type=int, default=d.eval_max_views)
    g_ev.add_argument("--eval_reproj_compare", type=int, default=d.eval_reproj_compare)

    return parser


# -----------------------------------------------------------------------------
# 运行流程
# -----------------------------------------------------------------------------


@dataclass
class LoadedData:
    """一次运行所需的已加载数据。"""

    dataset_dir: str
    scanner_cfg: Dict[str, Any]
    geometry_json: str
    out_dir: str
    views: SpiralViewList
    projs: np.ndarray


def load_data(args: argparse.Namespace) -> LoadedData:
    """读 meta、解析几何路径、加载视角列表与投影栈。"""
    dataset_dir = osp.abspath(args.dataset)
    meta_path = osp.join(dataset_dir, "meta_data.json")
    if not osp.isfile(meta_path):
        raise FileNotFoundError(f"缺少 meta_data.json: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        scanner_cfg = json.load(f)["scanner"]

    geometry_json = resolve_geometry_json(dataset_dir, args.geometry_json)
    out_dir = osp.abspath(
        args.out_dir or osp.join(dataset_dir, "fdk_tune", args.run_name)
    )
    os.makedirs(out_dir, exist_ok=True)

    views = load_spiral_views(
        dataset_dir,
        geometry_json,
        args.object_scale,
        args.proj_subsample,
        args.view_skip,
    )
    projs = load_projections(dataset_dir, views)

    return LoadedData(
        dataset_dir=dataset_dir,
        scanner_cfg=scanner_cfg,
        geometry_json=geometry_json,
        out_dir=out_dir,
        views=views,
        projs=projs,
    )


def run_fdk(
    data: LoadedData,
    *,
    filter_name: Optional[str],
    flip_u: bool,
    flip_v: bool,
    normalize: str,
    percentile_hi: float,
    sanity_max: float,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """FDK 重建 → 校验 → 归一化 → 保存 ``vol_fdk.npy``。"""
    result: FdkTuneResult = reconstruct_helical_fdk(
        data.scanner_cfg,
        data.views,
        data.projs,
        filter_name=filter_name,
        flip_u=flip_u,
        flip_v=flip_v,
        verbose=verbose,
    )
    validate_volume(result.volume, max_val=sanity_max)

    vol, norm_stats = normalize_volume(result.volume, normalize, percentile_hi)
    report: Dict[str, Any] = {
        **result.report,
        "normalize": {"mode": normalize, **norm_stats},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    vol_path = osp.join(data.out_dir, "vol_fdk.npy")
    np.save(vol_path, vol)
    print(
        f"[save] {vol_path}  range=[{vol.min():.3f}, {vol.max():.3f}]",
        flush=True,
    )
    return vol, report, vol_path


def load_volume_for_eval(
    data: LoadedData,
    vol_fdk_arg: Optional[str],
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """``--eval_only``：从 ``--vol_fdk`` 或 ``vol_gt.npy`` 读体数据。"""
    vol_path = vol_fdk_arg or osp.join(data.dataset_dir, "vol_gt.npy")
    if not osp.isfile(vol_path):
        raise FileNotFoundError(vol_path)
    vol = np.load(vol_path).astype(np.float32)
    report = {
        "eval_only": True,
        "volume_source": osp.abspath(vol_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return vol, vol_path, report


def write_vol_gt(dataset_dir: str, vol: np.ndarray) -> None:
    """将伪 GT 写回病例目录（先备份已有 ``vol_gt.npy``）。"""
    dst = osp.join(dataset_dir, "vol_gt.npy")
    if osp.isfile(dst):
        bak = osp.join(dataset_dir, "vol_gt.bak.npy")
        os.replace(dst, bak)
        print(f"[backup] vol_gt.npy -> vol_gt.bak.npy", flush=True)
    np.save(dst, vol)
    print(f"[write_vol_gt] {dst}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    flip_u = args.flip_u
    flip_v = not args.no_flip_v

    data = load_data(args)
    print(f"[tune] dataset   {data.dataset_dir}", flush=True)
    print(f"[tune] geometry  {data.geometry_json}", flush=True)
    print(f"[tune] out_dir   {data.out_dir}", flush=True)
    print(
        f"[load] {len(data.views.stems)} views  shape={data.projs.shape}  "
        f"z=[{data.views.z_scene.min():.3f}, {data.views.z_scene.max():.3f}]  "
        f"flip_u={flip_u}  flip_v={flip_v}",
        flush=True,
    )

    # 1. 重建或仅加载体数据
    if args.eval_only:
        vol, vol_path, report = load_volume_for_eval(data, args.vol_fdk)
    else:
        vol, report, vol_path = run_fdk(
            data,
            filter_name=_tigre_filter_name(args.filter),
            flip_u=flip_u,
            flip_v=flip_v,
            normalize=args.normalize,
            percentile_hi=args.percentile_hi,
            sanity_max=args.sanity_max,
            verbose=args.verbose,
        )

    # 2. 剖面图 + Ax 重投影指标
    if not args.skip_eval:
        report["eval"] = evaluate_pseudo_gt(
            vol,
            data.scanner_cfg,
            data.views,
            data.projs,
            data.out_dir,
            flip_u=flip_u,
            flip_v=flip_v,
            max_reproj_views=args.eval_max_views,
            save_reproj_compare=args.eval_reproj_compare,
        )

    # 3. 报告与可选产物
    report_path = osp.join(data.out_dir, "fdk_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[save] {report_path}", flush=True)

    if args.preview:
        save_volume_preview(vol, osp.join(data.out_dir, "vol_fdk_preview.png"))

    if args.write_vol_gt and not args.eval_only:
        write_vol_gt(data.dataset_dir, vol)


if __name__ == "__main__":
    main()
