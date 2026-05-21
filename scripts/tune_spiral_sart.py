#!/usr/bin/env python3
"""
螺旋 CT 伪 GT：逐视角床台 z 的 TIGRE 迭代重建（默认 ossart_tv）。

最简用法（LDCT-L004 默认路径）::

    python scripts/tune_spiral_sart.py

换数据集：改 ``SartDefaults`` 中的相对路径，或传 ``--dataset`` / ``--geometry_json``。
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

from data_generator_usr.real_dataset.spiral_sart_tune import (
    FILTER_CHOICES,
    INIT_CHOICES,
    RECON_METHOD_CHOICES,
    SartTuneResult,
    SpiralViewList,
    evaluate_pseudo_gt,
    load_projections,
    load_spiral_views,
    normalize_volume,
    reconstruct_helical_sart,
    resolve_geometry_json,
    save_volume_preview,
    validate_volume,
)


# -----------------------------------------------------------------------------
# 默认配置 — 换项目时主要改这里
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SartDefaults:
    """LDCT-L004 螺旋数据集 + ossart_tv 默认参数（对齐 generate_data_usr）。"""

    # 路径（相对仓库根目录）
    dataset_relpath: str = "data/LDCT-L004/real_dataset/ldct_l004_spiral_nt200_projno_rescale"
    geometry_relpath: str = "data/LDCT-L004/SPIRAL_processed_rect/scanner_geometry.json"
    run_name: str = "iter01_ossart_tv_skip5_flipu"

    # 加载
    object_scale: float = 50.0
    proj_subsample: int = 1
    view_skip: int = 5

    # 迭代重建
    method: str = "ossart_tv"
    niter: int = 8
    blocksize: int = 20
    init: str = "helical_fdk"
    fdk_filter: str = "shepp_logan"
    lmbda: float = 1.0
    lmbda_red: float = 0.999
    tvlambda: float = 30.0
    tviter: int = 25

    # 探测器翻转
    flip_u: bool = True
    flip_v: bool = True

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
        out_dir = osp.join(dataset, "sart_tune", self.run_name)
        return dataset, geometry, out_dir


def _tigre_filter_name(cli_filter: str) -> Optional[str]:
    """CLI 滤波名 → TIGRE：``ram_lak`` 对应 ``None``（仅用于 helical_fdk 初值）。"""
    return None if cli_filter == "ram_lak" else cli_filter


# -----------------------------------------------------------------------------
# 命令行
# -----------------------------------------------------------------------------


def build_parser(defaults: Optional[SartDefaults] = None) -> argparse.ArgumentParser:
    d = defaults or SartDefaults()
    dataset_def, geometry_def, _ = d.paths()

    parser = argparse.ArgumentParser(
        description="螺旋迭代伪 GT（helical_per_view SART/OS-SART/TV）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="无参数运行 ≈ ossart_tv, niter=8, view_skip=5, helical_fdk 初值。",
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
        help="输出目录；默认 <dataset>/sart_tune/<run_name>",
    )
    g_paths.add_argument("--run_name", default=d.run_name, help="sart_tune 子目录名")

    g_load = parser.add_argument_group("加载")
    g_load.add_argument("--object_scale", type=float, default=d.object_scale)
    g_load.add_argument("--proj_subsample", type=int, default=d.proj_subsample)
    g_load.add_argument("--view_skip", type=int, default=d.view_skip, help="每隔 N 张取 1 张")

    g_recon = parser.add_argument_group("迭代重建")
    g_recon.add_argument("--method", default=d.method, choices=RECON_METHOD_CHOICES)
    g_recon.add_argument("--niter", type=int, default=d.niter)
    g_recon.add_argument("--blocksize", type=int, default=d.blocksize)
    g_recon.add_argument("--init", default=d.init, choices=INIT_CHOICES)
    g_recon.add_argument(
        "--fdk_filter",
        default=d.fdk_filter,
        choices=FILTER_CHOICES,
        help="init=helical_fdk 时 FDK 初值的滤波",
    )
    g_recon.add_argument("--lmbda", type=float, default=d.lmbda)
    g_recon.add_argument("--lmbda_red", type=float, default=d.lmbda_red)
    g_recon.add_argument("--tvlambda", type=float, default=d.tvlambda)
    g_recon.add_argument("--tviter", type=int, default=d.tviter)
    g_recon.add_argument(
        "--sanity_max",
        type=float,
        default=100.0,
        help="重建后体素最大值上限",
    )
    g_recon.add_argument("--verbose", action="store_true", help="TIGRE 详细日志")

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
        help="保存 vol_sart_preview.png",
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
        "--vol_sart",
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
        args.out_dir or osp.join(dataset_dir, "sart_tune", args.run_name)
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


def run_sart(
    data: LoadedData,
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
    flip_u: bool,
    flip_v: bool,
    normalize: str,
    percentile_hi: float,
    sanity_max: float,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """迭代重建 → 校验 → 归一化 → 保存 ``vol_sart.npy``。"""
    result: SartTuneResult = reconstruct_helical_sart(
        data.scanner_cfg,
        data.views,
        data.projs,
        method=method,
        niter=niter,
        blocksize=blocksize,
        init=init,
        fdk_filter=fdk_filter,
        lmbda=lmbda,
        lmbda_red=lmbda_red,
        tvlambda=tvlambda,
        tviter=tviter,
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

    vol_path = osp.join(data.out_dir, "vol_sart.npy")
    np.save(vol_path, vol)
    print(
        f"[save] {vol_path}  range=[{vol.min():.3f}, {vol.max():.3f}]",
        flush=True,
    )
    return vol, report, vol_path


def load_volume_for_eval(
    data: LoadedData,
    vol_sart_arg: Optional[str],
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """``--eval_only``：从 ``--vol_sart`` 或 ``vol_gt.npy`` 读体数据（与 fdk tune 一致）。"""
    vol_path = vol_sart_arg or osp.join(data.dataset_dir, "vol_gt.npy")
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
    fdk_filter = (
        _tigre_filter_name(args.fdk_filter) if args.init == "helical_fdk" else None
    )

    data = load_data(args)
    print(f"[tune] dataset   {data.dataset_dir}", flush=True)
    print(f"[tune] geometry  {data.geometry_json}", flush=True)
    print(f"[tune] out_dir   {data.out_dir}", flush=True)
    print(
        f"[load] {len(data.views.stems)} views  shape={data.projs.shape}  "
        f"method={args.method}  niter={args.niter}  "
        f"flip_u={flip_u}  flip_v={flip_v}",
        flush=True,
    )

    # 1. 重建或仅加载体数据
    if args.eval_only:
        vol, _, report = load_volume_for_eval(data, args.vol_sart)
    else:
        vol, report, _ = run_sart(
            data,
            method=args.method,
            niter=args.niter,
            blocksize=args.blocksize,
            init=args.init,
            fdk_filter=fdk_filter,
            lmbda=args.lmbda,
            lmbda_red=args.lmbda_red,
            tvlambda=args.tvlambda,
            tviter=args.tviter,
            flip_u=flip_u,
            flip_v=flip_v,
            normalize=args.normalize,
            percentile_hi=args.percentile_hi,
            sanity_max=args.sanity_max,
            verbose=args.verbose,
        )

    # 2. 剖面图 + Ax 重投影指标（与 fdk_tune 相同）
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
    report_path = osp.join(data.out_dir, "sart_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[save] {report_path}", flush=True)

    if args.preview:
        save_volume_preview(vol, osp.join(data.out_dir, "vol_sart_preview.png"))

    if args.write_vol_gt and not args.eval_only:
        write_vol_gt(data.dataset_dir, vol)


if __name__ == "__main__":
    main()
