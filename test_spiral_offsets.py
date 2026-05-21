#!/usr/bin/env python3
"""
测试脚本：演示新的螺旋偏移计算方式
对比两种方式：
1. 旧方式：固定 z_delta_per_view，导致 z 可能超出 volume 边界
2. 新方式：给定 z_start 和 z_end，自动计算步进，确保在 volume 范围内
"""

import numpy as np
import yaml


def _build_spiral_offsets(num_views, spiral_cfg):
    """新的螺旋偏移计算函数"""
    if not spiral_cfg:
        return None
    if spiral_cfg.get("enabled", True) is False:
        return None

    if num_views == 0:
        return None

    start = float(spiral_cfg.get("z_start", 0.0))
    
    # 新方式：使用 z_end
    if "z_end" in spiral_cfg:
        end = float(spiral_cfg.get("z_end", 0.0))
        step = (end - start) / max(1, num_views - 1) if num_views > 1 else 0.0
    else:
        # 向后兼容：使用 z_delta_per_view
        step = float(spiral_cfg.get("z_delta_per_view", 0.0))

    increments = np.arange(num_views, dtype=np.float32)
    return start + step * increments


def demo():
    print("=" * 70)
    print("螺旋投影偏移计算方式对比演示")
    print("=" * 70)
    
    # 参数设置
    num_views = 100
    sVoxel_z = 24.0  # volume z 方向尺寸
    offOrigin_z = -8.0  # volume z 中心偏移
    
    # volume z 范围：[offOrigin_z - sVoxel_z/2, offOrigin_z + sVoxel_z/2]
    z_min = offOrigin_z - sVoxel_z / 2
    z_max = offOrigin_z + sVoxel_z / 2
    
    print(f"\n体积配置：")
    print(f"  num_views = {num_views}")
    print(f"  sVoxel[z] = {sVoxel_z}")
    print(f"  offOrigin[z] = {offOrigin_z}")
    print(f"  Z 范围 = [{z_min:.2f}, {z_max:.2f}]")
    
    # 旧方式：固定步进
    print(f"\n【旧方式】固定 z_delta_per_view:")
    spiral_cfg_old = {
        "enabled": True,
        "z_start": -10.0,
        "z_delta_per_view": 0.1,
    }
    z_offsets_old = _build_spiral_offsets(num_views, spiral_cfg_old)
    print(f"  z_start = {spiral_cfg_old['z_start']}")
    print(f"  z_delta_per_view = {spiral_cfg_old['z_delta_per_view']}")
    print(f"  Z 偏移范围 = [{z_offsets_old.min():.2f}, {z_offsets_old.max():.2f}]")
    print(f"  超出范围？{z_offsets_old.max() > z_max or z_offsets_old.min() < z_min}")
    
    # 新方式：z_start 和 z_end
    print(f"\n【新方式】给定 z_start 和 z_end:")
    spiral_cfg_new = {
        "enabled": True,
        "z_start": z_min + 1.0,  # 距离下界 1.0
        "z_end": z_max - 1.0,    # 距离上界 1.0
    }
    z_offsets_new = _build_spiral_offsets(num_views, spiral_cfg_new)
    step_computed = (spiral_cfg_new["z_end"] - spiral_cfg_new["z_start"]) / (num_views - 1)
    print(f"  z_start = {spiral_cfg_new['z_start']:.2f}")
    print(f"  z_end = {spiral_cfg_new['z_end']:.2f}")
    print(f"  自动计算的 step = {step_computed:.6f}")
    print(f"  Z 偏移范围 = [{z_offsets_new.min():.2f}, {z_offsets_new.max():.2f}]")
    print(f"  在范围内？{z_offsets_new.max() <= z_max and z_offsets_new.min() >= z_min}")
    
    # 可视化对比
    print(f"\n【对比统计】")
    print(f"  旧方式最后超出范围的投影数：{np.sum(z_offsets_old > z_max)}")
    print(f"  新方式超出范围的投影数：{np.sum((z_offsets_new > z_max) | (z_offsets_new < z_min))}")
    
    print(f"\n✓ 新方式确保所有投影都在 volume 范围内，避免重建伪影或数据丢失")
    print("=" * 70)


if __name__ == "__main__":
    demo()
