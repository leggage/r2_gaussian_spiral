# 统一数据预处理（norm）

入口：

```bash
python data_preprocess/norm_pipeline.py \
  --config data_preprocess/configs/norm_pipeline.example.yml
```

先检查配置而不读取数据或调用 GPU：

```bash
python data_preprocess/norm_pipeline.py \
  --config data_preprocess/configs/your.yml \
  --validate-only
```

`dataset_type: syn` 读取 `raw_gt` DICOM 切片，由 YAML scanner/spiral 参数正投影；
`dataset_type: real` 同时读取 `raw_proj`，并通过 pydicom 数值 tag 直接解析 Siemens
CT-PD 私有几何字段。spiral 始终生成；stitch 默认关闭，仅在配置
`stitch.enabled: true` 时生成，并使用独立的 `stitch.n_train/n_test`。

real 数据的 DSD、DSO、探测器尺寸、pitch 和每圈采样数来自投影 DICOM；YAML
保留体素分辨率等重建设置。默认按照全部投影的 z 最小值/最大值自动更新
`scanner.sVoxel` 和 `scanner.offOrigin`，行为由 `real.*` 配置控制。

所有配置文件统一放在 `data_preprocess/configs/`。配置中的 `output_root` 建议设置为
`data`，输出目录遵循与训练结果相同的层级：

```text
data/{real|syn}/{organ}/{spiral|stitch}/ntrain{N}/{model}/
  vol_gt.npy
  init_{model}.npy
  meta_data.json
  proj_train/*.npy
  proj_test/*.npy
```

`init_*.npy` 是用该输出类型的全部投影进行 FDK 后采样的 `[x,y,z,density]` 点云，
不是 `vol_gt.npy` 的复制品。FDK/正投影依赖 TIGRE 和 CUDA。
推荐设置 `init.density_threshold: auto`：管线自动选择强度最高的候选唯一体素，
再无放回采样目标点数，避免固定阈值导致大量重复初始化点。数值阈值仍受支持；
若阈值以上的唯一体素不足，管线会报错而不会重复采样。
