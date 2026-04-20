# 第四阶段训练手册（Stage 4 Training Manual）

## 一、概述

第四阶段（Stage 4）是整个 MDVSC 流水线的 **端到端联合训练** 阶段。它将 Stage 2.1（SharedEncoder + DetRecovery + Detail Bypass + 图像重建）和 Stage 3（MDVSC v2 特征压缩）的训练成果组装为一个完整的可微分流水线，进行联合微调。

目标是让所有模块在端到端梯度回传下协同优化，消除分阶段训练引入的模块间接口不匹配问题。

整体数据流：

```
原始视频帧 [B, T, 3, 640, 640]
  → RT-DETR backbone（冻结）
    ├─ stage1: [B*T, 256, 160, 160]  ← Detail Bypass 输入
    ├─ stage2: [B*T, 512, 80, 80]
    ├─ stage3: [B*T, 1024, 40, 40]
    └─ stage4: [B*T, 2048, 20, 20]
          │                               │
          ▼                               ▼
  SharedEncoder (per-level)          DetailCompressor
    → 3×[B,T,256,H,W]                 → [B,T,32,20,20]  ~1% overhead
          │                               │
          ▼                               │
  MDVSC v2 (compress/transmit/restore)    │
    → 3×[B,T,256,H,W] (restored)         │
          │                               │
    ├─────┤                               │
    ▼     ▼                               ▼
  DetRecoveryHead    TaskAdapt×3 ←── DetailDecompressor
    → projected 256    + DetailAware ReconHead
    （检测一致性）        → [B,T,3,640,640]（重建帧）

Loss = feature_loss + det_recovery + recon_l1 + recon_mse + recon_ssim + recon_edge
```

## 二、与前序阶段的关系

| 组件 | 来源 | Stage 4 中的角色 |
|------|------|----------------|
| RT-DETR backbone | HuggingFace 预训练 | 始终冻结，提供 teacher 特征 |
| SharedEncoder | Stage 2 / Stage 2.1 | Phase 1 冻结，Phase 2/3 可训练 |
| MDVSC v2 | Stage 3 | 始终可训练 |
| DetRecoveryHead | Stage 2.1 | 始终可训练 |
| DetailCompressor/Decompressor | Stage 2.1 | 始终可训练 |
| Reconstruction heads | Stage 2.1 | 始终可训练 |

## 三、三阶段渐进解冻策略

Stage 4 采用 **渐进解冻**（Progressive Unfreezing）策略，分三个 Phase 进行训练：

### Phase 1（默认 8 epochs，lr=1e-4）

- **冻结** SharedEncoder
- **训练** MDVSC v2、DetRecoveryHead、Detail Bypass、Reconstruction heads
- 目的：让各模块在 SharedEncoder 不变的前提下适配端到端联合损失

### Phase 2（默认 8 epochs，lr=5e-5）

- **全部解冻**：所有模块参与训练
- 降低学习率，防止 SharedEncoder 快速偏移导致下游模块失配
- 目的：SharedEncoder 与 MDVSC v2 协同优化

### Phase 3（默认 4 epochs，lr=2e-5）

- **全部解冻**，极低学习率
- 目的：精细微调，收敛到最优解

## 四、文件结构

```
src/semantic_rtdetr/semantic_comm/stage4_model.py      # Stage 4 模型定义
src/semantic_rtdetr/training/stage4_config.py           # Stage 4 配置 dataclass
src/semantic_rtdetr/training/stage4_trainer.py          # Stage 4 训练器
scripts/train_mdvsc_stage4.py                           # 训练入口脚本
scripts/plot_stage4_metrics.py                          # 训练曲线绘图脚本
configs/mdvsc_stage4.yaml                               # 默认配置文件
```

## 五、前置条件

1. **RT-DETR 权重**：`pretrained/rtdetr_r50vd/` 目录下已有模型文件。
2. **Stage 2.1 Checkpoint**：已训练好的 `outputs/mdvsc_stage2_1/best.pt`，包含 SharedEncoder、DetRecoveryHead、Detail Bypass、Reconstruction heads 权重。
3. **Stage 3 Checkpoint**：已训练好的 `outputs/mdvsc_stage3/best.pt`，包含 MDVSC v2 权重。
4. **数据集**：与之前各阶段相同（ImageNet VID）。

## 六、训练命令

### 基本用法

```bash
python scripts/train_mdvsc_stage4.py \
    --config configs/mdvsc_stage4.yaml \
    --stage2-1-ckpt outputs/mdvsc_stage2_1/best.pt \
    --stage3-ckpt outputs/mdvsc_stage3/best.pt
```

### 完整参数覆盖

```bash
python scripts/train_mdvsc_stage4.py \
    --config configs/mdvsc_stage4.yaml \
    --data /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train \
    --stage2-1-ckpt outputs/mdvsc_stage2_1/best.pt \
    --stage3-ckpt outputs/mdvsc_stage3/best.pt \
    --output outputs/mdvsc_stage4_exp01
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径，默认 `configs/mdvsc_stage4.yaml` |
| `--data` | 覆盖 `data.train_source_path`（训练数据目录） |
| `--stage2-1-ckpt` | Stage 2.1 的 `.pt` 文件路径（提供 SharedEncoder + dual decoder 权重） |
| `--stage3-ckpt` | Stage 3 的 `.pt` 文件路径（提供 MDVSC v2 权重） |
| `--output` | 覆盖 `output.output_dir`（输出目录） |

## 七、配置文件说明

`configs/mdvsc_stage4.yaml` 的关键节点：

### mdvsc 节

```yaml
mdvsc:
  # SharedEncoder
  backbone_channels: [512, 1024, 2048]
  shared_channels: 256
  # MDVSC v2
  latent_dims: [64, 80, 96]
  common_keep_ratios: [0.6, 0.7, 0.8]
  individual_keep_ratios: [0.15, 0.2, 0.25]
  block_sizes: [4, 2, 1]
  spatial_strides: [2, 2, 1]
  apply_cross_level_fusion: true
  apply_masks: true
  channel_mode: identity           # 可切换为 awgn
  snr_db: 20.0
  # Reconstruction
  reconstruction_hidden_channels: 160
  reconstruction_detail_channels: 64
  # Detail bypass
  detail_latent_channels: 32
  detail_spatial_size: 20
```

### optimization 节

```yaml
optimization:
  batch_size: 1
  optimizer: adamw
  scheduler: onecycle
  grad_clip_norm: 1.0

  phase1:
    epochs: 8
    lr: 1.0e-4
    freeze_shared_encoder: true    # Phase 1 冻结 SharedEncoder
    freeze_mdvsc_v2: false
    freeze_det_recovery: false
    freeze_detail_bypass: false
    freeze_reconstruction: false

  phase2:
    epochs: 8
    lr: 5.0e-5
    freeze_shared_encoder: false   # Phase 2 解冻所有
    ...

  phase3:
    epochs: 4
    lr: 2.0e-5
    freeze_shared_encoder: false   # Phase 3 精细微调
    ...
```

每个 Phase 独立创建 optimizer 和 OneCycleLR scheduler,确保不同 Phase 间学习率不干扰。

### loss 节

```yaml
loss:
  # 特征压缩损失（Stage 3 目标）
  feature_loss_type: smooth_l1
  feature_loss_weight: 1.0
  level_loss_weights: [1.0, 1.0, 1.0]
  # 检测恢复损失（Stage 2.1 目标）
  det_recovery_weight: 1.0
  level_recovery_weights: [1.0, 1.0, 1.0]
  # 重建损失（Stage 2.1 目标）
  recon_l1_weight: 1.0
  recon_mse_weight: 0.25
  recon_ssim_weight: 0.25
  recon_edge_weight: 0.2
```

### initialization 节

```yaml
initialization:
  stage2_1_checkpoint:    # Stage 2.1 的 best.pt
  stage3_checkpoint:      # Stage 3 的 best.pt
  full_checkpoint:        # 或者直接提供一个 Stage 4 的完整 checkpoint（用于恢复训练）
  strict: false
```

**权重加载逻辑**：

- 如果指定 `full_checkpoint`：直接加载全部权重（用于断点续训）。
- 如果指定 `stage2_1_checkpoint` + `stage3_checkpoint`：
  - 从 Stage 2.1 加载：`shared_encoder.*`、`det_recovery_head.*`、`detail_compressor.*`、`detail_decompressor.*`、`reconstruction_head.*`、`reconstruction_refinement_heads.*`
  - 从 Stage 3 加载：原 `level_modules.*`、`cross_level_fusion.*` → 映射到 `mdvsc_v2.level_modules.*`、`mdvsc_v2.cross_level_fusion.*`

## 八、输出

训练完成后，输出目录结构：

```
outputs/mdvsc_stage4/
  resolved_config.json          # 完整配置
  dataset_info.json             # 数据集统计
  metrics.jsonl                 # 逐 epoch 指标（含 phase 标记）
  final_summary.json            # 最终训练摘要
  best.pt                       # 最优 checkpoint
  latest.pt                     # 最新 checkpoint
  epoch_001.pt ... epoch_020.pt
  visualizations/
    train/
      epoch_001_reconstruction.png    # GT vs 重建帧 vs |diff|
      epoch_001_feature_maps.png      # shared teacher vs MDVSC restored vs |diff|
      epoch_001_det_recovery.png      # projected teacher vs det recovery vs |diff|
      epoch_001_entropy_masks.png     # 公共/个体熵热图 + 掩码
    val/
      ...
```

### metrics.jsonl 格式

每行一个 JSON 对象，新增 `phase` 和 `local_epoch` 字段：

```json
{
  "epoch": 1,
  "phase": "phase1",
  "local_epoch": 1,
  "train": {
    "total_loss": 2.345,
    "feature_loss_loss": 0.89,
    "det_recovery_loss": 0.12,
    "recon_l1_loss": 0.45,
    "recon_mse_loss": 0.034,
    "recon_ssim_loss": 0.18,
    "recon_edge_loss": 0.09,
    "recon_psnr": 23.5,
    "recon_ssim": 0.82,
    "common_active_ratio": 0.65,
    "individual_active_ratio": 0.18,
    "detail_tx_ratio": 0.012
  },
  "val": { ... },
  "lr": 1.0e-4
}
```

## 九、绘图

训练结束后生成汇总曲线图：

```bash
python scripts/plot_stage4_metrics.py --metrics outputs/mdvsc_stage4/metrics.jsonl
```

输出到 `outputs/mdvsc_stage4/plots/`：

- `combined_loss.png`：总损失 + 特征/检测/重建子损失概览，红色虚线标记 Phase 分界
- `reconstruction.png`：重建各分项损失 + PSNR/SSIM 曲线
- `mask_activity.png`：公共/个体分支激活比例
- `learning_rate.png`：学习率调度曲线，标注 Phase 名称
- `detail_tx_ratio.png`：Detail Bypass 传输开销比

## 十、可视化结果怎么看

### reconstruction.png

每行一帧（最多 `visualization_num_frames` 帧），三列：GT 原始帧、重建帧、|差异|。

- 差异图逐 epoch 变暗 → 重建质量在改善。
- 如果高频纹理区域差异大但平坦区域好 → 考虑增大 `detail_latent_channels` 或 `detail_spatial_size`。

### feature_maps.png

每行一个 level，三列：shared teacher / MDVSC v2 restored / |diff|。

- 与 Stage 3 的可视化格式一致。观察端到端微调后压缩保真度是否优于单独的 Stage 3。

### det_recovery.png

每行一个 level，三列：projected teacher / det recovery / |diff|。

- 新增可视化：检验经过 MDVSC v2 压缩 → 重建后，DetRecoveryHead 能否有效恢复检测特征。
- 如果 diff 显著大于纯 Stage 2.1 → `det_recovery_weight` 可能需要提高。

### entropy_masks.png

与 Stage 3 相同格式。观察联合训练是否改变了掩码分布。

## 十一、常见问题

### Q: 必须同时提供 Stage 2.1 和 Stage 3 的 checkpoint 吗？

强烈建议。没有任何 checkpoint 从随机初始化开始联合训练会非常困难。如果只有一个，也可以单独提供：
- 只有 Stage 2.1 → MDVSC v2 随机初始化，训练更慢但可行
- 只有 Stage 3 → SharedEncoder 和 dual decoder 随机初始化，效果大幅下降

### Q: 训练总 epoch 数怎么算？

默认 Phase1(8) + Phase2(8) + Phase3(4) = **20 个全局 epoch**。可以在 YAML 中调整每个 Phase 的 epochs。

### Q: Phase 之间学习率怎么切换？

每个 Phase 独立创建新的 optimizer 和 OneCycleLR scheduler。Phase 结束后 optimizer state 会丢弃，新 Phase 从指定 lr 重新开始 warmup + cosine 退火。

### Q: 某个 Phase 可以跳过吗？

可以。将该 Phase 的 `epochs` 设为 0 即可跳过。

### Q: 如何使用 AWGN 信道？

```yaml
mdvsc:
  channel_mode: awgn
  snr_db: 20.0
```

建议先在 `identity` 模式下训收敛，再切换到 `awgn` 做信道适配训练。

### Q: total_loss 不下降？

检查清单：
1. 确保两个 checkpoint 都正确加载（查看终端输出的 `Loaded N params from ...` 日志）。
2. Phase 1 的 lr 是否合适（默认 1e-4，可以先试 5e-5 更保守）。
3. `feature_loss_weight` 与 `recon_*_weight` 的比例是否合理。如果 feature_loss 大幅主导，降低 `feature_loss_weight` 到 0.5。
4. MDVSC v2 的 `apply_masks` 是否为 true——如果 keep_ratio 过低，信息丢失太多导致重建困难。

### Q: 如何做消融实验？

- 关掉 Detail Bypass：冻结 `detail_compressor` 和 `detail_decompressor`，或设置 `detail_latent_channels=0`（需代码适配）。
- 关掉 MDVSC 掩码：`apply_masks: false`，测端到端上界。
- 只训特征压缩：设 `recon_*_weight=0`、`det_recovery_weight=0`，等价于带端到端梯度的 Stage 3。

## 十二、断点续训

如果训练中断，使用 `full_checkpoint` 恢复：

```bash
python scripts/train_mdvsc_stage4.py \
    --config configs/mdvsc_stage4.yaml
```

在 YAML 中设置：

```yaml
initialization:
  full_checkpoint: outputs/mdvsc_stage4/latest.pt
```

注意：当前实现从 Phase 1 重新开始。如果需要精确恢复到某个 Phase 的某个 epoch，需要手动调整 YAML 中对应 Phase 的 epochs 或手动跳过已完成的 Phase（设 epochs=0）。
