# 第三阶段训练手册（Stage 3 Training Manual）

## 一、概述

第三阶段（Stage 3）训练的目标是验证 **MDVSC v2** 模型在语义特征压缩-传输-恢复链路中的特征保真度。

整体数据流如下：

```
原始视频帧
  → RT-DETR backbone（冻结）→ 原始 backbone 特征 [512/1024/2048]
  → SharedEncoder（冻结，来自 stage-2 checkpoint）→ shared 256 特征（作为 teacher target）
  → MDVSC v2（可训练）→ 恢复的 256 特征
  → loss = smooth_l1(恢复特征, shared 256 特征)
```

**关键约束**：

- RT-DETR 完全冻结，不参与梯度计算。
- SharedEncoder 完全冻结，其权重从你指定的 stage-2 `.pt` checkpoint 中加载。
- 只有 MDVSC v2 模型的参数参与训练。
- 损失函数只有特征图还原精度（per-level smooth L1 或 MSE），无帧重建损失、无检测一致性损失。

## 二、MDVSC v2 模型架构改进

相比 stage-1 使用的 MDVSC v1，v2 模型做了三项结构性改进：

### 1. 渐进压缩 + Skip Connection

v1 的 `PerLevelMDVSC` 直接用单层 1×1 Conv 把 256 维特征压缩到 32/48/64 维 latent，再用单层解码器恢复。压缩比过大、容量不足。

v2 改为两阶段 U-Net 式编解码器：

```
编码: 256 → mid → latent_dim     (每阶段 2×ResBlock + SE attention)
          ↓skip
解码: latent_dim + skip → mid → 256  (concat skip 后 1×1 融合 + 2×ResBlock)
```

skip connection 让细粒度空间信息可以绕过最窄瓶颈传递。

### 2. 可学习时间加权平均

v1 用硬 `mean(dim=1)` 做公共/个体分解，假设公共信息就是 GOP 帧的时间均值。

v2 用 `TemporalAttentionDecomposer`：一个轻量 1×1 conv 对每帧打分，softmax 后加权求和得到 common latent，再做减法得 individual。网络可以学习到更合理的公共-个体划分。

### 3. 熵模型网络驱动的 Top-k 掩码

v1 直接对公共/个体特征的绝对值取 top-k 做掩码——特征值大并不一定意味着对重建更重要。

v2 引入 `EntropyMaskGate`：用一个小型 CNN (`entropy_net`) 预测每个元素的熵分数，然后对**负熵**（低熵 = 更可预测 = 对重建更重要）取 top-k 生成掩码。训练时通过 STE soft mask 保证梯度回传到熵网络。

## 三、文件结构

```
src/semantic_rtdetr/semantic_comm/mdvsc_v2.py      # MDVSC v2 模型定义
src/semantic_rtdetr/training/stage3_config.py       # Stage-3 配置 dataclass
src/semantic_rtdetr/training/stage3_trainer.py      # Stage-3 训练器
scripts/train_mdvsc_stage3.py                       # 训练入口脚本
scripts/plot_stage3_metrics.py                      # 训练曲线绘图脚本
configs/mdvsc_stage3.yaml                           # 默认配置文件
```

## 四、前置条件

1. **RT-DETR 权重**：`pretrained/rtdetr_r50vd/` 目录下已有模型文件。
2. **Stage-2 Checkpoint**：你需要提供一个已训练好的 stage-2 checkpoint `.pt` 文件，其中包含 `shared_encoder.*` 的权重。
3. **数据集**：ImageNet VID 数据集，目录结构形如：

```
/root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train/
  ILSVRC2015_VID_train_0000/
    ILSVRC2015_train_00000000/
      000000.JPEG
      000001.JPEG
      ...
  ILSVRC2015_VID_train_0001/
    ...
```

## 五、训练命令

### 基本用法

```bash
python scripts/train_mdvsc_stage3.py \
    --config configs/mdvsc_stage3.yaml \
    --stage2-ckpt outputs/mdvsc_stage2/best.pt
```

### 完整参数覆盖

```bash
python scripts/train_mdvsc_stage3.py \
    --config configs/mdvsc_stage3.yaml \
    --data /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train \
    --stage2-ckpt outputs/mdvsc_stage2/best.pt \
    --output outputs/mdvsc_stage3_exp01
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径，默认 `configs/mdvsc_stage3.yaml` |
| `--data` | 覆盖 `data.train_source_path`（训练数据目录） |
| `--stage2-ckpt` | **必须**，覆盖 `initialization.stage2_checkpoint`（stage-2 的 `.pt` 文件路径） |
| `--output` | 覆盖 `output.output_dir`（输出目录） |

## 六、配置文件说明

`configs/mdvsc_stage3.yaml` 的关键节点：

### mdvsc 节

```yaml
mdvsc:
  backbone_channels: [512, 1024, 2048]    # backbone 各 level 原始通道数
  feature_channels: [256, 256, 256]        # SharedEncoder 输出通道数 = MDVSC v2 输入通道数
  latent_dims: [64, 80, 96]               # 各 level 的 latent 压缩维度
  common_keep_ratios: [0.6, 0.7, 0.8]     # 公共分支熵掩码保留比例
  individual_keep_ratios: [0.15, 0.2, 0.25] # 个体分支熵掩码保留比例
  block_sizes: [4, 2, 1]                  # 熵掩码的空间块大小
  apply_cross_level_fusion: true           # 是否启用跨 level 特征融合
  apply_masks: true                        # 是否启用掩码
  channel_mode: identity                   # 信道模式（identity / awgn）
  snr_db: 20.0                             # AWGN 信噪比
```

### initialization 节

```yaml
initialization:
  stage2_checkpoint:    # stage-2 的 .pt 文件路径（包含 SharedEncoder 权重）
  checkpoint:           # MDVSC v2 自身的 checkpoint（用于继续训练）
  strict: false
```

### loss 节

```yaml
loss:
  feature_loss_type: smooth_l1    # 可选 smooth_l1 或 mse
  level_loss_weights: [1.0, 1.0, 1.0]  # 各 level 的损失权重
```

## 七、冻结策略

| 组件 | 状态 | 来源 |
|------|------|------|
| RT-DETR backbone + encoder + decoder | 冻结 | HuggingFace 预训练权重 |
| SharedEncoder | 冻结 | stage-2 checkpoint |
| MDVSC v2 (level_modules + cross_level_fusion) | **可训练** | 随机初始化或 MDVSC checkpoint |

## 八、输出

训练完成后，输出目录结构如下：

```
outputs/mdvsc_stage3/
  resolved_config.json      # 实际使用的完整配置
  dataset_info.json          # 数据集统计
  metrics.jsonl              # 逐 epoch 训练/验证指标
  final_summary.json         # 最终训练摘要
  best.pt                    # 最优 checkpoint
  latest.pt                  # 最新 checkpoint
  epoch_001.pt ... epoch_020.pt  # 各 epoch checkpoint
  visualizations/
    train/
      epoch_001_feature_maps.png     # teacher vs restored 特征对比
      epoch_001_entropy_masks.png    # 熵热图 + 掩码可视化
      epoch_001_cosine_sim.png       # 逐 level 余弦相似度图
    val/
      ...
```

## 九、绘图

训练结束后，用以下命令生成汇总曲线图：

```bash
python scripts/plot_stage3_metrics.py --metrics outputs/mdvsc_stage3/metrics.jsonl
```

输出三张图到 `outputs/mdvsc_stage3/plots/`：

- `feature_loss.png`：总特征损失 + 各 level 损失曲线
- `mask_activity.png`：公共/个体分支激活比例
- `learning_rate.png`：学习率调度

## 十、可视化结果怎么看

### feature_maps.png

每行一个 level，三列：teacher（SharedEncoder 输出）、restored（MDVSC v2 输出）、|diff|（绝对误差）。

- 如果 diff 逐 epoch 变浅变暗 → 特征恢复在改善。
- 如果某个 level 的 diff 始终很大 → 该 level 的 latent_dim 可能需要增大，或该 level 的 loss_weight 需要提高。

### entropy_masks.png

每行一个 level，四列：common 熵热图、common 掩码、individual 熵热图、individual 掩码。

- 熵热图亮区 = 高熵（不重要，容易被淘汰）。
- 掩码白区 = 保留、黑区 = 丢弃。
- 理想情况：掩码应该保留目标区域对应的特征，丢弃背景区域。

### cosine_sim.png

每个 level 一张余弦相似度图。

- 绿色区域 = 高相似度（恢复好）。
- 红色区域 = 低相似度（恢复差）。
- 随训练进行，绿色面积应该越来越大。

## 十一、常见问题

### Q: 没有 stage-2 checkpoint 怎么办？

必须先完成 stage-2 训练。SharedEncoder 是连接 backbone 和 MDVSC 的关键桥梁，随机初始化的 SharedEncoder 无法提供有意义的特征目标。

### Q: feature_loss 下降很慢？

优先检查：
1. latent_dims 是否过小（默认 [64, 80, 96]，比 v1 的 [32, 48, 64] 已显著增大）。
2. apply_masks 是否为 true——如果 keep_ratio 过低，信息丢失过多。
3. 学习率是否合适——默认 2e-4，可尝试 5e-4。

### Q: 如何关闭掩码做上界试验？

```yaml
mdvsc:
  apply_masks: false
```

这时 MDVSC v2 会以全量 latent 做编解码，不进行任何稀疏筛选，给出当前架构的特征恢复上界。
