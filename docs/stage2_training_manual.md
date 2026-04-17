# 第二阶段训练中文操作说明书

本文面向当前仓库里的第二阶段训练入口，目标是让你在服务器上把 Stage 2 训练真正跑起来，并且知道每一步为什么这样做。

## 一、第二阶段的目标是什么

第二阶段的核心目标是验证 SharedEncoder + 双解码器架构，训练时不经过 MDVSC 传输模块，直接在特征空间做无损通路。具体要做到三件事：

1. 从 RT-DETR backbone 原始特征（512/1024/2048 通道）出发，通过 SharedEncoder 压缩到 256 通道的共享表示；
2. 从共享表示恢复出与 RT-DETR encoder_input_proj 输出一致的 projected 256 特征（DetRecoveryHead）；
3. 从共享表示重建出高质量视频帧（LightReconstructionHead）。

**关键设计决策**：图像重建和 projected 256 恢复必须始终一起训练、一起反向传播损失。不做任何分阶段预训练。原因是 SharedEncoder 是两个任务共享的特征提取器，两路损失同时回传可以确保共享表示同时对检测恢复和图像重建友好，而不是偏向其中一方。

## 二、架构总览

```
RT-DETR backbone (frozen)
  ├─ stage2: [B, 512, 80, 80]
  ├─ stage3: [B, 1024, 40, 40]
  └─ stage4: [B, 2048, 20, 20]
        │
        ▼
SharedEncoder (per-level independent, no FPN)
  每个 level: Conv1×1(backbone_ch → 256) + GELU + ResBlock(256)
  输出: 3 × [B, T, 256, H, W]
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
DetRecoveryHead                    TaskAdaptationBlock × 3
  每个 level:                        + LightReconstructionHead
  Conv1×1(256→256, no bias) + BN      → 重建帧 [B, T, 3, H, W]
  → projected 256 恢复
  3 × [B, T, 256, H, W]
```

各模块说明：

- **SharedEncoder**：逐 level 独立的 Conv1×1 投影 + GELU + ResidualBlock，不使用 FPN，保证每个 level 都可以独立地被 DetRecoveryHead 恢复。C_shared 固定为 256。
- **DetRecoveryHead**：逐 level 的 Conv1×1(256→256, bias=False) + BatchNorm2d(256)，结构上对齐冻结的 teacher encoder_input_proj，目标是让恢复出的 projected 256 尽量逼近 teacher 输出。
- **TaskAdaptationBlock**：轻量 refinement 模块，将共享 256 特征适配到重建解码器需要的分布。
- **LightReconstructionHead**：复用 Stage 1 的重建头，包含 additive FPN + DW-separable 上采样 + 可学习 hf_scale，输出 base RGB + 高频残差，两者相加得到最终重建帧。

RT-DETR teacher 全程冻结，不参与梯度计算。

## 三、训练前需要准备什么

与第一阶段相同，训练前必须满足两个前提：

1. 你本地或服务器上有可读的视频帧数据（帧目录或视频文件）。
2. 运行环境能够拿到 RT-DETR teacher 权重。

权重默认位于：

```text
pretrained/rtdetr_r50vd/
```

如果该目录不存在，先运行：

```bash
python scripts/download_rtdetr_weights.py
```

## 四、服务器环境准备

推荐环境与第一阶段完全一致：

- Linux 服务器；
- Python 3.10 及以上；
- GPU 优先（显存建议 8GB+）；
- 首次准备权重时能访问 Hugging Face，或者已从别的机器拷贝好了 `pretrained/rtdetr_r50vd/` 目录。

安装流程：

```bash
git clone <你的仓库地址>
cd my_first_try
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_rtdetr_weights.py
```

## 五、数据目录怎么准备

第二阶段与第一阶段共用相同的数据加载器，支持两类输入：

1. 帧目录：一个目录里按顺序放图片帧。
2. 单个视频文件。

递归扫描根目录也支持。数据索引缓存默认在 `.cache/stage2_data`。

如果你使用 ImageNet VID：

```yaml
data:
  dataset_name: imagenet_vid
  train_source_path: /absolute/path/to/ILSVRC/Data/VID/train
  subset_seed: 42
  source_fraction: 0.2
  sample_fraction: 0.2
```

一个通用目录示例：

```text
data/
  clip_001/
    000001.jpg
    000002.jpg
    ...
  clip_002/
    000001.jpg
    000002.jpg
    ...
```

## 六、配置文件

第二阶段使用独立的配置文件：

- `configs/mdvsc_stage2.yaml`

### 数据相关

与第一阶段完全相同，参见第一阶段说明书的数据配置部分。

### 模型相关

```yaml
mdvsc:
  backbone_channels: [512, 1024, 2048]    # RT-DETR r50vd backbone 各 level 通道数
  shared_channels: 256                     # SharedEncoder 输出通道，固定 256
  reconstruction_hidden_channels: 160      # 语义主干解码宽度
  reconstruction_detail_channels: 64       # 细节分支宽度
  reconstruction_head_type: light          # light 或 standard
  reconstruction_use_checkpoint: false     # 是否使用梯度检查点节省显存
```

- `backbone_channels` 必须与 RT-DETR r50vd backbone 对齐：`[512, 1024, 2048]`。
- `shared_channels` 固定为 256，因为 RT-DETR 的 encoder_input_proj 输出就是 256。
- `reconstruction_hidden_channels` 和 `reconstruction_detail_channels` 直接影响重建质量，如果显存允许可以尝试 256 / 128。

### 训练相关

```yaml
optimization:
  batch_size: 1
  num_workers: 4
  use_amp: true
  amp_dtype: float16
  optimizer: adamw
  epochs: 20                  # 总训练轮数
  lr: 2.0e-4                  # 学习率
  weight_decay: 1.0e-4
  adam_beta1: 0.9
  adam_beta2: 0.95
  scheduler: onecycle          # 支持 constant、cosine、onecycle
  warmup_epochs: 1
  warmup_start_factor: 0.2
  min_lr_ratio: 0.1
  onecycle_pct_start: 0.15
  onecycle_div_factor: 25.0
  onecycle_final_div_factor: 1000.0
  grad_clip_norm: 1.0
  log_every: 10
  save_every_epochs: 1
  max_steps_per_epoch:         # 可选，限制每 epoch 最大 step 数
  seed: 42
```

注意：第二阶段是单阶段训练，只有 `epochs` 和 `lr` 两个核心控制项，没有分阶段的 pretrain/joint 区分。所有模块从第 1 轮开始就一起训练。

### 损失相关

```yaml
loss:
  det_recovery_weight: 1.0         # 检测恢复总权重
  recon_l1_weight: 1.0             # 像素 L1 重建权重
  recon_mse_weight: 0.25           # 像素 MSE 重建权重
  recon_ssim_weight: 0.25          # SSIM 重建权重
  recon_edge_weight: 0.2           # 高频边缘重建权重
  ssim_downsample_factor: 2        # SSIM 计算前下采样倍数
  level_recovery_weights: [1.0, 1.0, 1.0]  # 各 level 检测恢复权重
```

总损失公式：

$$
L_{\text{total}} = w_{\text{L1}} \cdot L_{\text{L1}} + w_{\text{MSE}} \cdot L_{\text{MSE}} + w_{\text{SSIM}} \cdot L_{\text{SSIM}} + w_{\text{edge}} \cdot L_{\text{edge}} + w_{\text{det}} \cdot \sum_{l=0}^{2} w_l \cdot \text{MSE}(\hat{P}_l, P_l)
$$

其中 $\hat{P}_l$ 是 DetRecoveryHead 恢复的 projected 特征，$P_l$ 是冻结 teacher encoder_input_proj 输出的 projected 特征。

### 输出相关

```yaml
output:
  output_dir: outputs/mdvsc_stage2
  save_visualizations: true
  visualization_every_epochs: 1
  visualization_num_frames: 4
```

### 初始化相关

```yaml
initialization:
  full_checkpoint:              # 可选，完整的 Stage2MDVSC checkpoint
  stage1_recon_checkpoint:      # 可选，第一阶段重建头 checkpoint
  strict: false
```

- `stage1_recon_checkpoint`：如果你有第一阶段训练好的重建头权重，可以在这里加载，加速重建支路的收敛。只会加载 `reconstruction_head.*` 和 `reconstruction_refinement_heads.*` 前缀的参数。
- `full_checkpoint`：加载完整的 Stage2MDVSC 权重，用于断点续训或 fine-tune。

## 七、怎么启动训练

```bash
python scripts/train_mdvsc_stage2.py \
  --config configs/mdvsc_stage2.yaml \
  --data /absolute/path/to/your/frame/data \
  --output outputs/mdvsc_stage2
```

`--data` 和 `--output` 是可选的命令行覆盖参数，优先级高于 YAML 配置。

如果你使用 ImageNet VID 子集：

```bash
python scripts/train_mdvsc_stage2.py \
  --config configs/mdvsc_stage2.yaml \
  --data /absolute/path/to/ILSVRC/Data/VID/train
```

如果你怀疑 AMP 带来了数值不稳定，可以在 YAML 里关闭：

```yaml
optimization:
  use_amp: false
```

## 八、训练过程中会产出什么

输出目录结构：

```text
outputs/mdvsc_stage2/
  resolved_config.json          # 本次训练实际使用的配置
  dataset_info.json             # 数据集摘要
  metrics.jsonl                 # 每个 epoch 的训练和验证指标
  latest.pt                     # 最近一次 checkpoint
  best.pt                       # 当前最优 checkpoint（按验证集 total_loss）
  epoch_001.pt ... epoch_020.pt # 按周期保存的 checkpoint
  final_summary.json            # 训练结束摘要
  visualizations/
    train/
      epoch_001_reconstruction.png
      epoch_001_feature_recovery.png
    val/
      epoch_001_reconstruction.png
      epoch_001_feature_recovery.png
```

Checkpoint 内容包括：

- `model_state`：完整的 Stage2MDVSC 权重。
- `optimizer_state`：优化器状态。
- `scheduler_state`：调度器状态。
- `grad_scaler_state`：如果启用了 AMP float16。
- `config`：当次训练的完整配置。
- `summary`：当前 epoch 的指标摘要。

## 九、可视化图分别表示什么

### 1. reconstruction 图

`epoch_XXX_reconstruction.png` 包含四行：

- 第一行：原始帧；
- 第二行：最终重建帧；
- 第三行：base 图像（LightReconstructionHead 的低频输出）；
- 第四行：高频残差热图。

这张图直接反映 SharedEncoder → ReconRefinement → LightReconstructionHead 整条通路的重建质量。

### 2. feature_recovery 图

`epoch_XXX_feature_recovery.png` 包含每个 level 三列：

- 第一列：teacher projected 特征均值热图；
- 第二列：DetRecoveryHead 恢复的 projected 特征均值热图；
- 第三列：两者绝对误差热图。

这张图直接反映 SharedEncoder → DetRecoveryHead 对 teacher encoder_input_proj 输出的逼近程度。误差热图越暗说明恢复越精确。

## 十、训练时建议重点盯哪些指标

建议优先看：

1. **`det_recovery_loss`**：检测恢复 MSE，越低说明 SharedEncoder → DetRecoveryHead 恢复 projected 特征越精确。
2. **`recon_psnr`**：重建峰值信噪比，越高越好。
3. **`recon_ssim`**：重建结构相似度（已转换为 0-1，1 最好）。
4. **`total_loss`**：总损失，用于判断整体收敛趋势。
5. **`recon_mse_loss`**：像素级 MSE，用于计算 PSNR。

理想训练现象：

- `det_recovery_loss` 和 `total_loss` 持续下降；
- `recon_psnr` 持续上升，20 epoch 后建议达到 25+ dB；
- `recon_ssim` 持续上升，目标 0.8+；
- feature_recovery 可视化里的误差热图逐 epoch 变暗；
- reconstruction 可视化里重建帧逐 epoch 变清晰。

异常现象与诊断：

| 异常现象 | 可能原因 | 建议操作 |
|---------|---------|---------|
| 总损失震荡不降 | 学习率过高 | 降低 `lr` 到 1e-4 |
| det_recovery 快速下降但重建停滞 | 重建头容量不够 | 增大 `reconstruction_hidden_channels` |
| 重建快速提升但 det_recovery 停滞 | `det_recovery_weight` 过低 | 增大到 2.0 或 5.0 |
| 训练中出现 NaN | AMP 数值溢出 | 关闭 AMP 或切 bfloat16 |
| 重建图全灰或全黑 | 梯度爆炸或模型初始化问题 | 检查 `grad_clip_norm`，确认数据归一化正确 |

## 十一、与第一阶段的关键区别

| 维度 | 第一阶段 | 第二阶段 |
|-----|---------|---------|
| 输入特征 | RT-DETR encoder_input_proj 输出的 projected 256 | RT-DETR backbone 原始输出 512/1024/2048 |
| 特征压缩 | MDVSC（含 mask、量化、信道传输） | SharedEncoder（无损投影，无传输） |
| 检测恢复 | TaskAdaptationBlock | DetRecoveryHead (Conv1×1 + BN) |
| 训练阶段 | 三段式（recon pretrain → MDVSC bootstrap → joint） | 单阶段联合训练 |
| 目的 | 验证 MDVSC 传输 + 重建的基线能力 | 验证切点上移 + SharedEncoder + 双解码器的信息保留能力 |

第二阶段训练稳定后，下一步是在 SharedEncoder 之后接入 MDVSC 传输模块，形成完整的 backbone → SharedEncoder → MDVSC → 双解码器 pipeline。

## 十二、显存不够时怎么调

优先按这个顺序降：

1. 把 `optimization.batch_size` 调成 1。
2. 开启 `mdvsc.reconstruction_use_checkpoint: true`（梯度检查点，用时间换显存）。
3. 把 `data.frame_height` 和 `data.frame_width` 调小（例如 480）。
4. 把 `data.gop_size` 从 4 调成 2。
5. 把 `data.source_fraction` 和 `data.sample_fraction` 再调小。
6. 把 `optimization.num_workers` 调低。

## 十三、常见问题

### 1. 第一次运行很慢

通常是因为第一次加载 RT-DETR 权重和扫描数据索引。建议先单独执行一次：

```bash
python scripts/download_rtdetr_weights.py
```

数据索引第一次扫描后会缓存，后续启动会快很多。

### 2. 服务器无法下载 RT-DETR 权重

在可联网机器上先下载，然后把整个 `pretrained/rtdetr_r50vd/` 目录拷贝到服务器。确保配置里的 `detector.local_path` 指向该目录。

### 3. 加载第一阶段重建头 checkpoint 后效果变差

检查第一阶段 checkpoint 的 `reconstruction_hidden_channels` 和 `reconstruction_detail_channels` 是否与第二阶段配置一致。如果不一致，设 `initialization.strict: false` 可以跳过不匹配的参数，但这意味着部分权重没有加载。

### 4. det_recovery_loss 始终很高

- 检查 `loss.level_recovery_weights` 是否全为 0（那就没有监督信号了）。
- 检查 `loss.det_recovery_weight` 是否过小。
- 确认 `RTDetrBaseline.extract_backbone_and_projected_features` 方法正常工作（backbone 和 projected 特征应该有合理的数值范围）。

### 5. feature_recovery 热图里 teacher 和 student 完全不同

训练刚开始时这是正常的。如果 10 个 epoch 后仍然完全不同，检查 backbone_channels 配置是否正确，SharedEncoder 和 DetRecoveryHead 是否在正常接收梯度。

## 十四、当前阶段之后该做什么

第二阶段跑稳后，建议按这个顺序往下走：

1. 确认 `det_recovery_loss` 收敛到较低值，说明 SharedEncoder 能保留足够的检测信息。
2. 确认重建 PSNR 达到合理水平（25+ dB），说明 SharedEncoder 也保留了足够的视觉信息。
3. 在 SharedEncoder 之后插入 MDVSC 传输模块（mask + 量化 + 信道），形成 Stage 3。
4. Stage 3 的传输模块可以从第一阶段已有的 MDVSC 权重初始化。
5. 最终补带标注的检测监督和完整评测闭环。
