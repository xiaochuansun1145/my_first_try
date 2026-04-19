# 第二阶段 2.1 训练中文操作说明书

本文面向当前仓库里的 Stage 2.1 训练入口，目标是让你在服务器上把 Stage 2.1 训练真正跑起来。

## 一、Stage 2.1 的目标是什么

Stage 2.1 在 Stage 2（SharedEncoder + 双解码器）的基础上，增加了 **Detail Bypass** 旁路——把 RT-DETR backbone 最浅层的 stage1 特征（256 通道、stride 4、160×160）额外压缩成一个很小的 packet 一起传输，用来补偿深层特征丢失的高频细节信息，从而提升图像重建质量。

具体要做到四件事：

1. SharedEncoder 把 backbone 原始特征（512/1024/2048）投影为共享 256 表示（同 Stage 2）；
2. DetRecoveryHead 从共享表示恢复 projected 256 检测特征（同 Stage 2）；
3. DetailCompressor 把 stage1（256ch, 160×160）压缩为 `[32, 20, 20]` 的 tiny packet，传输开销仅约 1%；
4. DetailDecompressor 把 packet 展开并融合到重建路径的 stride-8 层，LightReconstructionHead 利用细节特征生成更清晰的重建帧。

**关键决策**：与 Stage 2 一样，图像重建和 projected 256 恢复始终一起训练、一起反向传播。Detail Bypass 只帮助重建分支，不影响检测恢复。

## 二、架构总览

```
RT-DETR backbone (frozen)
  ├─ stage1: [B, 256, 160, 160]  ← Detail Bypass 输入
  ├─ stage2: [B, 512, 80, 80]
  ├─ stage3: [B, 1024, 40, 40]
  └─ stage4: [B, 2048, 20, 20]
        │                              │
        ▼                              ▼
SharedEncoder (per-level)          DetailCompressor
  → 3×[B,T,256,H,W]                 Conv1×1(256→32) + GELU
        │                              + AdaptiveAvgPool2d(20)
        ├───────────┐                  + ResBlock(32)
        ▼           ▼                  → [B,T,32,20,20]  ~1% 传输开销
DetRecoveryHead  TaskAdapt×3               │
  → projected256   + Recon ←───── DetailDecompressor
                     Head            Conv1×1(32→160) + GELU
                   → [B,T,3,H,W]    + ResBlock + bilinear upsample
                                     → additive fusion at stride-8
```

各新增模块说明：

- **DetailCompressor**：Conv1×1(256→32) + GELU + AdaptiveAvgPool2d(20) + ResBlock(32)。把 stage1 的 160×160 压缩到 20×20，通道从 256 降到 32，总量从 ~6.5M 降到 12.8K 标量。
- **DetailDecompressor**：Conv1×1(32→160) + GELU + ResBlock(160) + 双线性上采样到 stride-8 分辨率，然后 additive fusion 到重建路径。
- **DetailAwareLightReconstructionHead**：继承 LightReconstructionHead，在 additive FPN + refine 之后、progressive upsampling 之前，把 detail features 加到 stride-8 融合特征上。

## 三、训练前需要准备什么

与 Stage 2 完全相同：

1. 本地或服务器上有可读的视频帧数据。
2. RT-DETR teacher 权重已就绪。

权重默认位于：

```text
pretrained/rtdetr_r50vd/
```

如果该目录不存在，先运行：

```bash
python scripts/download_rtdetr_weights.py
```

## 四、服务器环境准备

推荐环境与 Stage 2 完全一致：

- Linux 服务器；
- Python 3.10 及以上；
- GPU 优先（显存建议 8GB+）；
- 首次准备权重时能访问 Hugging Face，或者已拷贝好 `pretrained/rtdetr_r50vd/` 目录。

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

## 五、数据目录

Stage 2.1 与 Stage 2 共用相同的数据加载器。

当前服务器上 ImageNet VID 的数据目录为：

```
/root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train
```

如果你使用其他数据，将路径替换即可。数据格式要求与 Stage 1/2 一致：帧目录或视频文件，支持递归扫描。

## 六、配置文件

Stage 2.1 使用独立的配置文件：

- `configs/mdvsc_stage2_1.yaml`

### 模型相关（重点：新增 Detail Bypass 参数）

```yaml
mdvsc:
  backbone_channels: [512, 1024, 2048]    # RT-DETR r50vd backbone 各 level 通道数
  shared_channels: 256                     # SharedEncoder 输出通道，固定 256
  reconstruction_hidden_channels: 160      # 重建解码宽度
  reconstruction_detail_channels: 64       # 细节分支宽度
  reconstruction_head_type: light          # 当前 Stage 2.1 仅支持 light
  reconstruction_use_checkpoint: false     # 是否使用梯度检查点
  # Detail Bypass 新增参数
  stage1_channels: 256                     # backbone stage1 通道数，固定 256
  detail_latent_channels: 32               # 压缩后 detail packet 通道数
  detail_spatial_size: 20                  # 压缩后 detail packet 空间分辨率
```

- `detail_latent_channels`：控制 detail packet 的通道数。32 是推荐值，增大到 64 可以提升细节但传输开销翻倍（仍然 < 3%）。
- `detail_spatial_size`：控制 detail packet 的空间大小。20 对应 20×20 = 400 个空间位置。减小到 10 可以进一步降低传输开销，但细节恢复能力下降。
- 传输开销计算：`detail_latent_channels × detail_spatial_size² / main_features_size`。默认配置约 1%。

### 训练相关

```yaml
optimization:
  batch_size: 1
  num_workers: 4
  use_amp: true
  amp_dtype: float16
  optimizer: adamw
  epochs: 20
  lr: 2.0e-4
  weight_decay: 1.0e-4
  adam_beta1: 0.9
  adam_beta2: 0.95
  scheduler: onecycle
  warmup_epochs: 1
  warmup_start_factor: 0.2
  min_lr_ratio: 0.1
  onecycle_pct_start: 0.15
  onecycle_div_factor: 25.0
  onecycle_final_div_factor: 1000.0
  grad_clip_norm: 1.0
  log_every: 10
  save_every_epochs: 1
  max_steps_per_epoch:
  seed: 42
```

与 Stage 2 一样是单阶段联合训练，没有分阶段预训练。

### 损失相关

```yaml
loss:
  det_recovery_weight: 1.0
  recon_l1_weight: 1.0
  recon_mse_weight: 0.25
  recon_ssim_weight: 0.25
  recon_edge_weight: 0.2
  ssim_downsample_factor: 2
  level_recovery_weights: [1.0, 1.0, 1.0]
```

总损失公式与 Stage 2 一致：

$$
L_{\text{total}} = w_{\text{L1}} \cdot L_{\text{L1}} + w_{\text{MSE}} \cdot L_{\text{MSE}} + w_{\text{SSIM}} \cdot L_{\text{SSIM}} + w_{\text{edge}} \cdot L_{\text{edge}} + w_{\text{det}} \cdot \sum_{l=0}^{2} w_l \cdot \text{MSE}(\hat{P}_l, P_l)
$$

Detail Bypass 不引入额外损失项——它通过改善重建帧质量来间接降低重建损失。

### 初始化相关

```yaml
initialization:
  full_checkpoint:              # 完整的 Stage2_1MDVSC checkpoint（断点续训用）
  stage2_checkpoint:            # Stage 2 的 best.pt checkpoint（热启动）
  strict: false
```

- `stage2_checkpoint`：**强烈建议使用**。从 Stage 2 的 best.pt 热启动，可以复用已收敛的 SharedEncoder、DetRecoveryHead、ReconstructionRefinementHeads 和 LightReconstructionHead 权重。只有新增的 DetailCompressor 和 DetailDecompressor 从随机初始化开始。
- `full_checkpoint`：加载完整的 Stage2_1MDVSC 权重，用于断点续训。

## 七、怎么启动训练

### 最简启动（使用当前服务器数据目录）

```bash
python scripts/train_mdvsc_stage2_1.py \
  --config configs/mdvsc_stage2_1.yaml \
  --data /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train
```

### 从 Stage 2 热启动（推荐）

```bash
python scripts/train_mdvsc_stage2_1.py \
  --config configs/mdvsc_stage2_1.yaml \
  --data /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train \
  --stage2-checkpoint outputs/mdvsc_stage2/best.pt
```

### 自定义输出目录

```bash
python scripts/train_mdvsc_stage2_1.py \
  --config configs/mdvsc_stage2_1.yaml \
  --data /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train \
  --stage2-checkpoint outputs/mdvsc_stage2/best.pt \
  --output outputs/mdvsc_stage2_1_exp1
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--config` | YAML 配置文件路径 | `configs/mdvsc_stage2_1.yaml` |
| `--data` | 覆盖 `data.train_source_path` | YAML 中的值 |
| `--output` | 覆盖 `output.output_dir` | `outputs/mdvsc_stage2_1` |
| `--stage2-checkpoint` | Stage 2 的 best.pt 路径 | 无 |

### 如果想使用子集训练

在 YAML 中配置：

```yaml
data:
  dataset_name: imagenet_vid
  train_source_path: /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train
  subset_seed: 42
  source_fraction: 0.2
  sample_fraction: 0.2
```

然后启动时可以省略 `--data`：

```bash
python scripts/train_mdvsc_stage2_1.py \
  --config configs/mdvsc_stage2_1.yaml \
  --stage2-checkpoint outputs/mdvsc_stage2/best.pt
```

## 八、训练过程中会产出什么

输出目录结构：

```text
outputs/mdvsc_stage2_1/
  resolved_config.json
  dataset_info.json
  metrics.jsonl
  latest.pt
  best.pt
  epoch_001.pt ... epoch_020.pt
  final_summary.json
  visualizations/
    train/
      epoch_001_reconstruction.png
      epoch_001_feature_recovery.png
      epoch_001_detail_packet.png        ← Stage 2.1 新增
    val/
      epoch_001_reconstruction.png
      epoch_001_feature_recovery.png
      epoch_001_detail_packet.png        ← Stage 2.1 新增
```

与 Stage 2 对比，新增了 `detail_packet.png` 可视化——展示 DetailCompressor 输出的压缩 packet 的通道均值热图。

## 九、可视化图分别表示什么

### 1. reconstruction 图

与 Stage 2 一致，四行分别为：原始帧、最终重建帧、base 图像、高频残差热图。

对比 Stage 2 的同一张图，重点看：
- 重建帧的高频细节（边缘、纹理）是否更清晰；
- 高频残差是否包含了更多有意义的结构信息（而非噪声）。

### 2. feature_recovery 图

与 Stage 2 一致，每个 level 三列：teacher projected 特征热图、恢复热图、误差热图。

Detail Bypass 不影响检测恢复路径，因此这张图应该与 Stage 2 表现类似。

### 3. detail_packet 图（Stage 2.1 新增）

`epoch_XXX_detail_packet.png` 展示 DetailCompressor 输出的 `[32, 20, 20]` packet 的通道均值热图。

用来观察：
- packet 是否有结构化的空间模式（好现象，说明压缩保留了有意义的空间信息）；
- packet 是否全部为近零值（坏现象，说明 Compressor 没有学到有用信息）。

## 十、怎么把数值日志画成曲线图

训练后运行：

```bash
python scripts/plot_stage2_metrics.py \
  --metrics outputs/mdvsc_stage2_1/metrics.jsonl
```

默认在 `outputs/mdvsc_stage2_1/plots/` 下生成：

- `loss_overview.png`：6 子图——total loss、recon MSE、det recovery、L1、SSIM、edge loss
- `quality_metrics.png`：PSNR + SSIM 曲线
- `learning_rate.png`：学习率曲线
- `detail_tx_ratio.png`：detail packet 传输比例曲线（Stage 2.1 独有）

## 十一、训练时建议重点盯哪些指标

建议优先看：

1. **`recon_psnr`**：Stage 2.1 的核心目标就是提升重建质量，PSNR 应该比 Stage 2 明显更高。
2. **`recon_ssim`**：结构相似度，目标 0.85+（Stage 2 基线如果是 0.8 左右，Stage 2.1 应该有提升）。
3. **`det_recovery_loss`**：应与 Stage 2 持平，Detail Bypass 不应拖累检测恢复。
4. **`total_loss`**：总损失持续下降。
5. **`detail_tx_ratio`**：传输比例应稳定在配置值附近（默认 ~1%），训练过程中不会变化（因为它只取决于模型结构，不取决于权重）。

### 与 Stage 2 的预期对比

| 指标 | Stage 2 预期 | Stage 2.1 预期 |
|-----|------------|--------------|
| `det_recovery_loss` | ~0.029 | ~0.029（持平） |
| `recon_psnr` | 25+ dB | 27+ dB（提升 1-3 dB） |
| `recon_ssim` | 0.80+ | 0.85+（提升） |
| 传输开销 | 基线 100% | 基线 + ~1% |

### 异常现象与诊断

| 异常现象 | 可能原因 | 建议操作 |
|---------|---------|---------|
| PSNR 没有比 Stage 2 提升 | Detail Bypass 没学到有用信息 | 检查 detail_packet 可视化是否有结构 |
| det_recovery_loss 比 Stage 2 明显退化 | 热启动权重没有正确加载 | 检查 `--stage2-checkpoint` 路径是否正确 |
| 训练中出现 NaN | AMP 数值溢出 | 关闭 AMP 或切 bfloat16 |
| detail_packet 热图全为零 | Compressor 梯度为零 | 检查 Decompressor 输出是否参与重建损失 |
| 总损失震荡不降 | 学习率过高 | 降低 `lr` 到 1e-4 |

## 十二、与 Stage 2 的关键区别

| 维度 | Stage 2 | Stage 2.1 |
|-----|---------|-----------|
| 输入特征 | backbone stage2/3/4 | backbone stage1/2/3/4（多了 stage1） |
| Detail Bypass | 无 | DetailCompressor + DetailDecompressor |
| 传输开销 | 基线 | 基线 + ~1% |
| 热启动来源 | Stage 1 重建头 checkpoint | **Stage 2 的 best.pt** |
| 重建质量预期 | PSNR 25+ dB | PSNR 27+ dB |
| 检测恢复 | DetRecoveryHead | 完全相同 |
| 训练方式 | 单阶段联合训练 | 单阶段联合训练 |
| 配置文件 | `configs/mdvsc_stage2.yaml` | `configs/mdvsc_stage2_1.yaml` |
| 训练脚本 | `scripts/train_mdvsc_stage2.py` | `scripts/train_mdvsc_stage2_1.py` |

## 十三、显存不够时怎么调

与 Stage 2 一致的优先级顺序：

1. 把 `optimization.batch_size` 调成 1。
2. 开启 `mdvsc.reconstruction_use_checkpoint: true`。
3. 把 `data.frame_height` 和 `data.frame_width` 调小（例如 480）。
4. 把 `data.gop_size` 从 4 调成 2。
5. 缩小 `data.source_fraction` 和 `data.sample_fraction`。

Stage 2.1 的额外显存开销很小（DetailCompressor 和 DetailDecompressor 参数量约几万），主要增量来自 stage1 特征 tensor（256×160×160），约占额外 10-20MB。

## 十四、常见问题

### 1. 从 Stage 2 热启动时哪些模块被加载？

以下前缀的权重会被加载：
- `shared_encoder.*`
- `det_recovery_head.*`
- `reconstruction_head.*`
- `reconstruction_refinement_heads.*`

以下模块从随机初始化开始：
- `detail_compressor.*`
- `detail_decompressor.*`

### 2. 不从 Stage 2 热启动可以吗？

可以，但不推荐。从零开始训练意味着 SharedEncoder 和 DetRecoveryHead 需要重新收敛，浪费训练时间。建议先跑完 Stage 2 拿到 best.pt，再用它热启动 Stage 2.1。

### 3. 为什么 Stage 2.1 不支持 standard reconstruction head？

DetailAwareLightReconstructionHead 目前只实现了对 LightReconstructionHead 的扩展。standard head 的 decode 路径不同，暂不支持 detail fusion。如有需要可以后续补充。

### 4. detail_tx_ratio 一直不变是正常的吗？

正常。`detail_tx_ratio = detail_packet_size / main_features_size` 完全由模型结构决定，不受训练权重影响。它的值在默认配置下约为 0.01（1%）。

### 5. 可以直接用 Stage 2 的绘图脚本吗？

可以。`scripts/plot_stage2_metrics.py` 同时支持 Stage 2 和 Stage 2.1 的 `metrics.jsonl`。如果日志中包含 `detail_tx_ratio` 指标，会自动多生成一张 `detail_tx_ratio.png`。

## 十五、完整启动命令参考

```bash
# 1. 确认权重就绪
ls pretrained/rtdetr_r50vd/

# 2. 确认数据目录
ls /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train | head -5

# 3. 从 Stage 2 热启动训练 Stage 2.1
python scripts/train_mdvsc_stage2_1.py \
  --config configs/mdvsc_stage2_1.yaml \
  --data /root/autodl-tmp/dataset/VID/raw/ILSVRC2015/Data/VID/train \
  --stage2-checkpoint outputs/mdvsc_stage2/best.pt

# 4. 训练完成后绘图
python scripts/plot_stage2_metrics.py \
  --metrics outputs/mdvsc_stage2_1/metrics.jsonl
```
