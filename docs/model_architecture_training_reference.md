# 模型结构、训练超参数与指标换算参考

本文档面向当前仓库的“实际实现”，不是论文设想版。它把当前代码里已经存在的 RT-DETR 切点、语义信道、ProjectMDVSC、重建头、检测分支、训练阶段、损失定义和指标换算方式统一整理到一处。

相关实现入口：

- `src/semantic_rtdetr/detector/rtdetr_baseline.py`
- `src/semantic_rtdetr/semantic_comm/mdvsc.py`
- `src/semantic_rtdetr/semantic_comm/channel.py`
- `src/semantic_rtdetr/semantic_comm/codec.py`
- `src/semantic_rtdetr/training/stage1_trainer.py`
- `src/semantic_rtdetr/training/stage1_config.py`

## 1. 端到端链路总览

当前仓库中真正被训练的主链路如下：

1. 输入 GOP 视频张量：`[B, T, 3, H, W]`
2. 展平到 teacher RT-DETR 输入：`[B*T, 3, H, W]`
3. 经过 RT-DETR backbone 和 `encoder_input_proj`，得到 3 层 projected backbone feature
4. 把 3 层 feature 重新整理成序列：每层为 `feature_sequence_l = [B, T, C_l, H_l, W_l]`
5. 送入 `ProjectMDVSC`
6. `ProjectMDVSC` 先做每层的 latent 编码、公共/个体分解、mask、信道扰动和恢复
7. 恢复后的多层 feature 再分成两条轻分支：
   - detection refinement 分支
   - reconstruction refinement 分支
8. reconstruction 分支进入 `ReconstructionHead` 输出重建帧
9. detection 分支回填成 `EncoderFeatureBundle`，重新送回 RT-DETR hybrid encoder、decoder 和检测头

如果输入分辨率是当前默认的 `640 x 640`，则 RT-DETR 切点处的 3 层 projected feature 为：

| level | stride | shape |
| --- | --- | --- |
| level 0 | 8 | `[B*T, 256, 80, 80]` |
| level 1 | 16 | `[B*T, 256, 40, 40]` |
| level 2 | 32 | `[B*T, 256, 20, 20]` |

这组 shape 已经由仓库内样例产物 `outputs/rtdetr_baseline/feature_contract.json` 固化。

## 2. RT-DETR 到信道切点的结构与输入输出

### 2.1 输入预处理

`RTDetrBaseline.prepare_frame_tensor_batch` 会把 `[B*T, 3, H, W]` 的浮点张量转成 PIL 图像，再交给 Hugging Face 的 `AutoImageProcessor`。

默认单帧输入输出：

- 输入：`pixel_values = [N, 3, 640, 640]`
- 模型最终输出：
  - `logits = [N, 300, 80]`
  - `pred_boxes = [N, 300, 4]`

其中：

- `N = B*T`
- `300` 是 query 数
- `80` 是 COCO 类别数

### 2.2 RT-DETR backbone 到 projected backbone

当前本地权重 `pretrained/rtdetr_r50vd` 对应的关键配置为：

- `num_feature_levels = 3`
- `num_queries = 300`
- `d_model = 256`
- `encoder_layers = 1`
- `decoder_layers = 6`
- `encoder_ffn_dim = 1024`
- `decoder_ffn_dim = 1024`
- `encoder_in_channels = [512, 1024, 2048]`
- `feat_strides = [8, 16, 32]`

从 backbone 到切点的逐层结构如下。

| 模块 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| backbone 输出 level 0 | ResNet 主干输出 | `[N, 512, 80, 80]` | `[N, 512, 80, 80]` |
| backbone 输出 level 1 | ResNet 主干输出 | `[N, 1024, 40, 40]` | `[N, 1024, 40, 40]` |
| backbone 输出 level 2 | ResNet 主干输出 | `[N, 2048, 20, 20]` | `[N, 2048, 20, 20]` |
| `encoder_input_proj[0]` | `Conv2d(512,256,1,1,bias=False) + BatchNorm2d(256)` | `[N, 512, 80, 80]` | `[N, 256, 80, 80]` |
| `encoder_input_proj[1]` | `Conv2d(1024,256,1,1,bias=False) + BatchNorm2d(256)` | `[N, 1024, 40, 40]` | `[N, 256, 40, 40]` |
| `encoder_input_proj[2]` | `Conv2d(2048,256,1,1,bias=False) + BatchNorm2d(256)` | `[N, 2048, 20, 20]` | `[N, 256, 20, 20]` |

当前语义通信切点就固定在这里：`encoder_input_proj` 之后、`hybrid encoder` 之前。

### 2.3 切点契约：EncoderFeatureBundle

切点处数据结构是 `EncoderFeatureBundle`，包含：

- `feature_maps`: 3 个 `[N, 256, H_l, W_l]` 张量
- `spatial_shapes`: `[3, 2]`，记录各层 `[H_l, W_l]`
- `level_start_index`: `[3]`，记录 decoder flatten 偏移
- `strides`: `[8, 16, 32]`

默认 `640 x 640` 输入下：

- `spatial_shapes = [[80,80],[40,40],[20,20]]`
- `level_start_index = [0, 6400, 8000]`
- `flattened_sequence_length = 8400`

## 3. 语义编解码器与信道结构

### 3.1 当前 MVP codec

`FeatureSemanticCodec` 目前是占位实现，只做 level 选择，不做真正压缩。

| 步骤 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| `resolve_selected_levels` | 读取配置 `selected_levels`，默认全选 | `EncoderFeatureBundle` | level 索引列表 |
| `encode` | 选择若干 level，打包成 `FeaturePacket` | 完整 bundle | 只含 selected levels 的 `FeaturePacket` |
| `decode` | 把收到的 level 回填到原始 bundle | `FeaturePacket + reference_bundle` | 完整 `EncoderFeatureBundle` |

`FeaturePacket` 元信息包括：

- `selected_levels`
- `bypassed_levels`
- `adaptor`

当前 `adaptor` 只支持 `identity`。

### 3.2 当前特征信道

`FeatureChannel` 也有两种模式：

| 模式 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| `identity` | 直接 clone | feature packet | 无失真 feature packet |
| `awgn` | 按层测 signal power，再加高斯噪声 | feature packet + `snr_db` | 含噪声 feature packet |

当前信道侧直接输出的指标包括：

- `feature_mse`
- `feature_psnr_db`
- `measured_snr_db`
- `estimated_payload_bytes_fp32`

默认每层配置如下：

| level | 输入通道 | latent_dim | common_keep_ratio | individual_keep_ratio | block_size |
| --- | --- | --- | --- | --- | --- |
| level 0 | 256 | 48 | 0.5 | 0.125 | 8 |
| level 1 | 256 | 64 | 0.625 | 0.1875 | 4 |
| level 2 | 256 | 96 | 0.75 | 0.25 | 2 |

### 4.1 单层通信模块 PerLevelMDVSC

先展平成 `flat_sequence = [B*T, C_in, H_l, W_l]`，再依次经过下列子模块。

#### 4.1.1 feature_adaptor

| 顺序 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | `Conv2d(C_in, latent_dim, 1x1)` | `[B*T, C_in, H, W]` | `[B*T, latent_dim, H, W]` |
| 2 | `GELU` | 同上 | 同上 |
| 3 | `ResidualBlock(latent_dim)` | 同上 | 同上 |

`ResidualBlock` 结构是：

1. `Conv2d(C,C,3x3,padding=1)`
2. `GELU`
3. `Conv2d(C,C,3x3,padding=1)`
4. 残差相加
5. `GELU`

#### 4.1.2 latent_encoder

| 顺序 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | `Conv2d(latent_dim, latent_dim, 3x3, padding=1)` | `[B*T, latent_dim, H, W]` | 同 shape |
| 2 | `GELU` | 同上 | 同上 |
| 3 | `ResidualBlock(latent_dim)` | 同上 | 同上 |

#### 4.1.3 公共/个体分解

| 变量 | 定义 | shape |
| --- | --- | --- |
| `common_latent` | 沿时间维求平均 | `[B, latent_dim, H, W]` |
| `individual_latent` | `latent_sequence - common_latent` | `[B, T, latent_dim, H, W]` |

#### 4.1.4 common gate

`ChannelImportanceGate` 作用在 `common_latent` 上。

| 步骤 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | `abs().mean(dim=(2,3))` | `[B, C, H, W]` | `[B, C]` 通道重要性分数 |
| 2 | top-k 通道选择 | `[B, C]` | `[B, C]` |
| 3 | 训练期 soft mask + STE，推理期 hard mask | `[B, C]` | `[B, C, 1, 1]` |

输出 `common_mask` 后，实际送信道的是：


`BlockImportanceGate` 作用在每一帧的个体残差上。

| 步骤 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | 按通道求绝对值均值 | `[B, C, H, W]` | `[B, 1, H, W]` |
| 2 | padding 到 block 对齐 | `[B, 1, H, W]` | padded tensor |
| 3 | `avg_pool2d(block_size, stride=block_size)` | padded tensor | `[B, 1, H/block, W/block]` |
| 4 | flatten 后 top-k | block score | block mask |
| 5 | repeat_interleave 回原图 | block mask | `[B, 1, H, W]` |

输出 `individual_mask` 后，送信道的是：

- `transmitted_individual = frame_residual * individual_mask`

每个 level 的每一帧都执行：

1. `transmitted_common` 经过 `identity` 或 `awgn`
2. `transmitted_individual` 经过 `identity` 或 `awgn`
3. 两者相加后进入 `latent_decoder`
4. 再经过 `feature_decoder`
5. 再经过 `refinement`

具体层结构如下。

`latent_decoder`:

| 顺序 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | `ResidualBlock(latent_dim)` | `[B, latent_dim, H, W]` | 同 shape |
| 2 | `Conv2d(latent_dim, latent_dim, 3x3, padding=1)` | 同上 | 同 shape |
| 3 | `GELU` | 同上 | 同 shape |

`feature_decoder`:

| 顺序 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | `ResidualBlock(latent_dim)` | `[B, latent_dim, H, W]` | 同 shape |
| 2 | `Conv2d(latent_dim, C_in, 1x1)` | 同上 | `[B, C_in, H, W]` |

`refinement`:

| 顺序 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | `Conv2d(C_in, C_in, 3x3, padding=1)` | `[B, C_in, H, W]` | 同 shape |
| 2 | `GELU` | 同上 | 同 shape |
| 3 | `ResidualBlock(C_in)` | 同上 | 同 shape |
- `individual_mask_l = [B, T, 1, H_l, W_l]`
- `LevelTransmissionStats`

恢复后的 3 层 feature 会分别进入两组相同结构的 `TaskAdaptationBlock`：

- `detection_refinement_heads`
每个 `TaskAdaptationBlock(C)` 的结构：

1. `Conv2d(C, C, 1x1)`
2. `GroupNorm(groups, C)`
3. `GELU`
其中 `ReconstructionResidualBlock(C)` 的结构是：

1. `GroupNorm(groups, C)`
2. `GELU`
3. `Conv2d(C, C, 3x3, padding=1)`
4. `GroupNorm(groups, C)`
5. `GELU`
6. `Conv2d(C, C, 3x3, padding=1)`
7. 残差相加

- 输入：`[B, T, 256, H_l, W_l]`
- 输出：`[B, T, 256, H_l, W_l]`

### 5.1 语义投影与多尺度融合

| 模块 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| `level2_semantic_proj` | `FusionBlock(256,256)` | `[B,256,20,20]` | `[B,256,20,20]` |
| `level1_semantic_proj` | `FusionBlock(256,256)` | `[B,256,40,40]` | `[B,256,40,40]` |
| `level0_semantic_proj` | `FusionBlock(256,256)` | `[B,256,80,80]` | `[B,256,80,80]` |
| `level0_detail_proj` | `FusionBlock(256,128)` | `[B,256,80,80]` | `[B,128,80,80]` |

`FusionBlock(in,out)` 的结构：
3. `GELU`
4. `ReconstructionResidualBlock(out)`
5. `ReconstructionResidualBlock(out)`
1. `level2` nearest upsample 到 `40x40`，与 `level1` concat，得到 `[B,512,40,40]`
2. 经过 `fuse_16 = FusionBlock(512,256)`，得到 `[B,256,40,40]`
3. `fused_16` nearest upsample 到 `80x80`，与 `level0_semantic` concat，得到 `[B,512,80,80]`
4. 经过 `fuse_8 = FusionBlock(512,256)`，得到 `[B,256,80,80]`
5. 再与 `level0_detail` concat，得到 `[B,384,80,80]`
6. 经过 `detail_fuse_8 = FusionBlock(384,256)`，得到 `[B,256,80,80]`

### 5.2 渐进上采样解码

1. `Conv2d(in, out*4, 3x3, padding=1)`
2. `GELU`
3. `PixelShuffle(2)`

当前 3 个上采样阶段如下：

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| `up_stage2(128,64)` | `[B,128,160,160]` | `[B,64,320,320]` |
| `up_stage3(64,32)` | `[B,64,320,320]` | `[B,32,640,640]` |

`decoded = [B,32,H,W]` 后继续执行：

1. `output_refinement`: 2 个 `ReconstructionResidualBlock(32)`
2. `base_layer`: `Conv2d(32,3,1x1) + Sigmoid`，得到 `base = [B,3,H,W]`
3. `high_frequency_refinement`: 2 个 `ReconstructionResidualBlock(32)`
因此 reconstruction head 的最终输出有 3 份：

- `reconstructed_frames = [B, T, 3, H, W]`
- `reconstructed_high_frequency_residuals = [B, T, 3, H, W]`

## 6. 检测分支 downstream RT-DETR 的结构与输入输出

当前 ProjectMDVSC 并没有自己实现一个新的检测头，而是把 `detection_sequences` 回填回 `EncoderFeatureBundle`，继续复用原始 RT-DETR 的 downstream 检测路径。

### 6.1 hybrid encoder

当前本地 RT-DETR r50vd 的 hybrid encoder 主要由 4 部分构成：

| 子模块 | 结构摘要 | 输入 | 输出 |
| --- | --- | --- | --- |
| `encoder` | 1 个 `RTDetrEncoder`，内部只有 1 个 `RTDetrEncoderLayer` | 3 层 `[N,256,H_l,W_l]` | 编码后的多尺度特征 |

`RTDetrEncoderLayer` 内部结构：

1. self-attention，`q/k/v/out` 全是 `Linear(256,256)`
2. `LayerNorm(256)`
3. FFN: `Linear(256,1024) -> GELU -> Linear(1024,256)`
4. `LayerNorm(256)`
当前共有 3 个 `decoder_input_proj[level]`，结构相同：

- `Conv2d(256,256,1x1,bias=False) + BatchNorm2d(256)`

输入输出 shape 不变，只做线性投影和归一化。

### 6.3 RT-DETR decoder 与检测头

1. `self_attn`: `RTDetrMultiheadAttention`
   - `q_proj: Linear(256,256)`
   - `k_proj: Linear(256,256)`
   - `v_proj: Linear(256,256)`
   - `out_proj: Linear(256,256)`
2. `LayerNorm(256)`
   - `value_proj: Linear(256,256)`
   - `output_proj: Linear(256,256)`
4. `LayerNorm(256)`

decoder 相关 head：

| 模块 | 结构 | 输入 | 输出 |
| --- | --- | --- | --- |
| `query_pos_head` | `Linear(4,512) -> Linear(512,256)` | reference boxes | `[N,300,256]` |
| `class_embed` | 6 个 `Linear(256,80)` | decoder hidden state | 每层分类 logits |

## 7. Stage-1 训练信息与超参数

当前 stage-1 训练是三阶段：

| 阶段 | 作用 | 训练参数 | 冻结参数 |
| --- | --- | --- | --- |
| `reconstruction_pretrain` | 先让重建头学会从 projected feature 重建图像 | `reconstruction_head`、`reconstruction_refinement_heads` | `level_modules`、`detection_refinement_heads`、全部 RT-DETR |
| `mdvsc_bootstrap` | 先让 MDVSC 结构逼近 teacher feature | `level_modules` | `reconstruction_head`、两组 refinement 中的 detection/reconstruction head、全部 RT-DETR |
| `joint_training` | 联合优化特征恢复、图像重建和检测一致性 | `level_modules`、`reconstruction_head`、`reconstruction_refinement_heads`、`detection_refinement_heads` | 全部 RT-DETR |

RT-DETR teacher 在整个 stage-1 中始终冻结，只作为 feature 与 detection consistency 的参考。

## 7.2 stage-1 全部配置项

下面按代码默认值和 YAML 默认值整理当前训练超参数。

| `hf_name` | `PekingU/rtdetr_r50vd` | RT-DETR 权重名 |
| `local_path` | `pretrained/rtdetr_r50vd` | 本地权重目录，存在时优先本地加载 |
| `cache_dir` | `None` | HF cache |
| `device` | `auto` | 自动选择 `cuda` 或 `cpu` |

### B. data 配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset_name` | `generic` | 数据集模式，支持 `generic` 和 `imagenet_vid` |
| `train_source_path` | `None` | 训练数据根路径 |
| `val_source_path` | `None` | 验证数据根路径 |
| `recursive` | `true` | 是否递归扫描目录 |
| `index_cache_dir` | `.cache/stage1_data` | 数据索引缓存目录 |
| `subset_seed` | `42` | 子集抽样种子 |
| `source_fraction` | `1.0` | 保留多少比例的源序列 |
| `sample_fraction` | `1.0` | 保留多少比例的 GOP 样本 |
| `gop_size` | `4` | 每个样本的帧数 |
| `frame_height` | `640` | resize 高度 |
| `frame_width` | `640` | resize 宽度 |
| `frame_stride` | `1` | GOP 内帧间隔 |
| `gop_stride` | `2` | 样本滑窗步长 |
| `max_frames_per_source` | `None` | 每个 source 最多使用多少帧 |
| `max_sources` | `None` | 最多保留多少 source |
| `max_samples` | `None` | 最多保留多少样本 |
| `train_val_split` | `0.1` | 没有单独验证集时，从 train 划出验证集比例 |

### C. mdvsc 配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `feature_channels` | `[256,256,256]` | 3 层输入特征通道数 |
| `latent_dims` | `[48,64,96]` | 各层 latent 维度 |
| `common_keep_ratios` | `[0.5,0.625,0.75]` | 公共分支通道保留比例 |
| `individual_keep_ratios` | `[0.125,0.1875,0.25]` | 个体分支空间保留比例 |
| `block_sizes` | `[8,4,2]` | 个体分支 block mask 大小 |
| `reconstruction_hidden_channels` | `256` | 重建主干宽度 |
| `reconstruction_detail_channels` | `128` | 最高分辨率细节支路宽度 |
| `apply_masks` | `false` in dataclass, `true` in `configs/mdvsc_stage1.yaml` | 是否启用结构化 mask |
| `channel_mode` | `identity` | 信道模式 |
| `snr_db` | `20.0` | AWGN 模式目标 SNR |

说明：`Stage1MDVSCConfig` 代码默认 `apply_masks = false`，但仓库主 YAML `configs/mdvsc_stage1.yaml` 已经把它覆盖成 `true`。

### D. optimization 配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `batch_size` | `1` | batch size |
| `num_workers` | `4` | DataLoader worker 数 |
| `use_amp` | `true` | CUDA 上默认启用 AMP |
| `amp_dtype` | `float16` | 支持 `float16` / `bfloat16` |
| `reconstruction_pretrain_epochs` | `3` | 第一阶段 epoch 数 |
| `reconstruction_pretrain_lr` | `3e-4` | 第一阶段学习率 |
| `mdvsc_bootstrap_epochs` | `3` | 第二阶段 epoch 数 |
| `mdvsc_bootstrap_lr` | `1e-4` | 第二阶段学习率 |
| `epochs` | `10` | joint training epoch 数 |
| `lr` | `1e-4` | joint training 学习率 |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `scheduler` | `cosine` | 支持 `constant` / `cosine` |
| `warmup_epochs` | `1` | cosine 前 warmup epoch |
| `warmup_start_factor` | `0.2` | warmup 起始倍率 |
| `min_lr_ratio` | `0.1` | cosine 最小学习率比例 |
| `grad_clip_norm` | `1.0` | 梯度裁剪阈值 |
| `log_every` | `10` | 日志打印步频 |
| `save_every_epochs` | `1` | checkpoint 保存周期 |
| `max_steps_per_epoch` | `None` | 每个 epoch 的最大 step |
| `seed` | `42` | 训练随机种子 |

### E. loss 配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `feature_loss_weight` | `1.0` | feature loss 总权重 |
| `recon_l1_weight` | `1.0` | L1 重建权重 |
| `recon_mse_weight` | `0.25` | MSE 重建权重 |
| `recon_ssim_weight` | `0.25` | SSIM 重建权重 |
| `recon_edge_weight` | `0.2` | 边缘损失权重 |
| `detection_logit_weight` | `0.05` | detection logits 一致性权重 |
| `detection_box_weight` | `0.05` | detection boxes 一致性权重 |
| `level_loss_weights` | `[1.0,1.0,1.0]` | 各层 feature loss 权重 |

### F. output 配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `output_dir` | `outputs/mdvsc_stage1` | 输出目录 |
| `save_visualizations` | `true` | 是否保存训练可视化 |
| `visualization_every_epochs` | `1` | 可视化保存周期 |
| `visualization_num_frames` | `4` | 每次可视化的帧数 |

## 7.3 ImageNet VID 子集训练配置

`configs/mdvsc_stage1_imagenet_vid_subset.yaml` 对默认值做了如下关键覆盖：

| 字段 | 覆盖值 |
| --- | --- |
| `data.dataset_name` | `imagenet_vid` |
| `data.source_fraction` | `0.1` |
| `data.sample_fraction` | `0.2` |
| `optimization.batch_size` | `2` |
| `optimization.reconstruction_pretrain_epochs` | `10` |
| `optimization.reconstruction_pretrain_lr` | `2e-4` |
| `optimization.mdvsc_bootstrap_epochs` | `6` |
| `optimization.epochs` | `10` |
| `optimization.lr` | `5e-5` |
| `loss.recon_l1_weight` | `0.75` |
| `loss.recon_mse_weight` | `0.1` |
| `loss.recon_ssim_weight` | `1.0` |
| `loss.detection_logit_weight` | `0.02` |
| `loss.detection_box_weight` | `0.02` |
| `output.output_dir` | `outputs/mdvsc_stage1_imagenet_vid_subset` |

## 7.4 优化器、学习率调度和 AMP

当前训练器的固定实现是：

- 优化器：`AdamW`
- reconstruction pretrain 阶段：不使用 scheduler
- bootstrap / joint 阶段：`cosine` 或 `constant`
- `cosine` 模式下支持 `LinearLR` warmup + `CosineAnnealingLR`
- 只有在 CUDA 且 `amp_dtype=float16` 时启用 `GradScaler`
- 每步训练后执行 `clip_grad_norm_`

## 8. 当前可直接记录的损失、含义与指标换算

当前训练器每个 epoch 会记录这些量：

- `total_loss`
- `feature_loss`
- `recon_l1_loss`
- `recon_mse_loss`
- `recon_ssim_loss`
- `recon_edge_loss`
- `detection_logit_loss`
- `detection_box_loss`
- `common_active_ratio`
- `individual_active_ratio`

下面给出各项定义与如何换成更具体的指标。

### 8.1 feature_loss

定义：

`feature_loss` 是各 level 的加权 `SmoothL1Loss(restored_sequence, target_sequence)` 之和。

当前公式可以写成：

$$
L_{feature} = \sum_l w_l \cdot \operatorname{SmoothL1}(\hat{F}_l, F_l)
$$

它本质上是“恢复特征和 teacher 特征差多少”的代理指标，不是直接可发表的视觉指标。

推荐换算方式：

1. 直接把它作为 feature distortion 代理损失报告。
2. 如果想变成更可解释的数值，额外统计每层 feature MSE，再换算 feature PSNR：

$$
\operatorname{PSNR}_{feature} = 20 \log_{10}(\max |F|) - 10 \log_{10}(\operatorname{MSE}(\hat{F}, F))
$$

3. 如果要和信道脚本保持一致，使用 `channel.py` 的 `feature_mse` / `feature_psnr_db` 定义。

### 8.2 recon_l1_loss

定义：

$$
L_{L1} = \operatorname{mean}(|\hat{I} - I|)
$$

它本身就等于像素级 MAE，可直接报告为：

- `pixel_mae = recon_l1_loss`

如果想要更直观，可以再乘 `255` 变成 8-bit 图像域误差：

$$
\operatorname{MAE}_{8bit} = 255 \cdot L_{L1}
$$

### 8.3 recon_mse_loss

定义：

$$
L_{MSE} = \operatorname{mean}((\hat{I} - I)^2)
$$

它可直接换成：

- `RMSE = sqrt(recon_mse_loss)`
- `PSNR`

如果图像已经归一化到 `[0,1]`，则：

$$
\operatorname{PSNR} = 10 \log_{10}\left(\frac{1}{L_{MSE}}\right) = -10 \log_{10}(L_{MSE})
$$

如果要换成 8-bit 域：

$$
\operatorname{PSNR}_{8bit} = 10 \log_{10}\left(\frac{255^2}{255^2 \cdot L_{MSE}}\right)
$$

数值上和归一化域公式等价。

### 8.4 recon_ssim_loss

定义：

当前实现里：

$$
L_{SSIM} = 1 - \operatorname{SSIM}(\hat{I}, I)
$$

因此换算非常直接：

$$
\operatorname{SSIM} = 1 - L_{SSIM}
$$

这也是最推荐写进图表或论文的方式，而不是直接报告 `recon_ssim_loss`。

### 8.5 recon_edge_loss

定义：

它是 x/y 方向梯度差异的 L1：

$$
L_{edge} = \operatorname{L1}(\partial_x \hat{I}, \partial_x I) + \operatorname{L1}(\partial_y \hat{I}, \partial_y I)
$$

它不等于标准视觉 benchmark 指标，更适合作为辅助质量约束。

推荐报告方式：

1. 直接报告 `edge_l1 = recon_edge_loss`
2. 如需“越大越好”的分数，可自定义：

$$
\operatorname{EdgeScore} = \frac{1}{1 + L_{edge}}
$$

但这不是当前代码内置指标，最好在图表中明确写成自定义换算。

### 8.6 detection_logit_loss

定义：

$$
L_{logit} = \operatorname{MSE}(\hat{Z}, Z_{teacher})
$$

其中：

- `\hat{Z}` 是 student feature 送回 RT-DETR 后的分类 logits
- `Z_teacher` 是冻结 teacher 的 logits

它不是 mAP，也不是分类准确率，而是 teacher-student 一致性度量。

推荐换算：

- `logit_rmse = sqrt(detection_logit_loss)`

如果你要转换成真实检测指标，需要：

1. 对 `student_outputs` 做 `post_process_object_detection`
2. 和带标注的验证集做 COCO AP / mAP 评测

也就是说，`detection_logit_loss` 只能说明“像不像 teacher”，不能直接等价于 AP。

### 8.7 detection_box_loss

定义：

$$
L_{box} = \operatorname{L1}(\hat{B}, B_{teacher})
$$

这里比较的是 RT-DETR 输出的归一化 box 参数，因此当前它是“归一化坐标空间下的 box MAE”。

推荐换算：

- `normalized_box_mae = detection_box_loss`
- 如果要换成像素尺度，可对 x/w 乘图像宽度，对 y/h 乘图像高度

但和 `detection_logit_loss` 一样，它仍然不是 mAP。要得到真实检测指标，必须走有标注数据集评测。

### 8.8 common_active_ratio 和 individual_active_ratio

它们不是 loss，而是稀疏激活比例统计。

| 指标 | 含义 |
| --- | --- |
| `common_active_ratio` | 公共 latent 通道平均保留比例 |
| `individual_active_ratio` | 个体 residual 空间块平均保留比例 |

它们可以进一步换算成近似通信负载。若第 `l` 层 latent 维度为 `D_l`、空间分辨率为 `H_l x W_l`、时间长度为 `T`，则不考虑索引开销和熵编码时，该层近似传输元素数为：

$$
N_l \approx D_l H_l W_l \rho^{common}_l + T D_l H_l W_l \rho^{individual}_l
$$

如果按 fp32 粗略估算 GOP 级 bit/pixel，可写成：

$$
\operatorname{bpp}_{gop} \approx \frac{32 \sum_l N_l}{T H W}
$$

这比当前 MVP `channel.py` 的全量 fp32 payload 更接近 ProjectMDVSC 的真实 latent 负载，但当前训练器还没有自动把这个指标写出。

## 9. 当前已经落盘的训练与评测产物

stage-1 训练输出目录中当前会生成：

- `resolved_config.json`
- `dataset_info.json`
- `metrics.jsonl`
- `latest.pt`
- `best.pt`
- `epoch_XXX.pt`
- `final_summary.json`
- `visualizations/...`（如果开启）

其中：

- `metrics.jsonl` 适合做训练曲线
- `final_summary.json` 适合汇总一次实验的最终配置和最好结果
- `visualizations` 适合人工检查重建、feature 对齐和 mask 行为

## 10. 文档使用建议

如果你要把当前实验写成报告，建议按下面顺序引用这些量：

1. 通信侧：`feature_mse`、`feature_psnr_db`、`measured_snr_db`、payload / bpp
2. 重建侧：`PSNR`、`SSIM = 1 - recon_ssim_loss`、`pixel_mae`
3. 检测侧：真正的 mAP 必须额外跑标注集评测；当前训练内置的只是 `detection_logit_loss` 和 `detection_box_loss` 两个 teacher-student proxy
4. 稀疏侧：`common_active_ratio`、`individual_active_ratio`，以及基于它们推导的近似 latent bpp

这样能把“特征恢复是否稳定”“图像重建是否可用”“检测是否还保真”“通信开销是多少”四条线明确区分开。