# RT-DETR Encoder 特征接口

本文档固定语义通信栈使用的 RT-DETR 切点接口。

如果需要查看整个仓库范围内的架构假设、训练意图和项目级设计选择，请参考 [docs/project_context.md](docs/project_context.md)。

## 切点

- 切分位置：RT-DETR hybrid encoder 之后、decoder 输入展平之前。
- 负载类型：多尺度 feature map，而不是 RGB 像素，也不是 decoder query。
- 当前参考模型：PekingU/rtdetr_r50vd。

## 负载结构

每个样本在切点处表示为一个 EncoderFeatureBundle，包含以下字段：

- feature_maps：shape 为 [B, C, H, W] 的张量列表。
- spatial_shapes：shape 为 [num_levels, 2] 的张量，用于记录每个 level 的 [H, W]。
- level_start_index：shape 为 [num_levels] 的张量，用于记录 decoder 展平准备时的偏移量。
- strides：每个 level 的空间步长，可选。

在切点处，这个 bundle 仍然只是原始语义源，还不是目标项目中的最终传输对象。目标项目会先做语义适配和 latent 压缩，再决定实际的通信格式。

语义通信侧当前会把这个 bundle 包装成一个 FeaturePacket：

- selected_levels：要经过语义信道传输的 encoder level。
- bypassed_levels：当前 MVP 中作为旁路保留、用于层级消融实验的 level。
- adaptor：传输前使用的特征 adaptor 模式。当前 MVP 只支持 identity。

在当前仓库 MVP 中，FeaturePacket 仍然只是面向信道实验的占位容器。在目标项目架构中，它应逐步演化成一个 latent 域语义 packet，而不是直接切一段原始 RT-DETR feature bundle 来传。

以当前基线图像 byc.jpg 为例，契约如下：

- level 0：shape 为 [1, 256, 80, 80]，stride 为 8
- level 1：shape 为 [1, 256, 40, 40]，stride 为 16
- level 2：shape 为 [1, 256, 20, 20]，stride 为 32

对应的 decoder 展平长度为 8400。

## 信道侧要求

- 语义通信模块必须保持所有被选中 level 的顺序和张量形状。
- 接收端必须精确恢复每个被传输 level 的张量形状。
- spatial_shapes 和 level_start_index 被视为控制元数据，不从噪声负载中反推。
- 当前 MVP 默认使用 float32 特征张量。

## Level 选择

- 当前默认传输集合是 level [0, 1, 2]。
- semcom 配置也可以只选择其中一部分 level 做消融。
- 当前 MVP 中，不在 selected_levels 中的 level 会被原样旁路。这只是为了分析方便，并不是最终部署假设。

目标视频系统最终应使用视频序列特征张量和 latent 域 codec。当前 level 选择机制的作用，是稳定接口实验和早期消融分析。

对第一版正式项目模型，默认约定如下：

- 三个 level 都传；
- 但三个 level 采用不同的 latent 维度与不同的压缩预算；
- 默认 latent 维度为 level 0 -> 48，level 1 -> 64，level 2 -> 96；
- 默认公共分支按单通道独立选择是否保留，个体分支按空间块做结构化稀疏；
- 默认详细值以 [docs/project_context.md](docs/project_context.md) 中的 V1 默认配置提案为准。

## MVP 信道模式

- identity：严格透传，用于无损验证。
- awgn：直接在特征张量上注入加性高斯白噪声。

## Decoder 交接

恢复后的 feature map 会先回填成完整的 EncoderFeatureBundle，然后经过 RT-DETR decoder_input_proj、展平，并送入原始 decoder，整个 decoder API 不做改动。

在目标项目架构中，这个恢复语义包除了送入 RT-DETR 检测分支外，还会同时进入一个重建分支。