# my_first_try
# semantic-rtdetr-video

RT-DETR 与 MDVSC 联合视频语义传输与目标检测研究项目。

## 文档

- [docs/project_context.md](docs/project_context.md)：项目公共上下文与当前目标架构。
- [docs/feature_interface.md](docs/feature_interface.md)：RT-DETR 切点与语义特征接口说明。
- [docs/plan.md](docs/plan.md)：项目概要、阶段计划与执行优先级。
- [docs/stage1_training_manual.md](docs/stage1_training_manual.md)：第一阶段训练中文操作说明书。

## 当前阶段
- [x] 项目概要
- [x] 仓库骨架
- [x] 开发容器
- [x] RT-DETR 基线
- [x] 语义通信基线
- [x] MDVSC 模型骨架
- [x] 第一阶段训练代码

## RT-DETR 基线最小可行版本

第一个可执行里程碑是一个可切分的 RT-DETR 基线，而不是完整的 RT-DETR + MDVSC 联合系统。
当前仓库已经包含一条最小离线推理路径，可以：

- 在单张图像上运行原始 RT-DETR 检测；
- 导出 hybrid encoder 之后的多尺度 encoder 侧特征图；
- 将这些特征图回灌到 decoder，并与直接基线输出做 round-trip 对比；
- 让 encoder 特征经过 identity 或 awgn 特征信道，并测量通信侧扰动。

### 快速开始

1. 安装依赖：

```bash
python -m pip install -r requirements.txt
```

2. 先把 RT-DETR 权重下载到项目目录：

```bash
python scripts/download_rtdetr_weights.py
```

默认会下载到 `pretrained/rtdetr_r50vd/`。当前配置已经默认优先使用这个本地目录；目录存在时，后续训练和推理不会再走线上下载。

3. 使用一张图运行基线脚本：

```bash
python scripts/run_rtdetr_baseline.py --image /absolute/path/to/example.jpg
```

4. 可选：使用 YAML 配置作为默认入口：

```bash
python scripts/run_rtdetr_baseline.py --config configs/rtdetr_baseline.yaml --image /absolute/path/to/example.jpg
```

5. 运行语义通信最小可行管线：

```bash
python scripts/run_semcom_pipeline.py --config configs/rtdetr_semcom_mvp.yaml --image /absolute/path/to/example.jpg
```

默认输出会写入 outputs/rtdetr_baseline/：

- summary.json：检测结果与 round-trip 一致性统计。
- feature_contract.json：RT-DETR 切点处的多尺度特征契约。
- encoder_features.pt：用于离线检查的 encoder 特征序列化文件。
- annotated.jpg：可选的检测可视化图。

### 当前范围

当前代码库仍然有意保持在基线与接口验证阶段。仓库已经包含单图 RT-DETR 切点验证和最小语义通信 MVP，但目标项目架构是一个视频输入、latent 域语义压缩、检测分支与重建分支并存的联合系统。

目标设计假设定义在 [docs/project_context.md](docs/project_context.md)。

其中，第一版默认技术配置也已经写入 [docs/project_context.md](docs/project_context.md)，包括：全 level 传输、每层 latent 维度、公共/个体分支的默认稀疏策略，以及训练阶段默认路线。

当前推荐训练路线也定义在 [docs/project_context.md](docs/project_context.md)：先训练无语义通信的双头上界模型，再插入 MDVSC 训练通信模块，最后做分层解冻的整体微调。

当前仓库已经补上第一阶段训练入口：

```bash
python scripts/train_mdvsc_stage1.py --config configs/mdvsc_stage1.yaml --data /absolute/path/to/data
```

当前默认训练路线已经改成三段式：先用冻结的 RT-DETR teacher 特征单独预训练 reconstruction head，再冻结 reconstruction head 训练 MDVSC，最后做 MDVSC + reconstruction head 联合微调。重建项除了 L1 和 MSE，也默认加入了 SSIM loss。

详细步骤见 [docs/stage1_training_manual.md](docs/stage1_training_manual.md)。

第一阶段通信接口切片定义在 [docs/feature_interface.md](docs/feature_interface.md)。

当前 semcom 管线分成三个阶段：

- 语义编解码器：选择哪些 encoder level 被传输；
- 特征信道：对被传输 level 注入 identity 或 AWGN 扰动；
- 语义解码器：把被传输 level 回填成完整的 RT-DETR 特征包。

这条三阶段管线目前是为了后续接入 MDVSC 而搭建的仓库骨架，并不是最终模型定义。
