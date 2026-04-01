# 第一阶段训练中文操作说明书

本文面向当前仓库里的第一阶段训练入口，目标是让你在服务器上把训练真正跑起来，并且知道每一步为什么这样做。

当前第一阶段训练的核心目的有三件事：

- 让 MDVSC 恢复出来的多尺度特征尽量逼近 RT-DETR encoder 特征；
- 让恢复出来的语义特征能够重建出视频帧；
- 让恢复特征送回冻结 RT-DETR 后，检测输出尽量接近 teacher 输出。

当前仓库默认已经把 mask 打开，也就是公共分支和个体分支都会执行结构化筛选；但这仍然属于“先把结构训稳”的阶段，不是最终版的完整率失真联合训练。

当前默认训练流程已经升级为三段式：

- 先冻结 RT-DETR teacher，只训练 reconstruction head，让它先学会把 projected backbone 特征还原成图像；
- 再冻结 reconstruction head，只训练 MDVSC，并且这一阶段只看 feature loss；
- 最后再做 MDVSC + reconstruction head 的联合训练，同时优化图像重建和检测一致性。

这样做的目的，是避免 reconstruction head 和 MDVSC 同时从随机初始化开始、彼此拖累收敛，同时减少在 MDVSC 刚接入时重建头被不稳定输入分布带偏的风险。

## 一、训练前需要准备什么

训练前必须满足两个前提：

1. 你本地或服务器上有可读的视频帧数据。
2. 运行环境能够拿到 RT-DETR teacher 权重。

现在仓库已经支持把 RT-DETR 权重提前下载到项目目录里。默认配置会优先读取：

- `pretrained/rtdetr_r50vd/`

如果这个目录存在，训练和推理会直接从本地加载，不再向 Hugging Face 发起下载请求；如果目录不存在，代码才会回退到 `hf_name` 对应的远端仓库。

## 二、服务器环境准备

推荐环境：

- Linux 服务器；
- Python 3.10 及以上；
- GPU 优先，没有 GPU 也能跑，但会慢很多；
- 首次准备权重时能访问 Hugging Face，或者你已经从别的机器拷贝好了本地权重目录。

建议安装流程：

```bash
git clone <你的仓库地址>
cd my_first_try
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_rtdetr_weights.py
```

如果你使用 conda，也可以换成自己的环境管理方式，但核心是确保 `requirements.txt` 里的依赖都已安装。

下载脚本执行完成后，权重默认位于：

```text
pretrained/
  rtdetr_r50vd/
```

如果你想放到别的位置，也可以手动指定：

```bash
python scripts/download_rtdetr_weights.py --output-dir /absolute/path/to/custom_rtdetr_r50vd
```

这时请同步修改对应 YAML 里的：

```yaml
model:
  local_path: /absolute/path/to/custom_rtdetr_r50vd
```

或者第一阶段训练配置里的：

```yaml
detector:
  local_path: /absolute/path/to/custom_rtdetr_r50vd
```

## 三、数据目录怎么准备

当前训练器支持两类输入：

1. 单个视频文件。
2. 帧目录，也就是一个目录里按顺序放图片帧。

训练器也支持递归扫描一个根目录，因此你可以直接把多个序列目录放在同一个上层目录里。

当前版本已经把“源目录扫描与帧索引”单独缓存到 `data.index_cache_dir`，默认是 `.cache/stage1_data`。第一次扫描大数据集仍然会慢，但后续重复启动训练时会优先复用缓存，不再每次都重新做完整索引。

一个通用目录示例：

```text
data/
  clip_001/
    000001.jpg
    000002.jpg
    000003.jpg
    ...
  clip_002/
    000001.jpg
    000002.jpg
    ...
```

## 四、ImageNet VID 应该怎么接

如果你使用的是完整 ImageNet VID，不需要手工拷贝出一个小数据集。

当前代码已经支持：

- 直接把完整的 ImageNet VID 帧目录根路径交给训练器；
- 再通过配置只抽取其中一部分序列和样本来训练。

推荐把 `train_source_path` 指向 ImageNet VID 的帧目录根，例如：

```text
/absolute/path/to/ILSVRC/Data/VID/train
```

然后使用下面三个字段控制“只取一部分数据来训练”：

- `data.subset_seed`：子集抽样随机种子。
- `data.source_fraction`：从全部序列目录里保留多少比例。
- `data.sample_fraction`：生成 GOP 样本后，再保留多少比例的样本。

比如：

```yaml
data:
  dataset_name: imagenet_vid
  train_source_path: /absolute/path/to/ILSVRC/Data/VID/train
  subset_seed: 42
  source_fraction: 0.2
  sample_fraction: 0.2
```

这组配置的含义不是“精确到数学意义上的 20% 全量数据”，而是：

- 先从完整序列集合里稳定地抽取大约 20% 的序列；
- 再从这些序列产生的 GOP 样本里稳定地抽取大约 20% 的样本。

## 五、配置文件怎么选

当前最常用的配置有两个：

- `configs/mdvsc_stage1.yaml`：通用第一阶段配置。
- `configs/mdvsc_stage1_imagenet_vid_subset.yaml`：面向完整 ImageNet VID 的子集训练配置。

如果你已经确定使用完整 ImageNet VID，并且只想先取一部分训练，建议直接从后者开始。

## 六、最常改的配置项

### 数据相关

- `data.dataset_name`：当前可写 `generic` 或 `imagenet_vid`。
- `data.train_source_path`：训练数据根路径。
- `data.val_source_path`：验证数据路径；如果留空，会从训练集里切分。
- `data.subset_seed`：子集抽样种子。
- `data.source_fraction`：抽多少比例的序列。
- `data.sample_fraction`：抽多少比例的 GOP 样本。
- `data.index_cache_dir`：数据索引缓存目录；设为 `null` 可以关闭缓存。
- `data.gop_size`：每个样本有多少帧。
- `data.frame_height` 和 `data.frame_width`：统一 resize 后送入模型的分辨率。

### 训练相关

- `optimization.batch_size`：批大小。
- `optimization.reconstruction_pretrain_epochs`：重建头预训练轮数。
- `optimization.reconstruction_pretrain_lr`：重建头预训练学习率。
- `optimization.mdvsc_bootstrap_epochs`：冻结重建头时，单独训练 MDVSC 的轮数。
- `optimization.mdvsc_bootstrap_lr`：MDVSC bootstrap 阶段的学习率。
- `optimization.epochs`：训练轮数。
- `optimization.lr`：学习率。
- `optimization.scheduler`：当前支持 `constant` 和 `cosine`。
- `optimization.warmup_epochs`：warmup 轮数。
- `optimization.max_steps_per_epoch`：可选，用于限制每个 epoch 最多跑多少 step。

### 模型相关

- `detector.local_path`：RT-DETR 本地权重目录，默认是 `pretrained/rtdetr_r50vd`。
- `detector.cache_dir`：可选的 Hugging Face 缓存目录。
- 当前第一阶段默认保持 RT-DETR 冻结，包括 reconstruction head 预训练、MDVSC bootstrap 和联合训练阶段都不解冻 teacher。
- `mdvsc.apply_masks`：当前建议保持 `true`。
- `mdvsc.channel_mode`：第一阶段建议先用 `identity`。
- `mdvsc.latent_dims`：当前默认是 48 / 64 / 96。
- `mdvsc.common_keep_ratios`：当前默认是 0.5 / 0.625 / 0.75。
- `mdvsc.individual_keep_ratios`：当前默认是 0.125 / 0.1875 / 0.25。
- `mdvsc.reconstruction_hidden_channels`：语义主干解码宽度，默认 192。
- `mdvsc.reconstruction_detail_channels`：最高分辨率细节分支宽度，默认 96。

### 损失相关

- `loss.recon_l1_weight`：像素 L1 重建权重。
- `loss.recon_mse_weight`：像素 MSE 重建权重。
- `loss.recon_ssim_weight`：SSIM 重建权重。
- reconstruction head 预训练阶段默认只使用重建相关损失，不使用 feature loss 和 detection consistency loss。
- MDVSC bootstrap 阶段默认只使用 feature loss，不使用任何重建损失或 detection consistency loss。
- 只有最后的 joint training 阶段才会把图像重建损失和 detection consistency loss 一起接入总损失。

## 七、推荐起步配置

如果你要在完整 ImageNet VID 上先取一部分训练，建议直接使用：

```yaml
data:
  dataset_name: imagenet_vid
  train_source_path: /absolute/path/to/ILSVRC/Data/VID/train
  subset_seed: 42
  source_fraction: 0.2
  sample_fraction: 0.2

mdvsc:
  apply_masks: true
  channel_mode: identity
```

这就是当前最推荐的“先取 20% 数据、mask 打开、先做结构逼近”的起步方式。

对应的优化默认值是：

```yaml
optimization:
  reconstruction_pretrain_epochs: 3
  reconstruction_pretrain_lr: 3.0e-4
  mdvsc_bootstrap_epochs: 3
  mdvsc_bootstrap_lr: 1.0e-4
  epochs: 10
  lr: 1.0e-4

loss:
  recon_l1_weight: 1.0
  recon_mse_weight: 0.25
  recon_ssim_weight: 0.25
```

## 八、怎么启动训练

最推荐命令：

```bash
python scripts/download_rtdetr_weights.py

python scripts/train_mdvsc_stage1.py \
  --config configs/mdvsc_stage1_imagenet_vid_subset.yaml \
  --data /absolute/path/to/ILSVRC/Data/VID/train
```

如果你想换输出目录：

```bash
python scripts/train_mdvsc_stage1.py \
  --config configs/mdvsc_stage1_imagenet_vid_subset.yaml \
  --data /absolute/path/to/ILSVRC/Data/VID/train \
  --output outputs/mdvsc_stage1_imagenet_vid_run01
```

## 九、训练过程中会产出什么

输出目录里会有这些文件：

- `resolved_config.json`：本次训练实际使用的配置。
- `dataset_info.json`：数据集摘要。
- `metrics.jsonl`：每个 epoch 的训练和验证指标。
- `latest.pt`：最近一次 checkpoint。
- `best.pt`：当前最优 checkpoint。
- `epoch_XXX.pt`：按周期保存的 checkpoint。
- `final_summary.json`：训练结束摘要。

如果 `output.save_visualizations=true`，还会额外生成：

- `visualizations/train/epoch_XXX_reconstruction.png`
- `visualizations/train/epoch_XXX_feature_maps.png`
- `visualizations/train/epoch_XXX_masks.png`
- 如果有验证集，对应的 `visualizations/val/...`

## 十、这些可视化图分别表示什么

### 1. reconstruction 图

`epoch_XXX_reconstruction.png` 里：

- 第一行是原始帧；
- 第二行是重建帧。

这张图主要看重建头是否学到了合理的帧级恢复能力。

在 reconstruction head 预训练阶段，这张图最直接；如果这个阶段结束时仍然明显模糊，优先怀疑 reconstruction head 容量或重建损失，而不是 MDVSC。

如果切到 MDVSC bootstrap 后重建图没有继续提升，这是正常现象，因为这一阶段默认不再对重建图施加直接监督；它的任务是先把压缩恢复后的 projected backbone 特征对齐 teacher 特征。

如果到 joint training 后重建图仍然明显变差，则往往表示检测一致性和重建目标之间还在抢占容量，或者当前切点特征对重建本身仍然不够友好。

### 2. feature_maps 图

`epoch_XXX_feature_maps.png` 里每个 level 会显示三张图：

- teacher 特征均值热图；
- restored 特征均值热图；
- 两者绝对误差热图。

这张图主要看语义特征恢复得是否接近 teacher。

### 3. masks 图

`epoch_XXX_masks.png` 里会显示：

- 公共分支通道 mask；
- 个体分支空间 mask。

因为你当前已经把 mask 打开，这张图会非常重要。它能直接告诉你：

- 模型是不是真的在做结构化筛选；
- 公共分支和个体分支的保留模式是否稳定；
- mask 是否出现全开、全关或极端震荡。

## 十一、怎么把数值日志画成曲线图

训练后可以运行：

```bash
python scripts/plot_stage1_metrics.py --metrics outputs/mdvsc_stage1_imagenet_vid_subset/metrics.jsonl
```

这会生成：

- `loss_overview.png`
- `detection_consistency.png`
- `mask_activity.png`
- `learning_rate.png`

如果 `metrics.jsonl` 里包含 `phase` 字段，绘图脚本会在图中用竖虚线标出 reconstruction pretrain、MDVSC bootstrap 和 joint training 的阶段切换位置。

如何解读这些图，见 `docs/training_visualization_guide.md`。

## 十二、训练时建议重点盯哪些指标

建议优先看：

1. `feature_loss`
2. `recon_ssim_loss`
3. `detection_logit_loss`
4. `detection_box_loss`
5. `common_active_ratio`
6. `individual_active_ratio`

当前阶段最理想的现象是：

- `feature_loss` 逐步下降；
- `recon_ssim_loss` 逐步下降；
- 检测一致性损失也逐步下降；
- mask 激活比例相对稳定，而不是严重抖动；
- `epoch_XXX_masks.png` 看起来有明显结构，不是全白或全黑。

当前更合理的阶段性现象是：

- reconstruction pretrain 阶段：重建相关损失先明显下降；
- MDVSC bootstrap 阶段：feature_loss 开始下降，但重建和 detection consistency 都不进入训练目标；
- joint training 阶段：重建和 detection consistency 一起接入后，总损失短暂上跳，然后重新收敛。

当前这版 reconstruction head 是按 RT-DETR projected backbone 的 C3/C4/C5 三层结构来设计的：

- level 2 和 level 1 主要提供高层语义与上下文；
- level 0 同时走一条语义分支和一条细节分支；
- 解码器先做 top-down 多尺度融合，再逐级上采样回到图像分辨率。

如果你主要想继续提升重建质量，最值得优先试的两个结构参数是：

- `mdvsc.reconstruction_hidden_channels`
- `mdvsc.reconstruction_detail_channels`

在显存允许时，可以先尝试提升到 224 和 128；如果显存较紧，则保持 192 和 96 更稳妥。

## 十三、显存不够时怎么调

优先按这个顺序降：

1. 把 `optimization.batch_size` 调成 1。
2. 把 `data.frame_height` 和 `data.frame_width` 调小。
3. 把 `data.gop_size` 从 4 调成 2。
4. 把 `data.source_fraction` 和 `data.sample_fraction` 再调小。
5. 把 `optimization.num_workers` 调低。

## 十四、常见问题

### 1. 第一次运行很慢

通常是因为第一次下载 RT-DETR 权重。建议先单独执行一次：

```bash
python scripts/download_rtdetr_weights.py
```

这样后面正式训练时就不会再临时下载。

### 2. 服务器无法下载 RT-DETR 权重

可以在可联网机器上先执行：

```bash
python scripts/download_rtdetr_weights.py
```

然后把整个 `pretrained/rtdetr_r50vd/` 目录拷贝到服务器项目根目录下。只要配置里的 `detector.local_path` 指向这个目录，训练就可以离线启动。

### 3. 数据目录识别不到

确认：

- 传入的是帧目录根，而不是标注目录；
- ImageNet VID 的帧目录下面确实有序列子目录；
- 路径填写的是 `Data/VID/train` 这一侧，而不是别的目录。

### 4. mask 图看起来几乎全白

说明当前保留比例过高、训练还没学出明显结构，或者当前阶段更偏向“尽量不破坏特征”。先结合 `feature_loss` 和 `detection_logit_loss` 一起判断，不要只看 mask 图单独下结论。

### 5. mask 图看起来几乎全黑

说明筛选过强，或者训练不稳定，优先检查：

- 学习率是否过高；
- 数据子集是不是太小；
- 当前配置是否过早叠加了其他扰动。

## 十五、当前阶段之后该做什么

第一阶段跑稳后，建议按这个顺序往下走：

1. 保持 mask 打开，确认结构化筛选是稳定的。
2. 再把 `mdvsc.channel_mode` 切到 `awgn`。
3. 再考虑补更真实的 payload 正则和 sparse packet 表达。
4. 最后再进一步补带标注的检测监督和更完整的评测闭环。