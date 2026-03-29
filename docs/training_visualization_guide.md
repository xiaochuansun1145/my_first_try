# 第一阶段训练可视化结果说明

本文解释第一阶段训练输出中的可视化结果应该怎么看，以及它们分别对应什么训练现象。

需要先说明一个事实：当前第一阶段训练器本身不会自动导出“原图/重建图/热力图”这一类图像面板。当前已经落地的训练输出主要是：

- metrics.jsonl；
- checkpoint；
- final_summary.json。

为了把这些数值结果变成可视化图，本仓库补充了一个绘图脚本：

```bash
python scripts/plot_stage1_metrics.py --metrics outputs/mdvsc_stage1/metrics.jsonl
```

运行后，默认会在 outputs/mdvsc_stage1/plots 下生成 4 张图：

- loss_overview.png
- detection_consistency.png
- mask_activity.png
- learning_rate.png

如果训练配置中打开了 output.save_visualizations，训练器本身还会在结果目录下额外保存三类逐 epoch 图像：

- reconstruction：原始帧和重建帧对比；
- feature_maps：teacher 特征、restored 特征、绝对误差热图；
- masks：公共分支通道 mask 和个体分支空间 mask。

## 一、先理解当前阶段的训练目标

第一阶段的目标不是直接做真实通信约束下的最终模型，而是先建立一个“无通信退化的任务上界”。

因此，当前图像可视化的重点不是码率，而是这三件事：

- 恢复出来的多尺度语义特征是否逐渐逼近 RT-DETR encoder 特征；
- 重建头是否逐渐学会从恢复语义特征还原视频帧；
- 恢复特征送回冻结 RT-DETR 后，检测输出是否逐渐逼近教师输出。

## 二、loss_overview.png 怎么看

这张图会展示四条最核心的训练曲线：

- total_loss
- feature_loss
- recon_l1_loss
- recon_mse_loss

### 1. total_loss

它是总损失，来自多项损失的加权和。

如果 total_loss 稳定下降，通常说明当前训练整体方向是对的。

如果 total_loss 大幅震荡，通常优先检查：

- 学习率是否偏大；
- 分辨率和 GOP 是否太大；
- 数据是否存在顺序或读取异常。

### 2. feature_loss

它衡量恢复出来的多尺度特征与 RT-DETR teacher 特征之间的差异。

这是第一阶段最关键的曲线之一，因为当前整个 MDVSC 结构首先要学会“不要破坏语义特征”。

如果 feature_loss 不下降，后面的重建和检测一致性通常也很难稳定。

### 3. recon_l1_loss 和 recon_mse_loss

这两条曲线衡量帧级重建质量。

- recon_l1_loss 更偏向整体结构和轮廓误差；
- recon_mse_loss 对局部像素差更敏感。

正常情况下，这两条曲线应该与 feature_loss 一起下降，但下降速度不一定一致。

如果 feature_loss 已经下降，而重建损失下降很慢，通常意味着：

- 重建头容量还不够；
- 恢复后的特征虽然接近 teacher，但对 RGB 还原还不够友好；
- 数据分辨率过高，导致重建头学习困难。

## 三、detection_consistency.png 怎么看

这张图展示两条检测一致性曲线：

- detection_logit_loss
- detection_box_loss

它们都不是带标注的真实检测损失，而是“student 恢复特征送回 detector 后”和“teacher 原始特征送回 detector 后”的输出差距。

### 1. detection_logit_loss

它衡量分类输出的一致性。

下降说明恢复特征保留了更多可供 decoder/head 使用的语义类别信息。

### 2. detection_box_loss

它衡量边框回归输出的一致性。

下降说明恢复特征不仅保住了语义类别信息，也保住了定位相关的信息。

如果这两条曲线下降得很慢，而 feature_loss 看起来已经不错，通常说明：

- 当前恢复特征在数值上接近 teacher，但对 decoder 更敏感的结构信息还没完全保住；
- per-level refinement 或 reconstruction 结构还不够强；
- 不同 level 的损失权重可能还需要重新分配。

## 四、mask_activity.png 怎么看

这张图展示：

- common_active_ratio
- individual_active_ratio

### 1. 如果当前配置是 apply_masks=false

这两条曲线接近 1.0 是正常现象，不是 bug。

原因是：当前第一阶段默认关闭 mask，模型不会真正筛掉公共分支或个体分支中的元素，因此统计到的激活比例就是“全部保留”。

也就是说，在上界训练阶段，这张图更多是在告诉你：

- 当前并没有进入稀疏传输训练；
- 模型是在全量 latent 上做结构逼近。

### 2. 如果你把 apply_masks 打开

这两条曲线就开始变得重要。

理想情况下：

- common_active_ratio 应该接近你配置的 common_keep_ratios 所对应的平均保留比例；
- individual_active_ratio 应该接近 individual_keep_ratios 对应的平均保留比例。

但训练期因为使用了可微的 STE/soft mask，它们不一定严格等于配置值，而是应当大致接近，并保持稳定。

如果打开 mask 后这两条曲线非常不稳定，通常意味着：

- mask 温度或训练调度需要单独设计；
- 仅靠当前损失还不足以稳定稀疏选择；
- 还缺显式码率约束或 payload 正则。

## 五、learning_rate.png 怎么看

这张图只是为了确认当前训练过程中的学习率设定是否符合预期。

如果你使用固定学习率，它应该基本是一条水平线。

它主要用于排查这类问题：

- 训练损失突然异常，是不是因为学习率被设错；
- 不同实验之间，是否真的用了同一套优化设置。

## 六、当前还没有哪些可视化

当前阶段还没有自动导出以下几类图，这些都属于后续建议补充项：

- 不同 level 的单独误差曲线；
- AWGN 条件下的 SNR 与重建/检测性能对照图。

也就是说，当前“训练可视化”主要还是数值曲线层面的可视化，而不是论文风格的图像面板。

## 七、怎么看一组训练结果是否健康

当前第一阶段，一组比较健康的训练结果通常表现为：

- total_loss、feature_loss、recon_l1_loss、recon_mse_loss 总体下降；
- detection_logit_loss、detection_box_loss 也随之下降；
- 当 apply_masks=false 时，active ratio 基本接近 1.0；
- train 和 val 曲线趋势一致，没有很早就明显分叉。

如果你后面打开 mask，那么除了上面这些条件，还要额外看：

- active ratio 是否接近目标保留比例；
- loss 是否在引入稀疏后出现剧烈恶化；
- detection consistency 是否比无 mask 上界退化得过快。

## 八、建议的使用顺序

推荐按下面顺序理解和查看图：

1. 先看 loss_overview.png，确认训练是否在正常收敛。
2. 再看 detection_consistency.png，确认恢复特征是否还保有检测语义。
3. 最后看 mask_activity.png，确认当前实验到底是不是在做稀疏传输训练。

如果你希望下一步做论文式展示，那么建议优先再补三类图：

1. 原图/重建图对比。
2. 公共/个体 mask 热力图。
3. 不同 SNR 条件下的性能对照图。