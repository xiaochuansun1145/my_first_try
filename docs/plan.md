## Plan: RT-DETR 与 MDVSC 联合系统 Brief

目标是在绿地工程中，把 RT-DETR 在 encoder 后切开，将多尺度 feature map 交给可改造的 MDVSC 语义通信模块传输，在接收端恢复可供 decoder/head 使用的语义表示，形成“视频语义传输 + 目标检测”联合系统。首版聚焦离线实验管线，采用双目标平衡策略，同时验证检测精度、传输码率/失真与系统可训练性。

**Steps**
1. 定义端到端链路：发送端完成视频帧采样、RT-DETR backbone+encoder 前向、feature map 打包；信道侧完成压缩/信道模拟/解码；接收端完成 feature 恢复、RT-DETR decoder/head 推理与评测。
2. 明确接口与张量契约：固定 encoder 输出层数、shape、dtype、batch 组织、多尺度对齐方式，以及 MDVSC 模块的输入输出协议。此步骤阻塞后续训练与评测设计。
3. 先做最小可行闭环：单路视频输入、单一数据集、固定信道条件、单模型配置，跑通推理和离线评测；暂不追求实时性与复杂自适应策略。*优先级最高*
4. 建立联合优化方案：以检测损失为主任务，叠加语义传输相关损失（码率、重建/感知或特征一致性），并定义训练阶段是分阶段训练还是端到端微调。*依赖 2*
5. 建立评测面板：检测侧看 mAP；通信侧看码率、PSNR/SSIM/LPIPS 或特征恢复误差；系统侧看时延、显存、吞吐。*可与 4 并行推进*
6. 在 MVP 稳定后，再扩展到多信道条件、切点对比、压缩率扫描和消融实验。

**Relevant files**
- /workspaces/my_first_try/README.md — 当前仅有最小占位，可作为后续项目说明入口。
- /workspaces/my_first_try/src — 当前为空，后续适合按 detector、semantic_comm、pipeline、metrics 分层。
- /workspaces/my_first_try/configs — 当前为空，后续放模型、训练、信道与数据配置。
- /workspaces/my_first_try/scripts — 当前为空，后续放训练、推理、评测脚本入口。
- /workspaces/my_first_try/tests — 当前为空，后续补接口契约与最小集成测试。

**Verification**
1. 用固定样例视频跑通发送端 encoder 输出到接收端 decoder 输入，验证 feature map shape、数值范围与尺度对齐完全一致。
2. 在单一数据集上完成一次离线推理，输出检测结果与通信指标，确认端到端链路闭环成立。
3. 做无信道退化与有信道退化两组对照，确认系统在理想信道下接近原始 RT-DETR 基线，在退化信道下呈现可解释的精度下降。
4. 对训练/推理日志记录码率、mAP、时延和显存，确保后续里程碑可比较。

**Decisions**
- 输入切点定为 RT-DETR encoder 后，传输整个 feature map，而不是像素级视频或 decoder 输出。
- 通信侧存在一版简化实现，默认允许重构，不将现有接口视为稳定约束。
- 首版交付是离线实验管线，不包含实时 demo、在线服务或部署优化。
- 目标是检测与传输双目标平衡，不为单一指标极致优化。
- 规划范围不包含论文写作、SOTA 横向竞赛、专用硬件适配。

**Further Considerations**
1. 建议优先明确“传输哪几层 encoder feature”与“是否需要额外 feature adaptor”，否则后续模块边界会频繁返工。
2. 建议 MVP 首先固定一个公开数据集与一组典型信道参数，避免系统、数据、信道三条轴同时变化导致难以定位问题。
3. 建议先采用分阶段训练作为保守起点：先冻结检测主干验证通信闭环，再尝试端到端联合微调。