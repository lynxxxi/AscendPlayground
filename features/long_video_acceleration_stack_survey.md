# 长序列视频模型加速：蒸馏、稀疏与量化叠加综述

更新时间：2026-07-29  
范围：Wan/Wan2.x 等长序列视频扩散/流模型上的蒸馏、稀疏、量化及其组合训练。本文只基于可读取的一手论文、arXiv 页面、GitHub README/文档整理；无法稳定读取正文的中文文章已删除，不作为事实依据。

## 0. 核心结论

长序列视频模型的加速通常不能靠单个技巧解决。Wan 这类 Video DiT（视频扩散 Transformer）模型同时受三类成本限制：

1. **采样步数成本**：diffusion/flow 模型要反复 denoise（去噪），40-50 步很常见；few-step distillation（少步蒸馏）直接把步数压到 4、3、2 甚至 1 步，是最大收益来源。
2. **时空 attention 成本**：视频 token 是 `(time, height, width)` 三维展开，full attention（全注意力）复杂度近似 `O(N^2)`；长视频帧数增加后，attention 成本和显存会迅速膨胀。
3. **系统与存储成本**：长视频 AR（autoregressive，自回归）或流式生成还会遇到 KV cache（键值缓存）、VAE decoding（视频解码）、sequence parallel（序列并行）、offload（显存/内存换入换出）等瓶颈。

当前值得关注的组合路线：

- **蒸馏 + 稀疏**：FastVideo/FastWan 用 VSA（Video Sparse Attention，视频稀疏注意力）和 DMD2（Distribution Matching Distillation v2，分布匹配蒸馏第二版）共同训练；USV（Unified Sparsification for Video，视频统一稀疏化）在此基础上叠加 token merging（token 合并）和动态预算。
- **稀疏 + QAT**：SLA2（Sparse-Linear Attention v2，稀疏-线性注意力第二版）把稀疏路由改成可学习，并对 sparse attention branch（稀疏注意力分支）做 low-bit QAT（低比特量化感知训练）。
- **蒸馏 + 量化 + 系统并行**：LongLive-2.0 用 NVFP4（NVIDIA 4-bit floating point，NVIDIA 4 位浮点）贯穿 AR 训练、DMD LoRA 蒸馏和 W4A4 推理；TurboDiffusion 用 SageAttention/SLA + rCM（score-regularized consistency model，分数正则一致性模型）+ W8A8（8 位权重/8 位激活）追求单卡极限速度；LightX2V 偏部署框架，整合 4-step distilled models、FP8/NVFP4、offload、cache、多硬件后端。

## 1. 总览表

| 工作/项目 | 对象模型 | 核心技术 | 是否训练 | QAT/量化 | 蒸馏 | 稀疏 | 速度收益口径 | 质量/精度指标 | 代码/模型 |
|---|---|---|---:|---|---|---|---|---|---|
| SLA2 | Wan2.1-1.3B-480P、Wan2.1-14B-720P | 可学习 sparse-linear router（稀疏-线性路由器）、learnable mixing ratio（可学习混合比例）、low-bit sparse attention QAT | 是 | sparse attention branch QAT；forward low-bit，backward FP16 | 否 | 是，最高报告 97% attention sparsity | attention runtime speedup 最高 18.6x；FLOPs 同表报告 | VBench 子项：IQ/OC/AQ/MS/SC；Vision Reward；FLOPs | 论文：[arXiv:2602.12675](https://arxiv.org/abs/2602.12675) |
| Attn-QAT | Wan2.1-1.3B/14B；另含 LLM | 4-bit attention QAT；修正 FlashAttention backward 中的精度不一致 | 是 | NVFP4/FP4 attention；非 attention 仍高精度 | 否 | 否 | RTX 5090 上 attention kernel 相对 SageAttention3 约 1.1x-1.5x | VBench 全子项、盲测人评、kernel throughput | 论文：[arXiv:2603.00040](https://arxiv.org/abs/2603.00040) |
| LongLive-2.0 | Wan2.2-TI2V-5B AR long-video | Balanced SP（均衡序列并行）、NVFP4 training/inference、NVFP4 KV cache、DMD LoRA、异步 VAE decoding | 是 | W4A4 NVFP4；KV cache NVFP4；LoRA-only DMD | 是 | 使用 sliding-window/sink 局部注意力，不是主打 attention pruning | training 最高 2.15x；inference 最高 1.84x；45.7 FPS | VBench、VBench-Long、FPS、E2E latency、显存/通信 | 论文：[arXiv:2605.18739](https://arxiv.org/abs/2605.18739) |
| USV | Wan2.1-1.3B-480P，81 frames | attention sparsity + token merging + step sparsity/distillation 统一策略 | 是 | 未主打量化 | 是，继承 FastVideo DMD2 sparse distillation | 是，attention/token/step 三维 | Wan baseline 上 E2E 22.7x，DiT denoising 83.3x | VBench Total/Quality/Semantic；单 H20 wall-clock | 论文：[arXiv HTML 2512.05754v1](https://arxiv.org/html/2512.05754v1) |
| FastVideo / FastWan | Wan2.1/Wan2.2 等 Video DiT | DMD2、VSA、Sparse Distillation、FSDP2、sequence parallel、activation checkpoint | 是 | 有 QAD/FP8 等模型集合，但核心 FastWan 是 VSA + DMD2 | 是 | 是，VSA | README 称 sparse distillation 可达 >50x denoising speedup；USV 表中 FastWan E2E 20.3x、DiT 73.0x | 依 recipe/论文，多用 VBench/latency | 项目：[GitHub](https://github.com/hao-ai-lab/fastvideo)，文档：[FastVideo Docs](https://hao-ai-lab.github.io/FastVideo/) |
| TurboDiffusion | Wan2.2-I2V-14B、Wan2.1-T2V-1.3B/14B | SageAttention、SLA/SageSLA、rCM、W8A8 linear quant、FastNorm/engineering | 部分训练，部分推理优化 | W8A8 linear quant；SageAttention low-bit attention | 是，rCM timestep distillation | 是，SLA/SageSLA | README 报告单 RTX 5090 上 100-200x；Wan2.1-1.3B 184s -> 1.9s 等 | README 主要报告 E2E diffusion latency；论文称 comparable quality | 项目：[GitHub](https://github.com/thu-ml/TurboDiffusion)，论文：[arXiv:2512.16093](https://arxiv.org/abs/2512.16093) |
| LightX2V | Wan2.1/Wan2.2、LTX、HunyuanVideo 等 | 轻量推理框架、4-step distilled models/LoRA、FP8/NVFP4、offload、cache、多硬件 | 训练与推理均覆盖，偏部署 | w8a8-int8、w8a8-fp8、w4a4-nvfp4 | 是，4-step CFG/step distill | 支持 sparse/NVFP4 sparse 相关模型 | README 报告 Wan2.2 14B NVFP4 sparse 单 RTX 5090 >50x；Wan2.1-I2V-14B step time 2.0x-2.1x | 主要为框架性能表和模型发布说明 | 项目：[GitHub](https://github.com/modeltc/lightx2v) |

## 2. 术语速查

| 英文术语 | 中文意思 | 在本文中的作用 |
|---|---|---|
| DiT, Diffusion Transformer | 扩散 Transformer | 视频扩散模型的主干网络，瓶颈通常在 attention 和 MLP |
| T2V / I2V / TI2V | 文生视频 / 图生视频 / 文图生视频 | Wan/Turbo/LightX2V 的常见任务形态 |
| Attention | 注意力 | 让每个 token 根据 Q/K/V 聚合其他 token 信息 |
| Full Attention | 全注意力 | 所有 token pair 都计算 attention，质量稳但成本高 |
| Sparse Attention | 稀疏注意力 | 只计算部分 token pair，降低 attention 复杂度 |
| Linear Attention | 线性注意力 | 用核函数/特征映射把 attention 近似成线性复杂度 |
| Router | 路由器 | 决定哪些 token pair 进入稀疏分支或线性分支 |
| Token Merging | Token 合并 | 把相似 token 临时合并计算，之后再还原 |
| Step Distillation | 步数蒸馏 | 把多步 teacher 的生成能力压缩到少步 student |
| DMD / DMD2 | 分布匹配蒸馏 / 第二版 | 用 real/fake score network 让 student 分布接近 teacher/真实分布 |
| rCM | 分数正则连续时间一致性模型 | TurboDiffusion 用于 few-step timestep distillation |
| QAT | 量化感知训练 | 训练时模拟或使用低比特，让模型适配量化误差 |
| PTQ | 训练后量化 | 训练后直接量化，成本低但极低比特下更容易掉点 |
| W8A8 / W4A4 | 8 位权重 8 位激活 / 4 位权重 4 位激活 | 表示 weight/activation 的量化位宽 |
| NVFP4 / FP4 / FP8 | NVIDIA 4 位浮点 / 4 位浮点 / 8 位浮点 | 低比特浮点格式，Blackwell 对 NVFP4 更友好 |
| KV Cache | 键值缓存 | 长视频 AR/流式生成中保存历史 K/V，减少重复计算但占显存 |
| SP, Sequence Parallel | 序列并行 | 把长序列 token 切给不同 GPU/rank |
| CFG | 分类器无关引导 | 扩散模型常用的 prompt 引导技巧，蒸馏后可尝试去 CFG |
| LoRA | 低秩适配 | 冻结主干，只训练低秩增量，适合 few-step/量化蒸馏 |
| E2E Latency | 端到端延迟 | 完整生成耗时；注意有些论文排除了 text encoder 或 VAE |
| VBench | 视频生成评测基准 | 常用质量/语义/运动/一致性评测 |

## 3. 稀疏方法原理

### 3.1 为什么视频模型特别需要稀疏

图像模型的 token 数主要来自空间分辨率；视频模型还多了时间维。若视频 latent token 数是 `N = T * H * W`，full attention 的核心矩阵是 `N x N`。帧数从 16 增到 81、121 或更长时，attention 的算力、显存、中间激活都会迅速增加。

但视频 token 存在大量冗余：

- 相邻帧变化通常较小，同一主体在时间上连续。
- 局部空间区域高度相关，例如背景、天空、墙面。
- 不同 timestep 的 denoising 难度不同，早期和后期对全局信息的需求不同。
- 不同 layer 的 attention entropy（注意力熵）不同，有些层本来就只关注少量 token。

稀疏算法本质上是在回答三个问题：

1. 哪些 token pair 真的需要互相看？
2. 哪些 token 可以临时合并或跳过？
3. 哪些 timestep/layer 可以用更激进的预算？

### 3.2 VSA：Video Sparse Attention，视频稀疏注意力

VSA 的形式可以写成：

```text
Attention(Q, K, V, M) = softmax(QK^T / sqrt(d) + log(M)) V
```

其中 `M` 是 binary mask（二值掩码）。`M_ij = 1` 表示第 `i` 个 query token 可以看第 `j` 个 key token；`M_ij = 0` 表示屏蔽。实际实现不会真的构造完整 `N x N` 后再乘 0，而是把 mask 映射成 block-sparse kernel（块稀疏算子），只算保留的块。

VSA 的关键不是公式，而是 mask 从哪里来：

- FastVideo/FastWan 侧的 VSA 通常是训练得到或搜索得到的 layer-wise sparsity pattern（逐层稀疏模式）。
- 推理时 student 模型固定使用这些 sparse attention block。
- 因为 student 在训练时已经见过 sparse path，它比“训练 dense、推理硬剪枝”更稳。

适合和蒸馏叠加的原因：DMD2 sparse distillation 里，student 本来就要学习用更少 step 接近 teacher。如果同时让 student 在训练中使用 VSA，它会一起适配“少步 + 少 attention pair”的误差，而不是推理时突然被剪。

### 3.3 SLA：Sparse-Linear Attention，稀疏-线性注意力

SLA 的想法是把 full attention 拆成两类信息：

- **少数重要 token pair**：用 sparse attention 精确计算。
- **大量弱相关 token pair**：用 linear attention 近似补偿。

直观理解：不是所有 token pair 都值得精算。对关键交互，例如主体和动作区域，保留稀疏精确 attention；对背景或弱相关区域，用低成本线性近似提供全局背景信息。

典型流程：

1. 对 Q/K 做 pooling（池化），得到更短的 compressed Q/K。
2. 用 compressed Q/K 估计每个 block/token pair 的重要性。
3. Top-k 选出重要 pair 进入 sparse branch（稀疏分支）。
4. 其余信息由 linear branch（线性分支）补齐。
5. 最后融合两个 branch 的输出。

原始 SLA 的风险：

- Top-k 路由是 heuristic（启发式）的，不一定选中对生成质量最关键的位置。
- sparse branch 和 linear branch 的数值尺度可能不匹配。
- 线性分支既要近似全局低频信息，又要补偿 sparse branch 的误差，任务过重。

### 3.4 SLA2：可学习路由 + 可学习混合比例

SLA2 针对 SLA 的启发式路由做了两点升级：

1. **Learnable Router（可学习路由器）**  
   不直接用 pooled Q/K 做 Top-k，而是先过可学习投影：

```text
score = proj_q(pool(Q)) * proj_k(pool(K))^T
mask = TopK(score)
```

这样 router 能学到“哪些 attention block 对拟合 full attention 输出最重要”，而不是只靠原始相似度。

2. **Learnable Mixing Ratio（可学习混合比例）**  
   输出不再是简单相加，而是：

```text
O = alpha * O_sparse + (1 - alpha) * O_linear
```

`alpha` 是可学习参数。它让模型自己决定某层/某头更依赖 sparse branch 还是 linear branch。

训练分两阶段：

- **阶段 1：router 蒸馏 full attention**  
  收集原模型各层、各 timestep 的 Q/K/V 和 full attention 输出，训练 router 与 `alpha` 去拟合 full attention。因为 hard Top-k 不可导，训练时用 SoftTop-k（软 Top-k）近似。
- **阶段 2：端到端 diffusion fine-tuning**  
  把原 attention 替换为 SLA2，继续用 diffusion loss 微调。推理时回到 hard Top-k 和真实 sparse kernel。

这类方法的优势是推理路径清晰：输入 Q/K -> router -> sparse mask -> sparse branch + linear branch -> 输出。它比固定 mask 更自适应，比纯 linear attention 更能保留关键局部交互。

### 3.5 USV：三维统一稀疏

USV 把稀疏拆成三个维度：

1. **Attention Sparsity（注意力稀疏）**：少算 token pair。
2. **Token Sparsity（token 稀疏）**：少处理冗余 token。
3. **Sampling Sparsity（采样稀疏）**：少走 denoising step。

USV 的核心不是“再加一个稀疏技巧”，而是让三类稀疏在同一个预算下协同。原因是单独把 attention sparsity 提到很高，会损失 motion 和 subject consistency；单独减少 step，会损失细节和 prompt adherence；单独 token merging，容易破坏细粒度空间纹理。三者一起动态分配时，可以把最激进的稀疏用在冗余最高的位置。

token merging 的逻辑：

1. 在视频 latent 的 `(T,H,W)` 上划分 3D local block（三维局部块）。
2. 每个 block 选 destination token（目标 token），其余是 source token（源 token）。
3. 用 attention key 的 head-averaged descriptor（多头平均描述向量）计算 source 到 destination 的 cosine similarity（余弦相似度）。
4. 选 top-r 最相似 source 合并到 destination。
5. attention/MLP 后 exact unmerge（精确还原），恢复 dense grid，避免输出 token 位置错乱。

动态预算的逻辑：

- 计算 layer/timestep 的 attention entropy（注意力熵）。
- 低 entropy 表示 attention 很集中，冗余高，可以提高稀疏率。
- 高 entropy 表示模型正在分散关注多个区域，可能正在处理复杂运动或语义，需要保护计算预算。

所以 USV 更像一个 policy（策略）系统：根据 timestep、layer、input 的状态决定本次该省多少 attention、该合并多少 token、该走哪些 step。

### 3.6 Sliding Tile / Window / Sink Attention

长视频生成常用局部窗口或滑动窗口：

- **Sliding Window Attention（滑动窗口注意力）**：每个 token 只看附近时间/空间窗口，复杂度从全局二次下降到局部线性。
- **Sink Token（汇聚 token）**：保留少量全局 token 作为跨窗口信息通道，避免完全局部化导致长程依赖断裂。
- **Sliding Tile Attention（滑动分块注意力）**：按 tile 处理局部 attention，便于 kernel 和显存优化。

这类方法更偏长视频系统结构。它的核心权衡是：窗口越小越快，但长程一致性越难；sink/global token 越多越稳，但速度收益下降。

## 4. 量化与 QAT 原理

### 4.1 量化到底在省什么

量化把 FP16/BF16 的权重或激活映射到更低比特格式。常见目标：

- **省显存**：weight、activation、KV cache 更小。
- **省带宽**：GPU memory load/store 更少。
- **用低比特 Tensor Core**：Blackwell 等硬件对 FP4/NVFP4 有更高吞吐。

但量化不是免费午餐。视频扩散模型对低比特很敏感，尤其是 attention：

- `QK^T` 的 logits 对 outlier（离群值）敏感。
- softmax 会放大 logits 的相对误差。
- `P @ V` 中的 probability `P` 和 value `V` 同时量化时，误差会叠加。
- few-step student 的采样步数少，单步误差更难被后续 step 修正。

### 4.2 PTQ vs QAT vs native low-bit training

| 方法 | 中文意思 | 做法 | 优点 | 风险 |
|---|---|---|---|---|
| PTQ | 训练后量化 | 训练完后校准/直接量化 | 成本低，上线快 | FP4 attention 容易明显掉质量 |
| QAT | 量化感知训练 | 训练时模拟低比特 forward，反向传播更新模型 | 能适配量化误差 | 训练成本更高，kernel/fake quant 要写对 |
| Native low-bit training | 原生低比特训练 | forward/backward 主要 GEMM 真实低比特执行 | 训练也加速 | 数值稳定和系统实现最难 |

### 4.3 SLA2 的 sparse attention QAT

SLA2 只在 sparse branch 做低比特 QAT，而不是全模型激进量化：

1. Q/K 量化后计算 sparse logits。
2. softmax 后把 probability `P` 和 value `V` 也量化。
3. 得到低比特 sparse branch 输出 `O_s`。
4. backward 保持 FP16，用高精度 Q/K/V 和 forward output 算梯度。

它的工程意义是降低风险：linear branch、MLP、norm、diffusion 主干仍保守，先把最重的 sparse attention kernel 量化。论文 ablation 指出，无 QAT 直接量化会造成视频质量下降；加入 QAT 后质量更稳。

### 4.4 Attn-QAT 的核心：让 backward 也尊重低比特 forward

FlashAttention 类 fused attention 为了省显存，backward 通常会重算 `P = softmax(QK^T)`，而不是保存完整 `P`。普通 QAT 只在 forward 插 fake quant，backward 重算时可能用更高精度 `P`，这会产生训练/推理不一致：

```text
forward:   P_low = quant(softmax(quant(Q) quant(K)^T))
backward:  P_high = softmax(Q K^T)   # 如果这样重算，梯度就对应错了
```

Attn-QAT 的修正：

- backward 重算 `P` 时也 fake-quantize，确保梯度对应低比特 forward。
- forward 额外保存 high-precision output `O'` 服务 softmax backward，修复 `P^T dP = dO^T O` 里的精度假设。

这解释了为什么 naive FP4 attention 或 SageAttention3 PTQ 会掉 VBench，而 Attn-QAT 能恢复到接近 BF16 attention 的水平：它不是只换了格式，而是把训练梯度也改成“知道低比特误差”的梯度。

### 4.5 LongLive-2.0 的 NVFP4 全链路

LongLive-2.0 的量化不是单个 kernel 优化，而是训练、蒸馏、推理、缓存都低比特化：

- **AR training（自回归训练）**：linear layers 用 NVFP4；weights 使用 2D block scaling（二维块缩放），activations/gradients 使用 1D block scaling（一维块缩放）；norm、optimizer、reduction 等敏感部分保持高精度。
- **DMD LoRA distillation（DMD LoRA 蒸馏）**：teacher/student 的 Real-Score、Fake-Score、Generator 都对齐到 W4A4 NVFP4 环境；冻结量化主干，只训练 LoRA。
- **KV cache quantization（KV 缓存量化）**：按 frame chunk 把 K/V 压到 NVFP4，降低长上下文缓存显存。
- **通信优化**：在 SP/TP 场景中，低比特激活和 KV 能减少跨卡通信量。

它对 Blackwell 友好，但如果目标是 Ascend 或其他硬件，需要重新评估低比特格式、kernel 支持和通信路径。

## 5. 蒸馏方法原理

### 5.1 为什么视频扩散需要蒸馏

标准 diffusion/flow 采样是迭代过程。每一步都调用一次 DiT。若 50 steps，每次生成就要跑 50 次主干；即使 attention/MLP 单步很快，端到端仍慢。蒸馏的目标是训练一个 student（学生模型）用更少 step 近似 teacher（教师模型）的生成分布。

蒸馏不是简单“teacher 输出什么，student 就回归什么”。视频生成里直接回归容易导致：

- teacher 轨迹很长，student 少步轨迹不一定逐点对齐。
- 同一个 prompt 有多种合理视频，逐点 MSE 会压低多样性。
- 少步模型每步跨度大，噪声尺度变化更剧烈。

所以 DMD/DMD2、rCM、consistency distillation 等更关注“生成分布接近”和“跨 timestep 一致”。

### 5.2 DMD / DMD2：Distribution Matching Distillation，分布匹配蒸馏

DMD 的目标是让 student 生成分布靠近真实/teacher 分布。它通常引入两个 score network（分数网络）：

- **Real Score Network（真实分数网络）**：代表 teacher 或真实数据分布的 score，常冻结。
- **Fake Score Network（伪样本分数网络）**：跟踪 student 当前生成分布，可训练。

训练直觉：

1. student 从噪声/中间状态生成样本。
2. 把生成样本重新加噪到某个 timestep。
3. real score 告诉模型“真实/teacher 分布希望往哪走”。
4. fake score 告诉模型“student 当前分布往哪走”。
5. 两者差值形成 distribution matching gradient（分布匹配梯度），推动 student 分布靠近 teacher。

DMD2 改进 DMD 的稳定性与训练效率，更适合大规模 text-to-image/video few-step distillation。FastVideo/FastWan 和 USV 采用的 sparse distillation 就是 DMD2-style：student 同时是 few-step 和 sparse attention，训练时直接适配这个轻量路径。

### 5.3 rCM：Score-Regularized Continuous-Time Consistency Model，分数正则连续时间一致性模型

Consistency model（一致性模型）的核心是：不同噪声时间点沿着同一生成轨迹，应该映射到一致的干净样本或一致表征。rCM 加入 score regularization（分数正则），让少步生成不仅满足一致性，还不要偏离扩散 teacher 的 score field（分数场）。

TurboDiffusion 使用 rCM 做 timestep distillation，把 Wan 的多步采样压到 1-4 steps。直观流程：

1. teacher 或原模型提供多步轨迹/score。
2. student 学习从较大噪声尺度直接跳到较小噪声尺度。
3. consistency loss 约束不同 timestep 的预测一致。
4. score regularization 防止 student 只学到平滑但失真的解。

它适合极致少步推理，但对训练数据分布、prompt 长度、sigma schedule（噪声尺度调度）敏感。

### 5.4 Step Distillation 与 CFG Distillation

**Step distillation（步数蒸馏）**把 `50 -> 8 -> 4 -> 2` steps 作为主要目标。常见做法：

- teacher 用 dense full-step 生成轨迹。
- student 学少数关键 timestep 的跳跃。
- 用 diffusion loss、consistency loss、distribution matching loss 或它们的组合训练。

**CFG distillation（分类器无关引导蒸馏）**则把 classifier-free guidance 的效果蒸进 student。原推理可能要分别跑 conditional/unconditional 两支再组合；蒸馏后 student 可在少步甚至 no-CFG 下接近高 CFG 质量，减少推理开销。

LightX2V 发布的 4-step distilled models/LoRA 多属于工程可用的 step/CFG distillation 组合。

### 5.5 LoRA few-step distillation

LoRA（低秩适配）的好处是只训练少量低秩参数：

- 主干 backbone 可冻结，训练便宜。
- 如果主干已量化，LoRA 可作为少步能力的增量模块。
- 多个 LoRA 可按任务或步数切换。

LongLive-2.0 的 DMD few-step distillation 采用 LoRA-only 更新：在 W4A4 NVFP4 环境下冻结 backbone，只训练 LoRA，把 AR model 的 denoising 降到 2 steps。这样能避免全量更新破坏长视频 AR 能力，也能降低低比特训练的不稳定。

## 6. 长视频特有算法与系统设计

### 6.1 长视频不只是“帧数更多”

长视频生成会遇到短视频没有或不明显的问题：

- **Long-range temporal consistency（长程时间一致性）**：人物、场景、物体身份要跨几十秒保持一致。
- **Multi-shot coherence（多镜头一致性）**：镜头切换后主体和语义仍要连贯。
- **KV cache growth（KV 缓存增长）**：AR 生成会保存历史 K/V，显存随历史增长。
- **Training parallelism（训练并行）**：单卡放不下长序列，需要 sequence parallel 或 context parallel。
- **VAE bottleneck（VAE 瓶颈）**：长视频编码/解码可能超过 DiT 之外的系统成本。

因此长视频加速要同时看算法和系统。

### 6.2 AR long-video：自回归长视频

AR long-video 的思想是按 chunk（片段）逐段生成视频。每次生成新的 temporal chunk 时，模型条件在历史 chunk 上，从而扩展到更长时长。

关键技术：

- **Sliding-window context（滑动窗口上下文）**：只保留最近一段历史，控制成本。
- **Sink tokens（汇聚 token）**：保留少量全局历史摘要，缓解滑窗导致的远程遗忘。
- **KV cache（键值缓存）**：复用历史 attention 的 K/V，减少重复算历史。
- **Streaming VAE decoding（流式 VAE 解码）**：边生成边解码，降低端到端等待时间。

LongLive-2.0 的贡献就是把 AR long-video 训练、NVFP4 低比特、Balanced SP、DMD LoRA、KV cache quantization、异步 VAE decoding 放在同一个系统里。

### 6.3 Balanced SP：均衡序列并行

普通 sequence parallel 会把长序列 token 切给不同 rank。问题是 long-video training 中，不同 chunk 的 clean/noisy token、loss-bearing token、VAE 编码成本可能不均衡，导致有的 GPU 很忙，有的 GPU 等通信。

Balanced SP 的目标：

- 每个 rank 持有同一 temporal chunk 中配对的 clean/noisy stream。
- 均衡每个 rank 上真正参与 loss 的 tokens。
- VAE encoding 也按 chunk sharding，避免 DiT 均衡但 VAE 不均衡。
- 降低 all-to-all/collective communication 的等待。

这对长视频重要，因为长序列训练的瓶颈常常不是单个矩阵乘，而是“显存 + 通信 + 数据布局”。

### 6.4 KV cache quantization：键值缓存量化

AR 或流式视频生成中，历史 K/V 会被后续 chunk 重复使用。KV cache 如果保持 BF16，长视频越长显存越高。KV cache quantization 把 K/V 存成 FP8、FP4、NVFP4 或 INT8：

- 优点：显存下降，跨卡传输下降。
- 风险：历史上下文信息被量化后，长程一致性可能下降。
- 评测重点：不要只看短视频 VBench，要看 VBench-Long、subject consistency、scene consistency、人评。

LongLive-2.0 报告 NVFP4 KV cache 约 3.6x compression，dequant overhead 低于 2%。

### 6.5 Offload / Cache / Parallel 的工程边界

LightX2V 的价值在于把很多部署优化放进统一框架：

- **Block offload（块级换出）**：按 transformer block 把不用的权重/激活移到 CPU/其他内存。
- **Phase offload（阶段换出）**：按 text encoder、DiT、VAE 等阶段调度显存。
- **Cache（缓存）**：复用中间结果或历史上下文。
- **CFG parallelism（CFG 并行）**：conditional/unconditional 分支并行。
- **多硬件后端**：NVIDIA、AMD、Ascend、Cambricon、MetaX 等。

这些优化不一定改变模型精度，但会影响实际可部署性。内部实验报告应区分“模型算法加速”和“系统部署加速”。

## 7. 叠加训练怎么做

### 7.1 FastVideo/FastWan：VSA + DMD2 sparse distillation

训练图：

```text
dense long-step teacher
        |
        | 生成 teacher 分布/score
        v
few-step sparse student + VSA
        |
        | DMD2 real/fake score distribution matching
        v
sparse-distilled few-step generator
```

具体含义：

1. teacher 保持 full attention 和较多 denoising steps，提供高质量分布参考。
2. student 从训练开始就使用 VSA sparse attention，并只走 reduced step set。
3. real score network 固定，fake score network 跟踪 student 分布。
4. DMD2 loss 推动 student 的生成分布靠近 teacher/real distribution。
5. 推理时只保留 few-step sparse student，不需要 teacher。

优势是训练/推理路径一致。风险是如果同时把 step 减太多、sparsity 提太高，student 容易在运动细节和主体一致性上掉点。

### 7.2 USV：VSA + DMD2 + token merging + entropy-aware budget

USV 以 FastVideo sparse-distilled generator 为起点，再加入三维预算：

1. warm-up：固定 sparsity，先稳定 sparse student。
2. token merging：在 3D local block 中合并高相似 token。
3. entropy-aware policy：按 layer/timestep/input 的 attention entropy 动态分配 attention/token sparsity。
4. loss：distillation loss + budget loss + entropy regularization + temporal consistency regularization。
5. inference：只保留 sparse student 和 policy，不再需要 dense teacher。

USV 的关键是“动态”。同一 prompt、同一视频的不同 timestep/layer，冗余程度不同；固定 mask 和固定 step schedule 不一定最优。

### 7.3 SLA2：learnable sparse-linear routing + QAT

SLA2 的叠加不压 step，而是改 attention 内部结构并做 low-bit QAT：

1. 收集 full attention Q/K/V 和输出。
2. 训练 learnable router 与 mixing ratio，拟合 full attention。
3. 替换原模型 attention 为 SLA2。
4. sparse branch 使用 low-bit forward 做 QAT。
5. diffusion loss 端到端 fine-tune。
6. 推理使用 hard Top-k router + sparse kernel + low-bit sparse branch。

如果要和 DMD2/rCM 继续叠加，建议顺序是：

1. dense few-step student 先稳定；
2. 插入 SLA2/VSA 做 sparse fine-tune；
3. 最后启用 attention QAT；
4. 每一步都保留 dense/fp16 对照 ablation。

### 7.4 LongLive-2.0：Balanced SP + NVFP4 + AR tuning + DMD LoRA

LongLive-2.0 的叠加更偏长视频全系统：

1. 把 bidirectional diffusion model fine-tune 成 long, multi-shot, interactive AR model。
2. 用 Balanced SP 支撑 16s/32s/64s 长序列训练。
3. AR training 中主要 GEMM 使用 NVFP4，敏感路径保高精度。
4. 在 AR-trained model 上做 one-stage DMD distillation，只训练 LoRA。
5. 推理阶段使用 W4A4 generator、NVFP4 KV cache、SP inference、async streaming VAE decoding。

这条路线最适合目标是“长视频、交互式、流式、高 FPS”的场景；如果只是 5s/81 frames 的短视频，收益不一定完全来自 AR 系统。

### 7.5 TurboDiffusion：SageAttention/SLA + rCM + W8A8 + 工程优化

TurboDiffusion 的默认组合：

- attention：SageAttention、SLA、SageSLA。
- timestep：rCM few-step distillation。
- linear：W8A8 quantization。
- 工程：FastNorm、TileLang W8A8、interactive serve、quantized checkpoints。

README 的参数暴露显示其推荐路径：`--num_steps 4`、`--attention_type sagesla`、`--sla_topk 0.1`，使用量化 checkpoint 时加 `--quant_linear`。它的结果多以 E2E diffusion latency 报告，需要注意是否包含 text encoder 和 VAE。

### 7.6 LightX2V：4-step distill + quant/offload/cache 推理框架

LightX2V 更像部署侧工具箱：

- 提供 Wan2.1/Wan2.2 4-step distilled models/LoRA。
- 提供 FP8/NVFP4/INT8 量化路径。
- 支持 block/phase offload，目标是消费级 GPU 也能跑 14B 480P/720P。
- 支持 Sage Attention、Flash Attention、Radial Attention、q8-kernel、vLLM 等算子/后端。

对我们工作的意义：可以作为 deployment checklist。训练期必须参与的优化包括蒸馏、QAT、训练期稀疏；推理期可插拔的包括 offload、cache、backend 切换、部分 PTQ。

## 8. 实验与评测

### 8.1 质量指标

| 指标 | 中文意思 | 作用 | 注意事项 |
|---|---|---|---|
| VBench Total | VBench 综合分 | 综合视频生成质量 | 不同版本/子项聚合可能不同 |
| VBench Quality | 视觉质量分 | 画质、时序、运动等 | 稀疏和量化后容易掉 |
| VBench Semantic | 语义一致性分 | prompt 与视频是否匹配 | 蒸馏后 prompt adherence 需重点看 |
| IQ, Imaging Quality | 成像质量 | 单帧清晰度/纹理 | SLA2、Attn-QAT 常用 |
| AQ, Aesthetic Quality | 审美质量 | 构图、色彩、观感 | 过饱和和模糊会影响 |
| OC, Overall Consistency | 整体一致性 | 视频结构稳定性 | 长视频尤其重要 |
| MS, Motion Smoothness | 运动平滑 | 动作是否连续 | few-step distill 容易影响 |
| SC, Subject Consistency | 主体一致性 | 主体是否漂移 | 长视频和 KV cache 量化重点 |
| Vision Reward | 视觉奖励模型 | 人类偏好 proxy | 不能完全替代人评 |
| Human Eval | 人工评测 | 主观质量 | 建议盲测、固定 prompt set |

### 8.2 效率指标

| 指标 | 中文意思 | 含义 | 最容易混淆的点 |
|---|---|---|---|
| Attention speedup | 注意力加速 | attention kernel/runtime 加速 | 不能等同端到端加速 |
| DiT denoising speedup | DiT 去噪加速 | 只统计 DiT denoising | 通常不含 text encoder/VAE |
| E2E generation latency | 端到端生成延迟 | 完整生成耗时 | 有些项目排除 text encoder 或 VAE |
| Step time | 单步耗时 | 单次 iteration/inference 时间 | 要乘 steps 才接近 denoising latency |
| FPS | 每秒帧数 | 流式长视频吞吐 | 是否含 VAE decoding 很关键 |
| FLOPs | 浮点计算量 | 理论计算量 | sparse/linear/token merge 要按真实路径算 |
| Peak memory | 峰值显存 | 最大显存占用 | 长视频常由 KV/VAE/activation 决定 |
| Communication time | 通信时间 | SP/TP/DP 跨卡通信 | 低比特激活/KV 会影响扩展性 |

### 8.3 代表性实验结果

| 工作 | 实验设置 | Baseline | 关键结果 | 评测口径 |
|---|---|---|---|---|
| SLA2 | Wan2.1-1.3B-480P、Wan2.1-14B-720P；私有 3000 个约 5s 视频；caption 用 Qwen3-VL-Flash | Full Attention/FlashAttn2、SLA、VSA、VMoBA | 最高 97% attention sparsity；报告 18.6x attention runtime speedup；97% sparsity 下仍优于多种 sparse baseline | VBench IQ/OC/AQ/MS/SC、Vision Reward、FLOPs |
| SLA2 ablation | 同上 | w/o QAT、Topk-router、不同 sparsity | w/o QAT 量化推理质量下降；learnable router 优于 Top-k router；低 sparsity 质量更好但速度收益下降 | VBench 子项 + VR |
| Attn-QAT | Wan2.1-1.3B 480P 81K examples；Wan2.1-14B 720P 13K examples；prompt augmentation 用 Qwen2.5-3B-Instruct | BF16、FP4 no training、SageAttention3 | Wan2.1-1.3B overall：BF16 0.8267，FP4 0.7785，SageAttention3 0.7834，Attn-QAT 0.8252；Wan2.1-14B overall：BF16 0.8335，FP4 0.7968，SageAttention3 0.8203，Attn-QAT 0.8279 | VBench all subcategories、99 prompts 盲测人评、kernel throughput |
| LongLive-2.0 training | AR long-video training，16s/32s/64s | BF16 w/o SP、BF16 SP、BF16 Balanced SP | 64s 上 BF16 w/o SP OOM，NVFP4 Balanced SP 相对 BF16 SP 约 2.1x；论文摘要称训练最高 2.15x | iteration time、memory |
| LongLive-2.0 inference | Blackwell W4A4；H100 SP inference；KV cache quant | BF16、BF16 KV cache | 摘要称 inference 最高 1.84x，LongLive-2.0-5B 达 45.7 FPS；KV cache 约 3.6x 压缩，dequant overhead <2% | FPS、E2E latency、通信时间、VBench/VBench-Long |
| USV | Wan2.1-1.3B，480p，81 frames，约 131K spatio-temporal tokens，单 H20 | Dense Wan2.1、FastWan | Dense Wan：50 steps；FastWan：80% sparsity/3 steps/E2E 20.3x/DiT 73.0x；USV：95% sparsity/3 steps/E2E 22.7x/DiT 83.3x | VBench Total/Quality/Semantic、wall-clock latency |
| TurboDiffusion | 单 RTX 5090，Wan2.2-I2V-A14B-720P、Wan2.1-T2V-1.3B/14B | Original、FastVideo | README：Wan2.1-1.3B 184s -> TurboDiffusion 1.9s；Wan2.2-I2V-A14B 4549s -> 38s；Wan2.1-14B-720P 4767s -> 24s | README 称 E2E diffusion latency，排除 text encoding 和 VAE decoding |
| LightX2V | Wan2.1-I2V-14B-480P，40 steps，81 frames | cfg / no cfg / no cfg + fp8 | H100 8 GPUs：0.75s/it -> 0.35s/it，2.1x；4090D 8 GPUs：4.75s/it -> 2.35s/it，2.0x；另发布 Wan2.2 14B NVFP4 sparse 单 RTX 5090 >50x | step time、框架 README 性能表 |

## 9. 项目源码/工程优化梳理

> 说明：本节只基于公开 GitHub README/文档/目录结构做源码侧梳理，没有 clone 或运行项目。

| 项目 | 训练支持 | 推理支持 | Attention backend | Quant backend | Distill recipe/model | 硬件支持 | API/CLI 与工程结构 |
|---|---|---|---|---|---|---|---|
| FastVideo | full finetuning、LoRA finetuning、data preprocessing、DMD2 stepwise distillation、Sparse Distillation；FSDP2、sequence parallel、selective activation checkpointing | `VideoGenerator` Python API、CLI、Dreamverse realtime app | Video Sparse Attention、Sliding Tile Attention、Sage Attention、多 attention backend | HuggingFace 上有 QAD/FP8 等模型集合，README 主线更强调 VSA/DMD2 | FastWan2.1/2.2 sparse distillation recipes；FastVideo Synthetic datasets | H100、A100、4090、Linux/Windows/MacOS；DGX Spark 安装说明 | monorepo；`docs/`、`recipes` 链接、`apps/dreamverse/`、训练/推理统一框架 |
| LightX2V | `lightx2v_train` 目录；GenRL、LoRA/蒸馏模型发布；README 主体偏推理部署 | `LightX2VPipeline`、configs、examples、Gradio/ComfyUI/Windows one-click/service deployment | Sage Attention、Flash Attention、Radial Attention、q8-kernel、sgl-kernel、vLLM | `w8a8-int8`、`w8a8-fp8`、`w4a4-nvfp4`；NVFP4 operator in `lightx2v_kernel` | Wan2.1/2.2 Distill Models/LoRAs、Wan2.1-Distill-NVFP4、4-step distilled LoRA | H100、4090D、RTX 30/40/50、AMD ROCm、Ascend 910B、Cambricon、MetaX、Intel AIPC、T-head PPU 等 | `configs/`、`examples/`、`lightx2v_kernel/`、`lightx2v_platform/`；支持 offload/caching/parallel |
| TurboDiffusion | 提供 finetuned/distilled checkpoints；论文侧包含 rCM distillation；README 以 inference 为主 | T2V/I2V scripts、interactive serve；量化/非量化 checkpoints | original/sla/sagesla；SageAttention；SpargeAttn dependency | W8A8 linear quant；`--quant_linear`；quant checkpoints | TurboWan2.1/TurboWan2.2；rCM 1-4 steps | RTX 5090/4090 用 quant checkpoint；H100 用 unquantized checkpoint | `turbodiffusion/inference/`、`turbodiffusion/serve/`、`TurboT2AV/`；参数暴露 `--num_steps`、`--sla_topk`、`--attention_type` |

## 10. 对我们 Wan 叠加工作的建议

### 10.1 推荐实验矩阵

为了知道每个 trick 的边际收益和误差叠加，建议至少跑以下矩阵。每个实验都固定 prompts、resolution、frames、seed set、hardware、scheduler，分别报告 DiT latency、E2E latency、peak memory、VBench、人工抽样。

| ID | 蒸馏 | 稀疏 | 量化/QAT | 目的 |
|---|---|---|---|---|
| A0 | 否 | 否 | 否 | dense Wan baseline，质量上限和延迟基准 |
| A1 | 是 | 否 | 否 | few-step distillation 单独收益，观察 motion/semantic 掉点 |
| A2 | 否 | 是 | 否 | sparse attention/token 单独收益，观察 flicker/subject consistency |
| A3 | 否 | 否 | PTQ/QAT | 量化单独收益，区分 W8A8、FP8、NVFP4 attention/linear/KV |
| A4 | 是 | 是 | 否 | FastWan/USV 类 sparse-distill 主线 |
| A5 | 否 | 是 | QAT | SLA2 类 sparse + low-bit attention |
| A6 | 是 | 否 | QAT | few-step student 的量化鲁棒性 |
| A7 | 是 | 是 | QAT | 三者叠加目标形态 |
| A8 | 是 | 是 | PTQ | 对比 QAT 是否必要，确认 PTQ 是否足够 |

### 10.2 推荐训练优先级

1. **先稳定 few-step 蒸馏**：把 50/40 steps 压到 4/3/2 steps 是最大收益来源。先用 dense attention 建立质量下限和调参经验。
2. **再插入训练期稀疏**：优先 VSA 或 SLA2 这种训练期显式适配的方式，不建议只在推理期 hard prune attention。
3. **最后做与推理一致的 QAT**：FP4/NVFP4 特别需要 QAT；Attn-QAT 说明 naive FP4 attention 或 PTQ 对 Wan 会掉 overall quality。
4. **如果目标硬件是 Blackwell，优先 NVFP4/W4A4**：如果目标硬件是 RTX 5090/B200/GB200，LongLive-2.0 和 Attn-QAT 的经验更相关；H100/4090 上则更现实地比较 FP8/W8A8/SageAttention/SageSLA。
5. **如果目标硬件是 Ascend，需要单独评估 kernel 可用性**：不要直接假设 NVFP4/SageAttention/Triton CUDA 路径可迁移，需确认 Ascend CANN、算子库、通信库是否支持等价低比特和 sparse kernel。
6. **长视频 AR 场景不要只看 DiT FLOPs**：KV cache、SP 通信、VAE decoding、offload 都可能成为主瓶颈。

### 10.3 建议记录的 acceptance metrics

- 质量：VBench Total/Quality/Semantic + 子项 IQ/AQ/MS/SC + 50-100 prompts 人工盲测。
- 速度：attention kernel、DiT denoising、E2E generation 分开报。
- 显存：peak memory、KV cache size、offload traffic。
- 稳定性：训练 loss、gradient norm、NaN/OOM、不同 seeds 方差。
- 退化类型：过饱和、动态不足、temporal flicker、主体漂移、prompt 忽略、细节糊化。

### 10.4 风险点

- **few-step 蒸馏 + 高 sparsity**：两者都在减少计算路径，容易牺牲细节和 motion diversity。
- **PTQ FP4 attention**：Attn-QAT 在 Wan 上显示直接 FP4 或 SageAttention3 PTQ 仍有明显质量损失，应优先 QAT。
- **训练/推理 mismatch**：SoftTop-k vs hard Top-k、fake quant vs real quant、dense teacher vs sparse student 都需要单独做 ablation。
- **指标口径混乱**：论文和 README 经常分别报告 attention speedup、DiT speedup、E2E diffusion latency、完整 E2E latency；内部汇报必须强制分栏。
- **数据分布依赖**：TurboDiffusion README 提到当前模型只在 long English prompts 上训练，非同分布 prompt 需要 augment；你们自己的中文/多语言/短 prompt 场景要额外测。

## 11. 来源

- [SLA2: Sparse-Linear Attention with Learnable Routing and QAT](https://arxiv.org/abs/2602.12675)
- [Attn-QAT: 4-Bit Attention With Quantization-Aware Training](https://arxiv.org/abs/2603.00040)
- [LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation](https://arxiv.org/abs/2605.18739)
- [USV: Unified Sparsification for Accelerating Video Diffusion Models](https://arxiv.org/html/2512.05754v1)
- [TurboDiffusion GitHub](https://github.com/thu-ml/TurboDiffusion)
- [TurboDiffusion paper](https://arxiv.org/abs/2512.16093)
- [FastVideo GitHub](https://github.com/hao-ai-lab/fastvideo)
- [FastVideo Docs](https://hao-ai-lab.github.io/FastVideo/)
- [LightX2V GitHub](https://github.com/modeltc/lightx2v)

