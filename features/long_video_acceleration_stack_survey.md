# 长序列视频模型加速：蒸馏、稀疏与量化叠加综述

更新时间：2026-07-29  
范围：Wan/Wan2.x 等长序列视频扩散/流模型上的蒸馏、稀疏、量化及其组合训练。本文只做文献与公开项目梳理，不运行项目、不下载权重、不做实际生成评测。

## 0. 结论先行

长序列视频模型的主要瓶颈来自两个方向：一是 DiT 对时空 token 做全局 attention，复杂度随 token 数近似二次增长；二是 diffusion/flow 采样需要多步迭代。当前有效路线通常不是单点优化，而是把 step distillation、sparse attention、低比特量化、KV/cache/parallel/offload 等一起叠加。

对 Wan 这类模型，比较成熟的组合范式有三类：

1. **蒸馏 + 稀疏**：FastVideo/FastWan 用 VSA sparse attention 和 DMD2 few-step distillation 共同训练；USV 在此基础上再加入 token merging 和动态预算策略。
2. **稀疏 + QAT**：SLA2 把 sparse-linear attention 的路由改成可学习，并在 sparse branch 上加入低比特 attention 的 QAT。
3. **蒸馏 + 量化 + 系统并行**：LongLive-2.0 用 NVFP4 贯穿 AR training、DMD LoRA 和 W4A4 inference；TurboDiffusion 用 SageAttention/SLA + rCM + W8A8 + 工程优化追求单卡极限速度；LightX2V 则偏部署侧，把 4-step distill、FP8/NVFP4、offload、cache、多后端封装成框架。

## 1. 总览表

| 工作/项目 | 对象模型 | 核心技术 | 是否训练 | QAT/量化 | 蒸馏 | 稀疏 | 速度收益口径 | 质量/精度指标 | 代码/模型 |
|---|---|---|---:|---|---|---|---|---|---|
| SLA2 | Wan2.1-1.3B-480P、Wan2.1-14B-720P | 可学习 sparse-linear router、learnable mixing ratio、low-bit sparse attention QAT | 是 | sparse attention branch QAT，forward low-bit，backward FP16 | 否 | 是，最高报告 97% attention sparsity | attention runtime speedup 最高 18.6x；FLOPs 同表报告 | VBench 子项：IQ/OC/AQ/MS/SC；Vision Reward；FLOPs | 论文：[arXiv:2602.12675](https://arxiv.org/abs/2602.12675) |
| Attn-QAT | Wan2.1-1.3B/14B；另含 LLM | 4-bit attention QAT，修正 FlashAttention backward 中的精度不一致 | 是 | NVFP4/FP4 attention；非 attention 仍高精度 | 否 | 否 | RTX 5090 上 attention kernel 相对 SageAttention3 约 1.1x-1.5x | VBench 全子项、盲测人评、kernel throughput | 论文：[arXiv:2603.00040](https://arxiv.org/abs/2603.00040) |
| LongLive-2.0 | Wan2.2-TI2V-5B AR long-video | Balanced SP、NVFP4 training/inference、NVFP4 KV cache、DMD LoRA、异步 VAE decoding | 是 | W4A4 NVFP4；KV cache NVFP4；LoRA-only DMD | 是 | AR sliding-window/sink 侧局部注意力，不是主打 attention pruning | training 最高 2.15x；inference 最高 1.84x；45.7 FPS | VBench、VBench-Long、FPS、E2E latency、显存/通信 | 论文：[arXiv:2605.18739](https://arxiv.org/abs/2605.18739) |
| USV | Wan2.1-1.3B-480P，81 frames | attention sparsity + token merging + step sparsity/distillation 统一策略 | 是 | 未主打量化 | 是，继承 FastVideo DMD2 sparse distillation | 是，attention/token/step 三维 | Wan baseline 上 E2E 22.7x，DiT denoising 83.3x | VBench Total/Quality/Semantic；单 H20 wall-clock | 论文：[arXiv HTML 2512.05754v1](https://arxiv.org/html/2512.05754v1) |
| FastVideo / FastWan | Wan2.1/Wan2.2 等 video DiT | DMD2、VSA、Sparse Distillation、FSDP2、sequence parallel、activation checkpoint | 是 | 有 QAD/FP8 等模型集合，但核心 FastWan 是 VSA + DMD2 | 是 | 是，VSA | README 称 sparse distillation 可达 >50x denoising speedup；USV 表中 FastWan E2E 20.3x、DiT 73.0x | 依 recipe/论文，多用 VBench/latency | 项目：[GitHub](https://github.com/hao-ai-lab/fastvideo)，文档：[FastVideo Docs](https://hao-ai-lab.github.io/FastVideo/) |
| TurboDiffusion | Wan2.2-I2V-14B、Wan2.1-T2V-1.3B/14B | SageAttention、SLA/SageSLA、rCM、W8A8 linear quant、FastNorm/engineering | 部分训练，部分推理优化 | W8A8 linear quant；SageAttention low-bit attention | 是，rCM timestep distillation | 是，SLA/SageSLA | README 报告单 RTX 5090 上 100-200x；Wan2.1-1.3B 184s -> 1.9s 等 | README 主要报告 E2E diffusion latency；论文称 comparable quality | 项目：[GitHub](https://github.com/thu-ml/TurboDiffusion)，论文：[arXiv:2512.16093](https://arxiv.org/abs/2512.16093) |
| LightX2V | Wan2.1/Wan2.2、LTX、HunyuanVideo 等 | 轻量推理框架、4-step distilled models/LoRA、FP8/NVFP4、offload、cache、多硬件 | 训练与推理均覆盖，偏部署 | w8a8-int8、w8a8-fp8、w4a4-nvfp4 | 是，4-step CFG/step distill | 支持 sparse/NVFP4 sparse 相关模型 | README 报告 Wan2.2 14B NVFP4 sparse 单 RTX 5090 >50x；Wan2.1-I2V-14B step time 2.0x-2.1x | 主要为框架性能表和模型发布说明 | 项目：[GitHub](https://github.com/modeltc/lightx2v) |
| 知乎 DMD/DMD2 解读 | DMD/DMD2 方法解读 | 分布匹配蒸馏介绍 | 不适用 | 不适用 | 是 | 否 | 未一手核验 | 未一手核验 | 链接：[知乎专栏](https://zhuanlan.zhihu.com/p/1953596250491434750)，正文未稳定抓取，仅作二手材料 |
| 微信文章 | 可能为 Wan 加速/量化/蒸馏解读 | 未一手确认 | 不适用 | 未确认 | 未确认 | 未确认 | 未一手核验 | 未一手核验 | 链接：[微信公众号](https://mp.weixin.qq.com/s/GdE5s0sSfVZWH8w5HVpm1w)，正文未稳定抓取，仅作二手材料 |

## 2. 稀疏方法分类

### 2.1 VSA：Video Sparse Attention

VSA 的基本形式是在每层 self-attention 中引入二值 mask，只允许部分 query-key pair 参与注意力计算。USV 对 FastVideo/VSA 的描述是：对第 `l` 层引入 learned binary mask `M^(l)`，masked-out 的位置通过 `log M` 进入 attention logits，推理时 FastVideo student 的 self-attention block 被替换为固定 sparsity pattern 的 VSA。来源：[USV §3.2](https://arxiv.org/html/2512.05754v1)。

要点：

- **mask 产生**：训练得到的 layer-wise binary mask，推理时固定。
- **粒度**：token-pair attention mask，通常可映射为 block/kernel 级实现。
- **训练/推理一致性**：FastVideo 的 sparse-distilled student 推理时使用固定 mask；USV 批评其“固定 step schedule + fixed masks”不够动态。
- **适合叠加**：天然可与 DMD2 few-step distillation 共同训练，形成 sparse-distill。

### 2.2 SLA：Sparse-Linear Attention

SLA 把 full attention 分解为 sparse branch 和 linear branch。原始 SLA 使用 pooled Q/K 得到 compressed attention，再按 top-k 规则把较重要位置交给 sparse branch，其余由 linear attention branch 近似。TurboDiffusion README 明确使用 SageAttention 和 SLA 做 attention acceleration。来源：[TurboDiffusion README](https://github.com/thu-ml/TurboDiffusion)、[SLA2 论文对 SLA 的回顾](https://arxiv.org/abs/2602.12675)。

要点：

- **mask 产生**：pool(Q)、pool(K) 后算 compressed score，再按 top-k 选择 sparse branch。
- **粒度**：block/token-pair，适合 FlashAttention-style sparse kernel。
- **问题**：SLA2 指出 SLA 的 sparse branch 有 row-wise scaling mismatch，且 heuristic routing 不一定是最优的 sparse/linear 分配。

### 2.3 SLA2：可学习路由 + sparse-linear mixing

SLA2 主要改了两件事：

1. 用 `proj_q(pool(Q)) * proj_k(pool(K))^T` 生成 compressed score，再 Top-k 得到 mask；`proj_q/proj_k` 是可学习投影。
2. 用 learnable ratio `alpha` 显式混合 sparse branch 和 linear branch：`O = alpha * O_s + (1 - alpha) * O_l`，避免让 linear branch 同时承担 residual compensation。

训练分两阶段：

- **Stage 1**：收集各层各 timestep 的 Q/K/V，训练 router `R` 和 `alpha` 去拟合 full attention 输出。因为 Top-k 不可导，训练时用 SoftTop-k。
- **Stage 2**：把 diffusion 模型中的 attention 替换为 SLA2，端到端 fine-tune diffusion loss；推理时回到 hard Top-k。来源：[SLA2 §5-§9](https://arxiv.org/html/2602.12675v1)。

要点：

- **mask 产生**：可学习 router，输入 Q/K，不用 V。
- **粒度**：pooled token block mask，GPU kernel 只需要 compressed mask。
- **训练/推理一致性**：Stage 1 用 SoftTop-k 解决可导，Stage 2 使用接近推理的 hard Top-k。
- **稀疏强度**：论文报告 85%、90%、95%、97% sparsity，最高 97% 下仍维持较好 VBench/VR。

### 2.4 USV：attention/token/step 三维统一稀疏

USV 的核心观点是单独优化 attention 或 step 会遇到边际收益递减，因此把三类稀疏放进一个预算控制框架：

- **attention sparsity**：哪些 token pair 能互相 attend。
- **token sparsity**：每层实际处理多少时空 token。
- **sampling sparsity**：采样轨迹走多少 denoising steps。

token merging 具体做法：

- 在 `(T,H,W)` 上切 3D local block。
- 每个 block 选一个 destination token，其余是 source token。
- 用 attention key 的 head-averaged descriptor 计算 source 到 destination 的 cosine similarity。
- 贪心选 top-r source merge 到 destination。
- attention/MLP 后用 exact unmerge 恢复 dense grid，保证输出仍对齐原时空位置。

动态策略具体做法：

- 计算 layer/timestep attention entropy。
- 低 entropy 表示注意力更集中、冗余更高，可以更激进稀疏。
- 在每个 timestep 的整体预算固定前提下，把更多稀疏额度分配给低 entropy layer，同时保护高 entropy layer。来源：[USV §3.1-§3.5](https://arxiv.org/html/2512.05754v1)。

### 2.5 SageSLA / SageAttention / sliding-tile

- **SageAttention**：低比特 attention kernel 系列，主要通过 Q/K smoothing、P 的量化策略等减小 attention PTQ 误差。Attn-QAT 将 SageAttention3 作为 FP4 PTQ baseline，并指出 QAT 可以替代一部分 outlier mitigation。
- **SageSLA**：TurboDiffusion README 中的默认 inference attention path，可理解为把 SLA 的 sparse-linear 结构与 SageAttention 类低比特 kernel 结合，暴露 `--attention_type sagesla` 和 `--sla_topk`。
- **Sliding Tile Attention**：FastVideo 文档列为 inference optimization；适合视频中的局部/滑窗 attention 复用。本文没有找到其与蒸馏/QAT 叠加训练的完整一手细节，因此只列为工程优化方向。

## 3. 量化与 QAT 分类

### 3.1 PTQ、QAT、native low-bit training 的差异

- **PTQ**：训练完成后做量化，可有校准但不更新模型参数。优点是便宜；缺点是在 FP4 这种极低比特下，attention 对 outlier 和动态范围很敏感。
- **QAT**：训练阶段显式模拟低比特 forward，用梯度更新模型以适配量化误差。适合减少训练/推理 mismatch。
- **native low-bit training**：forward/backward 的主要 GEMM 都真实走低精度，目标同时提升训练和推理效率；实现更难，需要处理梯度、reduction、norm、optimizer 等数值稳定问题。

### 3.2 SLA2 的 sparse + low-bit attention QAT

SLA2 只把低比特 attention 用在 sparse branch `O_s` 上：

- forward：量化 Q/K 后计算 sparse attention logits，softmax 后再量化 P/V，最后 dequant 得到 `O_s`。
- backward：保持 FP16-only，用原 FP16 Q/K/V 和 forward output 计算梯度。
- kernel：基于 FlashAttention 思路，避免显式构造完整 `S = QK^T`，只对 mask 命中的 block 做计算。

这条路线的优点是工程上比较稳：只在 sparse branch 引入低比特，linear branch 和 diffusion 主训练仍可保持相对保守。论文 ablation 显示，无 QAT 直接量化推理会造成视频质量下降；低比特量化约带来 1.3x kernel speedup。来源：[SLA2 §5、§9.4](https://arxiv.org/html/2602.12675v1)。

### 3.3 Attn-QAT 的 4-bit attention QAT

Attn-QAT 针对 FlashAttention-style fused operator 的问题更深入。普通 QAT 只在 matmul 输入上 fake quant，但 FlashAttention backward 会重算 attention probability `P`，如果 backward 重算的 `P` 精度和 forward 不一致，就会出现梯度不稳定。

Attn-QAT 的两个关键修正：

1. **Backward 重算 P 时也 fake-quantize**：保证 backward 梯度对应 forward 中实际用过的低精度激活。
2. **Forward 额外保留 high-precision O' 只服务 backward**：修复 FlashAttention softmax backward 中依赖 `P^T dP = dO^T O` 的精度假设。

实现上，训练 kernel 基于 Triton reference attention 插入 fake quant；推理 kernel 基于 SageAttention3 CUDA 实现。实验在 Wan2.1-1.3B/14B 上显示，FP4 或 SageAttention3 PTQ 会显著掉 VBench overall，而 Attn-QAT 基本恢复到 BF16 attention 水平。来源：[Attn-QAT §2-§3](https://arxiv.org/html/2603.00040v2)。

### 3.4 LongLive-2.0 的 NVFP4/W4A4

LongLive-2.0 是训练和推理全链路 NVFP4：

- **AR training**：linear layers 使用 NVFP4 recipe；weights 做 2D block scaling，activations/gradients 做 1D block scaling；reduction、norm stats、optimizer states 等数值敏感路径保持高精度。
- **DMD few-step distillation**：teacher/student 的 Real-Score、Fake-Score、Generator 都放到 W4A4 NVFP4 设定中；量化 backbone 冻结，只训练 LoRA。
- **Inference**：W4A4 generator；可把 LoRA 分支保留或 merge 成 fused low-rank kernel。
- **KV cache**：按 frame chunk 量化 K/V 到 NVFP4，实际约 3.6x cache compression，dequant overhead 报告低于 2%。

这类做法对 Blackwell 硬件非常友好，但对训练稳定性、kernel、通信和 LoRA 融合都有更高要求。来源：[LongLive-2.0 §2-§3](https://arxiv.org/html/2605.18739v2)。

### 3.5 LightX2V/TurboDiffusion 的部署量化

- TurboDiffusion：README 写明使用 W8A8 quantization 压缩 linear layers，并用 SageAttention/SLA 做 attention acceleration。量化 checkpoint 适合 RTX 5090/4090 等消费卡；H100 等大显存可用非量化 checkpoint。来源：[TurboDiffusion README](https://github.com/thu-ml/TurboDiffusion)。
- LightX2V：框架支持 `w8a8-int8`、`w8a8-fp8`、`w4a4-nvfp4`；README 标明有 Wan2.1 NVFP4 quantization-aware 4-step distilled models，以及 Wan2.2 NVFP4 sparse variant。来源：[LightX2V README](https://github.com/modeltc/lightx2v)。

## 4. 蒸馏方法分类

### 4.1 DMD / DMD2

DMD 的目标不是逐点复制 teacher 的完整多步轨迹，而是让 student 生成分布接近 teacher/真实数据分布。DMD2 改进 DMD 的训练稳定性和多步采样适配。FastVideo/USV 描述的 sparse distillation 使用 DMD2-style 机制：

- long-step full-attention teacher；
- few-step sparse student；
- frozen real score network；
- trainable fake score network；
- student 输出重新加噪后送入 real/fake score，得到 distribution-matching gradient；
- fake score network 用 diffusion loss 继续训练。

来源：[USV §3.2](https://arxiv.org/html/2512.05754v1)、[DMD2 论文](https://arxiv.org/abs/2405.14867)。

### 4.2 rCM：Score-Regularized Continuous-Time Consistency Model

TurboDiffusion 使用 rCM 做 timestep distillation，把 40-50 步甚至更多采样压到 1-4 步。README 中 `--num_steps` 支持 1-4，`--sigma_max` 控制 rCM 初始 sigma，较大 sigma 可增强质量但可能降低多样性。来源：[TurboDiffusion README](https://github.com/thu-ml/TurboDiffusion)、[TurboDiffusion 论文摘要](https://arxiv.org/abs/2512.16093)。

适合定位：

- 若目标是极致端到端速度，rCM 与 attention/linear quant 的叠加收益大。
- 风险是 prompt 分布、质量、多样性可能依赖蒸馏数据和 sigma/sampler 设定。

### 4.3 Step distillation / CFG distillation

LightX2V 多处发布 4-step distilled models/LoRA，并强调无 CFG 的 4-step 推理。其 README 还列出 CFG parallelism、step-distilled LoRA、FP8 权重等部署组合。来源：[LightX2V README](https://github.com/modeltc/lightx2v)。

这类蒸馏在工程中最直接的收益是把 denoising steps 从 40-50 降到 4 或 8。缺点是：

- 对训练数据覆盖、prompt 风格、motion 分布更敏感；
- 和稀疏/量化叠加时，训练-推理路径必须尽量一致，否则误差会叠加。

### 4.4 Self-Forcing / AR long-video tuning / LoRA few-step distillation

LongLive-2.0 直接把 bidirectional diffusion model fine-tune 成 long, multi-shot, interactive AR model，并尽量避开传统多阶段流程。其 few-step 能力来自独立 LoRA：

- 先做 long-video AR training；
- 再在 AR-trained model 上做 one-stage DMD distillation；
- 不 full fine-tune DiT backbone，只优化 LoRA modules；
- trained LoRA 可插入 AR model，将 denoising 降到 2 steps。

来源：[LongLive-2.0 §4.1](https://arxiv.org/html/2605.18739v2)。

## 5. 叠加训练怎么做

### 5.1 FastVideo/FastWan：VSA + DMD2 sparse distillation

训练图：

1. 从 dense Wan teacher 出发，teacher 保持 full attention 和长步采样。
2. student 替换为 VSA sparse attention，并只运行 reduced step set。
3. 用 DMD2-style real/fake score network 给 student 分布匹配梯度。
4. student 最终成为 few-step + VSA 的 sparse-distilled generator。

USV 论文把 FastWan baseline 概括为两维稀疏：architectural sparsity via VSA + sampling sparsity via reduced step schedule。FastVideo 文档也明确称 Sparse-distill 是把 DMD 和 VSA 放在同一个训练过程里。来源：[USV §3.2](https://arxiv.org/html/2512.05754v1)、[FastVideo Distillation Docs](https://hao-ai-lab.github.io/FastVideo/distillation/dmd)。

### 5.2 USV：VSA + DMD2 + token merging + entropy-aware budget

USV 不是从零训练，而是以 FastVideo sparse-distilled generator 为起点：

1. warm-up：固定 sparsity，先稳定 sparse student。
2. 加入 token merging：在 3D local blocks 中按 key similarity 合并高冗余 token。
3. 加入 entropy-aware policy：按 layer/timestep/input 动态分配 attention sparsity 和 token sparsity。
4. loss：distillation loss + budget loss + entropy regularization + temporal consistency regularization。
5. inference：只保留 sparse student 和 policy，不再需要 dense teacher。

核心意义：把“少算哪些 attention、少处理哪些 token、少走哪些 step”放在同一个预算优化问题里，而不是串行堆 trick。来源：[USV §3.3-§4](https://arxiv.org/html/2512.05754v1)。

### 5.3 SLA2：learnable sparse-linear routing + QAT

SLA2 的叠加不是 step distillation，而是 attention 内部结构和量化叠加：

1. 先用 full attention 的 Q/K/V 训练 router 和 `alpha`。
2. 将原模型 attention 替换成 SLA2。
3. sparse branch 用低比特 forward 做 QAT，backward 保持 FP16。
4. 全模型端到端 fine-tune diffusion loss。
5. 推理时 hard Top-k router + sparse kernel + low-bit sparse branch。

它适合插入在已蒸馏或未蒸馏的 Wan backbone 中，但如果与 DMD2/rCM 再叠加，应优先验证 training stability：step-distilled student 本身已改变采样分布，router/QAT 的误差可能与 few-step 误差相互放大。

### 5.4 LongLive-2.0：Balanced SP + NVFP4 + AR tuning + DMD LoRA

LongLive-2.0 的叠加重点是“训练/推理基础设施和算法共同设计”：

1. Balanced SP：每个 rank 持有同一 temporal chunk 的 clean/noisy paired stream，均衡 loss-bearing tokens，并让 VAE encoding 也按 chunk sharding。
2. NVFP4 training：AR long-video training 中的主要 GEMM 走 NVFP4，敏感路径保高精度。
3. DMD LoRA：few-step distillation 时 backbone 量化并冻结，只训练 LoRA，teacher/student 都对齐 W4A4。
4. Inference：W4A4 generator + NVFP4 KV cache + SP inference + async streaming VAE decoding。

这条路线对非常长、交互式、多 shot 视频尤其重要，因为瓶颈不只是 attention FLOPs，还包括 VAE、KV cache、通信和长上下文 memory。来源：[LongLive-2.0](https://arxiv.org/abs/2605.18739)。

### 5.5 TurboDiffusion：SageAttention/SLA + rCM + W8A8 + 工程优化

TurboDiffusion 的组合是：

- attention：SageAttention、SLA、SageSLA；
- timestep：rCM few-step distillation；
- linear：W8A8 quantization；
- 工程：FastNorm、TileLang W8A8、interactive serve、quantized checkpoints。

README 的使用参数能看出默认推荐路径：`--num_steps 4`、`--attention_type sagesla`、`--sla_topk 0.1`，使用量化 checkpoint 时加 `--quant_linear`。来源：[TurboDiffusion README](https://github.com/thu-ml/TurboDiffusion)。

### 5.6 LightX2V：4-step distill + quant/offload/cache 推理框架

LightX2V 的定位更像综合部署框架：

- 支持 Wan2.1/Wan2.2、HunyuanVideo、LTX、Qwen-Image 等。
- 提供 4-step distilled models/LoRA，推荐 4-step inference。
- 提供 FP8/NVFP4/INT8 等量化路径。
- 支持 block/phase offload，目标是消费级 GPU 也能跑 14B 480P/720P。
- 支持 Sage Attention、Flash Attention、Radial Attention、q8-kernel、vLLM 等算子/后端。

对你们的价值：可作为 deployment checklist，看哪些优化是“训练期必须参与”的，哪些只是“推理期可插拔”。来源：[LightX2V README](https://github.com/modeltc/lightx2v)。

## 6. 实验与评测

### 6.1 常见质量指标

| 指标 | 作用 | 注意事项 |
|---|---|---|
| VBench Total | 综合视频质量分 | 不同版本/子项聚合可能不同，必须标明 VBench 版本 |
| VBench Quality | 偏视觉质量、时序一致性等 | 适合看画面/运动是否掉 |
| VBench Semantic | 文本-视频语义一致性 | 稀疏/蒸馏后 prompt adherence 容易受影响 |
| IQ / Imaging Quality | 单帧图像质量 | SLA2、Attn-QAT 用该类子项 |
| AQ / Aesthetic Quality | 审美质量 | 容易受过饱和、模糊影响 |
| OC / Overall Consistency | 整体一致性 | 视频结构稳定性 |
| MS / Motion Smoothness | 运动平滑 | few-step distill 容易影响 |
| SC / Subject Consistency | 主体一致性 | 长视频/AR/KV cache 重要 |
| Vision Reward | 近似人类偏好的 reward model | SLA2 用于 human preference proxy |
| 盲测人评 | 主观质量验证 | Attn-QAT 对 99 个 VBench prompts 做 blind human eval |

### 6.2 常见效率指标

| 指标 | 含义 | 最容易混淆的点 |
|---|---|---|
| attention speedup | attention kernel 或 attention runtime 加速 | 不能等同端到端加速 |
| DiT denoising speedup | 只统计 DiT denoising 部分 | 通常不含 text encoder/VAE |
| E2E generation latency | 端到端生成时间 | 有些项目排除 text encoding 或 VAE decoding，需看原文 |
| step time | 单步 iteration/inference 时间 | 需要乘 steps 才接近 denoising latency |
| FPS | 长视频 streaming throughput | 是否含 VAE decoding 很关键 |
| FLOPs | 理论计算量 | sparse/linear branch、token merging 都要按真实路径算 |
| peak memory / KV cache size | 显存占用 | 长视频 AR 模型中 KV cache 随历史线性增长 |
| communication time | SP/TP/DP 通信开销 | 低比特 KV/QKV 传输可显著影响扩展性 |

### 6.3 代表性实验结果

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

## 7. 项目源码/工程优化梳理

> 说明：本节只基于公开 GitHub README/文档/目录结构做源码侧梳理，没有 clone 或运行项目。

| 项目 | 训练支持 | 推理支持 | Attention backend | Quant backend | Distill recipe/model | 硬件支持 | API/CLI 与工程结构 |
|---|---|---|---|---|---|---|---|
| FastVideo | full finetuning、LoRA finetuning、data preprocessing、DMD2 stepwise distillation、Sparse Distillation；FSDP2、sequence parallel、selective activation checkpointing | `VideoGenerator` Python API、CLI、Dreamverse realtime app | Video Sparse Attention、Sliding Tile Attention、Sage Attention、多 attention backend | HuggingFace 上有 QAD/FP8 等模型集合，README 主线更强调 VSA/DMD2 | FastWan2.1/2.2 sparse distillation recipes；FastVideo Synthetic datasets | H100、A100、4090、Linux/Windows/MacOS；DGX Spark 安装说明 | monorepo；`docs/`、`recipes` 链接、`apps/dreamverse/`、训练/推理统一框架 |
| LightX2V | `lightx2v_train` 目录；GenRL、LoRA/蒸馏模型发布；README 主体偏推理部署 | `LightX2VPipeline`、configs、examples、Gradio/ComfyUI/Windows one-click/service deployment | Sage Attention、Flash Attention、Radial Attention、q8-kernel、sgl-kernel、vLLM | `w8a8-int8`、`w8a8-fp8`、`w4a4-nvfp4`；NVFP4 operator in `lightx2v_kernel` | Wan2.1/2.2 Distill Models/LoRAs、Wan2.1-Distill-NVFP4、4-step distilled LoRA | H100、4090D、RTX 30/40/50、AMD ROCm、Ascend 910B、Cambricon、MetaX、Intel AIPC、T-head PPU 等 | `configs/`、`examples/`、`lightx2v_kernel/`、`lightx2v_platform/`；支持 offload/caching/parallel |
| TurboDiffusion | 提供 finetuned/distilled checkpoints；论文侧包含 rCM distillation；README 以 inference 为主 | T2V/I2V scripts、interactive serve；量化/非量化 checkpoints | original/sla/sagesla；SageAttention；SpargeAttn dependency | W8A8 linear quant；`--quant_linear`；quant checkpoints | TurboWan2.1/TurboWan2.2；rCM 1-4 steps | RTX 5090/4090 用 quant checkpoint；H100 用 unquantized checkpoint | `turbodiffusion/inference/`、`turbodiffusion/serve/`、`TurboT2AV/`；参数暴露 `--num_steps`、`--sla_topk`、`--attention_type` |

## 8. 对我们 Wan 叠加工作的建议

### 8.1 推荐实验矩阵

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

### 8.2 推荐训练优先级

1. **先稳定 few-step 蒸馏**：把 50/40 steps 压到 4/3/2 steps 是最大收益来源。先用 dense attention 建立质量下限和调参经验。
2. **再插入训练期稀疏**：优先 VSA 或 SLA2 这种训练期显式适配的方式，不建议只在推理期 hard prune attention。
3. **最后做与推理一致的 QAT**：FP4/NVFP4 特别需要 QAT；Attn-QAT 说明 naive FP4 attention 或 PTQ 对 Wan 会掉 overall quality。
4. **Blackwell 路线优先 NVFP4/W4A4**：如果目标硬件是 RTX 5090/B200/GB200，LongLive-2.0 和 Attn-QAT 的经验更相关；H100/4090 上则更现实地比较 FP8/W8A8/SageAttention/SageSLA。
5. **长视频 AR 场景不要只看 DiT FLOPs**：KV cache、SP 通信、VAE decoding、offload 都可能成为主瓶颈。

### 8.3 建议记录的 acceptance metrics

- 质量：VBench Total/Quality/Semantic + 子项 IQ/AQ/MS/SC + 50-100 prompts 人工盲测。
- 速度：attention kernel、DiT denoising、E2E generation 分开报。
- 显存：peak memory、KV cache size、offload traffic。
- 稳定性：训练 loss、gradient norm、NaN/OOM、不同 seeds 方差。
- 退化类型：过饱和、动态不足、temporal flicker、主体漂移、prompt 忽略、细节糊化。

### 8.4 风险点

- **few-step 蒸馏 + 高 sparsity**：两者都在减少计算路径，容易牺牲细节和 motion diversity。
- **PTQ FP4 attention**：Attn-QAT 在 Wan 上显示直接 FP4 或 SageAttention3 PTQ 仍有明显质量损失，应优先 QAT。
- **训练/推理 mismatch**：SoftTop-k vs hard Top-k、fake quant vs real quant、dense teacher vs sparse student 都需要单独做 ablation。
- **指标口径混乱**：论文和 README 经常分别报告 attention speedup、DiT speedup、E2E diffusion latency、完整 E2E latency；内部汇报必须强制分栏。
- **数据分布依赖**：TurboDiffusion README 提到当前模型只在 long English prompts 上训练，非同分布 prompt 需要 augment；你们自己的中文/多语言/短 prompt 场景要额外测。

## 9. 来源与可信度备注

一手来源：

- [SLA2: Sparse-Linear Attention with Learnable Routing and QAT](https://arxiv.org/abs/2602.12675)
- [Attn-QAT: 4-Bit Attention With Quantization-Aware Training](https://arxiv.org/abs/2603.00040)
- [LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation](https://arxiv.org/abs/2605.18739)
- [USV: Unified Sparsification for Accelerating Video Diffusion Models](https://arxiv.org/html/2512.05754v1)
- [TurboDiffusion GitHub](https://github.com/thu-ml/TurboDiffusion)
- [TurboDiffusion paper](https://arxiv.org/abs/2512.16093)
- [FastVideo GitHub](https://github.com/hao-ai-lab/fastvideo)
- [FastVideo Docs](https://hao-ai-lab.github.io/FastVideo/)
- [LightX2V GitHub](https://github.com/modeltc/lightx2v)

补充/待复核来源：

- [知乎 DMD & DMD2 解读](https://zhuanlan.zhihu.com/p/1953596250491434750)：搜索结果可确认主题是 DMD/DMD2 解读，但正文未稳定抓取，不作为关键事实依据。
- [微信公众号文章](https://mp.weixin.qq.com/s/GdE5s0sSfVZWH8w5HVpm1w)：正文未稳定抓取，不作为关键事实依据。

