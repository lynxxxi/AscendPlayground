# MindIE-SD 外部依赖调研和改进建议报告

## 1. 调研范围与结论摘要

本文基于当前仓库 `MindIE-SD_bugfix` 的静态调研结果输出，未联网核验外部站点的最新镜像或软件包状态。证据主要来自：

- 依赖与打包：`requirements.txt`、`requirements-test.txt`、`requirements-lint.txt`、`pyproject.toml`、`setup.py`
- 构建链路：`build/`、`csrc/CMakeLists.txt`、`csrc/ops/CMakeLists.txt`
- 镜像与环境：`docker/Dockerfile_910b_aarch64.ubuntu`、`docker/README.md`
- 文档与能力说明：`README.md`、`docs/zh/installing_guide.md`、`docs/zh/architecture.md`、`docs/zh/features/*`

调研按两个独立对外交付部件展开：

1. 部件1：`mindiesd` Python wheel/API 包。
2. 部件2：自定义算子/插件产物，包括 `mindiesd/plugin/*.so` 与 `mindiesd/ops/vendors/*`。

核心结论：

- `torch`、`torch_npu`、CANN、Ascend NPU 环境是系统能力成立的核心依赖，短期不应消除，只能通过版本矩阵、预编译包和环境检测降低耦合风险。
- `requirements.txt`、`setup.py`、文档和 Dockerfile 存在依赖口径不一致：`requirements.txt` 固定 `torch==2.9.0`，`setup.py` 只声明 `torch`、`torch_npu`；安装文档写 PyTorch `2.1.0`，Docker/requirements 使用 `2.9.0`；本地 Dockerfile 使用 Ubuntu 24.04，而官方镜像示例为 openEuler 24.03 LTS。
- `requirements.txt` 混入了运行、开发、示例和模型生态依赖，例如 `pre-commit` 属于开发工具，不应作为运行依赖安装。
- 自定义算子构建对 CANN 工具链、`bisheng`、`msopgen`、CMake、GCC/G++、PyTorch ABI 和芯片型号有强约束，应把“源码构建链路”和“预编译交付链路”分开管理。

## 2. 部件1：`mindiesd` Python wheel/API 包

### 2.1 部件说明

`mindiesd` 是对外暴露 Python API 的主要交付件，包含以下能力：

- `layers`：对外提供 attention、norm、rope、adalayernorm、量化相关 layer 接口。
- `cache_agent`：提供 DiTBlockCache、AttentionCache 等以存代算能力。
- `quantization`：提供 PTQ/权重量化/时间步量化接口。
- `compilation`：提供 `MindieSDBackend`，基于 `torch.compile` 和 FX/Inductor pattern 实现图替换。
- `eplb`、`offload`、`share_memory`、`utils`：提供多卡、专家负载均衡、显存/共享内存和基础工具能力。

仓库打包证据：

- `pyproject.toml` 声明项目名为 `mindiesd`，Python 要求为 `>=3.10`，动态依赖来自 `requirements.txt`。
- `setup.py` 自定义 `build_py`，构建 wheel 时会执行 `build/build.sh` 编译共享库，并将 `.so` 拷贝到 `mindiesd/plugin`。
- `pyproject.toml` 的 package data 声明包含 `mindiesd/plugin/*.so` 与 `mindiesd/ops/**/*`。

### 2.2 业务依赖识别与合理性分析

| 依赖 | 当前来源 | 引入原因 | 合理性分析 | 是否建议消除 | 消除或优化建议 |
| --- | --- | --- | --- | --- | --- |
| `torch==2.9.0` | `requirements.txt`；`setup.py` 仅声明 `torch` | 核心张量计算、模型模块、`torch.compile`、分布式通信、NPU 张量 API 入口 | 合理且不可缺少。MindIE-SD 是 PyTorch 生态上的 Ascend 加速包，`layers`、`cache`、`quantization`、`compilation` 都依赖 PyTorch | 不建议消除 | 统一声明方式：在约束文件中固定版本，在打包元数据中给出兼容范围；增加 PyTorch/torch_npu/CANN 兼容矩阵 |
| `torch_npu` | `setup.py`；Dockerfile 安装匹配 whl；大量代码直接 import | 调用 Ascend NPU 原生算子、NPU stream、format cast、量化 matmul、flash attention、MoE routing 等 | 合理且是 Ascend 加速能力的核心依赖 | 不建议消除 | 将其列入显式运行依赖或安装说明中的强制依赖；由于通常不来自 PyPI，应给出安装源、版本匹配和失败诊断 |
| `numpy==1.26.4` | `requirements.txt`；`mindiesd/eplb/greedy_algorithm.py` 直接使用 | EPLB/调度算法和测试中使用数值计算 | 合理，属于轻量基础依赖 | 不建议完全消除 | 若核心包想最小化依赖，可将仅测试使用的 numpy 调用隔离；当前 EPLB 直接依赖，仍应保留 |
| `einops` | `requirements.txt`；`mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py` 直接使用 | 稀疏 FA/RainFusion 中张量维度重排 | 合理，能显著降低张量 reshape 代码复杂度 | 不建议短期消除 | 若追求零依赖，可用原生 `torch.reshape/permute` 替换，但可读性下降，收益有限 |
| `safetensors` | 代码直接 import，但未在 `requirements.txt`/`setup.py` 中声明 | 量化权重加载，读取 `quant_model_weight_*.safetensors` | 合理，但当前声明缺失，存在安装后运行失败风险 | 不建议消除 | 补充到运行依赖或 `quantization` extra；量化能力可选时，设计 `pip install mindiesd[quant]` |
| `zmq`/`pyzmq` | `requirements.txt` 写 `zmq`；`mindiesd/share_memory.py` 直接 import `zmq` | 多实例共享内存句柄广播，PUB/SUB、REQ/REP 通信 | 对共享内存特性合理，但不是所有用户都会使用 | 不建议作为核心强制依赖 | 建议改为可选 extra：`mindiesd[share-memory]`；同时依赖名建议明确为 `pyzmq`，避免包名歧义 |
| `strenum` | `requirements.txt`；`mindiesd/quantization/mode.py` 在 Python 低版本 fallback 使用 | Python <3.11 下提供 `StrEnum` | 对 `requires-python >=3.10` 合理；Python 3.11+ 可用标准库 | 可在 Python 3.11+ 环境消除 | 若继续支持 Python 3.10，应保留；若收敛到 Python 3.11+，可移除并使用 `enum.StrEnum` |
| `diffusers==0.29.0` | `requirements.txt`；示例和文档围绕 diffusers 模型迁移 | 扩散模型生态适配，Cache DiT + diffusers 能力、示例代码、模型 pipeline | 作为模型生态依赖合理，但核心 `mindiesd` 包并非所有 API 都直接需要 | 不建议作为核心强制依赖 | 拆为 `mindiesd[diffusers]` 或示例依赖，避免安装基础 layer/算子时拉入完整模型生态 |
| `transformers==4.44.2` | `requirements.txt` | 文生图/多模态模型常见文本编码器和 pipeline 依赖 | 合理性取决于具体模型仓；当前 `mindiesd` 核心代码未直接 import | 不建议核心强制依赖 | 放入模型/示例依赖，按模型 README 安装 |
| `torchvision==0.24.0` | `requirements.txt` | 视觉模型处理、图像变换或模型示例依赖 | 与 `torch==2.9.0` 版本匹配有必要，但核心代码未直接依赖 | 可从核心依赖移出 | 放入示例或模型 extra |
| `open_clip_torch==2.26.1`、`timm==0.9.12`、`accelerate==0.26.1` | `requirements.txt` | 多模态模型组件、视觉 backbone、推理加速/模型加载生态 | 对模型示例合理，但对 `mindiesd` 基础能力不是必需 | 可从核心依赖移出 | 放入 `requirements-model.txt` 或具体模型仓依赖 |
| `av==12.0.0` | `requirements.txt` | 视频输入输出处理，服务/示例或视频生成场景需要 | 对视频模型合理，但核心 API 非必需 | 可从核心依赖移出 | 放入视频示例 extra |
| `tqdm==4.66.5`、`tensorboard==2.20.0`、`mmengine==0.10.4`、`ftfy==6.1.3`、`bs4` | `requirements.txt` | 进度展示、日志可视化、模型生态工具、文本清洗、HTML 解析 | 更像示例/模型/开发辅助依赖，放入核心运行依赖会扩大安装面 | 建议从核心依赖消除 | 按用途拆到 `requirements-example.txt`、`requirements-model.txt` 或 `requirements-dev.txt` |

### 2.3 构建工具依赖识别与引入原因

| 构建依赖 | 当前来源 | 引入原因 | 合理性分析 | 是否建议消除 | 消除或优化建议 |
| --- | --- | --- | --- | --- | --- |
| `setuptools>=65.5` | `pyproject.toml` | Python 包构建后端，发现 package、打包 package data | 合理，当前项目使用 setuptools 自定义构建 | 不建议消除 | 统一使用 PEP 517 构建，保留最低版本要求 |
| `wheel` | `pyproject.toml`、`setup.py` | 生成二进制 wheel，并通过自定义 `bdist_wheel` 标记非 pure wheel | 合理，包内包含 `.so` 与算子资源 | 不建议消除 | 保留；在构建文档中明确必须产出平台 wheel |
| `python -m build --wheel --no-isolation` | `docs/*/developer_guide/build_guide.md` | 源码构建 wheel 的推荐入口 | 合理，但 `--no-isolation` 表示依赖宿主环境完整性 | 不建议消除 | 补充标准构建容器，说明何时可使用隔离构建 |
| `bash build/build.sh` | `setup.py` 自定义 `build_py` | 构建 Python 包时同时构建 CANN 自定义算子和 PyTorch 扩展 | 对一体化交付合理，但加重普通 Python 包安装复杂度 | 建议弱化强耦合 | 提供预编译 wheel；源码构建时才触发 `build.sh`；支持跳过算子构建的环境变量 |
| Python `>=3.10` | `pyproject.toml` | 使用现代 typing/dataclass/torch 生态 | 合理 | 不建议消除 | 与 Dockerfile Python `3.11.4`、文档示例保持版本矩阵一致 |

### 2.4 其它依赖识别与引入原因

| 依赖类别 | 依赖 | 当前来源 | 引入原因 | 改进建议 |
| --- | --- | --- | --- | --- |
| 测试 | `pytest==8.3.4`、`coverage==7.6.10` | `requirements-test.txt` | 单元测试与覆盖率检查 | 保持独立文件，不进入运行依赖 |
| Lint/类型检查 | `pre-commit==4.1.0`、`mypy==1.13.0` | `requirements-lint.txt` | 提交前检查、类型检查 | 保持独立；注意 `requirements.txt` 中另有 `pre-commit==3.8.0`，应移除或统一 |
| 运行清单中混入的开发工具 | `pre-commit==3.8.0` | `requirements.txt` | 开发阶段钩子工具，不属于推理运行 | 从运行依赖移出，避免生产环境安装无关工具 |
| 服务化示例 | `ray`、`fastapi`、`uvicorn`、`pydantic`、`Pillow` | `examples/service/*.py`、`examples/cache/cache.py` 直接 import，但未在主 requirements 中完整声明 | 示例服务、异步 worker、HTTP API、请求模型、图像处理 | 新增 `examples/service/requirements.txt` 或 `requirements-service.txt` |
| 文档 | `docs/requirements-docs.txt` | 文档构建 | 生成项目文档 | 保持独立，不进入 wheel 依赖 |
| 第三方模型仓 | `wan`、`qwenimage_edit` 等 | examples 中 import，文档要求按 Modelers/ModelZoo 模型 README 安装 | 具体模型推理脚本依赖 | 不应由 MindIE-SD 核心包统一承载，继续由模型仓声明 |

### 2.5 部件1依赖合理性与消减建议

整体判断：

- 核心运行依赖合理，但依赖声明边界不清。
- 模型生态依赖、开发工具、示例服务依赖不应全部放在 `requirements.txt` 中作为默认运行依赖。
- 当前 `setup.py` 与 `pyproject.toml`/`requirements.txt` 双轨声明不一致，容易造成“源码开发安装”和“wheel 安装”行为不同。

建议按以下方式整改：

1. 建立分层依赖文件。
   - `requirements-runtime.txt`：只保留核心运行依赖，例如 `torch`、`torch_npu` 安装说明、`numpy`、`einops`、`safetensors`、`strenum`。
   - `requirements-model.txt`：放 `diffusers`、`transformers`、`torchvision`、`open_clip_torch`、`timm`、`accelerate`、`av`、`ftfy` 等模型生态依赖。
   - `requirements-service.txt`：放 `ray`、`fastapi`、`uvicorn`、`pydantic`、`Pillow`。
   - `requirements-dev.txt`：合并 lint/test/docs 依赖，移除运行清单中的 `pre-commit`。
2. 统一打包依赖入口。
   - 避免 `pyproject.toml` 从 `requirements.txt` 动态读取全量开发依赖。
   - `setup.py` 中的 `install_requires=["torch", "torch_npu"]` 应与实际运行依赖同步，或者迁移到 `pyproject.toml` 的 optional dependencies。
3. 将重型能力设计为 extras。
   - 例如 `mindiesd[quant]`、`mindiesd[diffusers]`、`mindiesd[service]`、`mindiesd[share-memory]`。
4. 增加导入级别的依赖检测。
   - 对可选能力使用延迟 import 和友好错误提示，避免用户只使用基础 layer 时因为未安装服务/模型依赖失败。
5. 发布兼容矩阵。
   - 明确 Python、torch、torch_npu、CANN、OS、芯片型号之间的验证组合。

## 3. 部件2：自定义算子/插件产物

### 3.1 部件说明

自定义算子/插件是 MindIE-SD 性能能力的底层交付件，主要包括两类：

- PyTorch 扩展插件：`csrc/plugin` 编译生成共享库，并在构建时拷贝到 `mindiesd/plugin`。
- CANN 自定义算子包：`csrc/ops` 通过 AscendC/TIK/CANN 工具链构建，输出到 `mindiesd/ops/vendors/*`。

相关构建证据：

- `build/build.sh` 顺序调用 `build/build_ops.sh` 和 `build/build_plugin.sh`。
- `build/build_ops.sh` 使用 CANN 的 `msopgen` 生成工程，并构建 `laser_attention`、`la_preprocess`、`block_sparse_attention`、`sparse_block_estimate` 等算子。
- `build/build_plugin.sh` 使用 CMake 构建 `csrc` 下 PyTorch 扩展，并根据 PyTorch C++ ABI 设置 `USER_ABI_VERSION`。
- `csrc/CMakeLists.txt` 链接 `torch`、`torch_cpu`、`torch_npu`、`ascendcl`、`libnnopbase.so`、`libopapi.so` 等运行库。

### 3.2 业务依赖识别与合理性分析

| 依赖 | 当前来源 | 引入原因 | 合理性分析 | 是否建议消除 | 消除或优化建议 |
| --- | --- | --- | --- | --- | --- |
| CANN Toolkit | `docker/Dockerfile_910b_aarch64.ubuntu`、`build/*`、`csrc/*` | 提供 Ascend 编译、运行、头文件、ACL/ACLNN、AscendC 等基础能力 | 合理且不可缺少，自定义算子无法脱离 CANN 构建和运行 | 不建议消除 | 固化版本矩阵，构建和运行分别声明 Toolkit/Kernels/driver 要求 |
| CANN Kernels/910B ops/NNAL | Dockerfile 安装 `Ascend-cann-910b-ops_8.5.1`、`Ascend-cann-nnal_8.5.1` | 提供设备算子、NN 加速库、算子运行环境 | 合理，与高性能 kernel 能力直接相关 | 不建议消除 | 将 runtime 镜像中需要的最小 CANN 组件列清楚 |
| AscendC | `csrc/ops/ascendc`、`build/build_ascendc_ops.sh` | 构建激光 attention、block sparse attention、sparse block estimate 等 AICore kernel | 合理，性能核心依赖 | 不建议消除 | 提供预编译交付，普通用户不需要本地 AscendC 编译 |
| TIK | `csrc/ops/tik`、`build/build_tik_ops.sh` | 构建 TIK flash attention 相关算子 | 合理，兼容部分算子实现路径 | 不建议直接消除 | 若未来 AscendC 覆盖完全，可逐步淘汰 TIK 实现 |
| PyTorch C++/ATen | `csrc/plugin/*.h/*.cpp`、`csrc/CMakeLists.txt` | 注册 PyTorch 自定义 op，接入 Tensor、Library、extension API | 合理，是 Python API 与底层算子的桥接层 | 不建议消除 | 保持 ABI 检测，按 torch 版本构建对应 wheel |
| `torch_npu` C++ headers/runtime | `csrc/plugin/pytorch_npu_helper.h`、`csrc/CMakeLists.txt` | 调用 NPU stream、format、OpCommand、CalcuOpUtil 等接口 | 合理，是 Ascend PyTorch 扩展的核心依赖 | 不建议消除 | 明确 torch_npu 版本与 CANN/torch 的绑定关系 |
| ACL/ACLNN/HCCL | `csrc/plugin/pytorch_npu_helper.h`、`csrc/CMakeLists.txt` | 访问底层 runtime、算子 API、通信头文件 | 合理，支撑自定义算子执行和多卡能力 | 不建议消除 | 对运行态缺库给出安装前检查脚本 |
| 芯片型号 `ascend910`、`ascend910b`、`ascend910_93` | `build/build_ops.sh`、`csrc/ops/*/*_def.cpp` | 为不同 Ascend SoC 生成算子配置 | 合理，算子与芯片能力强相关 | 不建议消除 | 按交付目标拆分多架构包或构建参数，减少不必要芯片产物 |

### 3.3 构建工具依赖识别与引入原因

| 构建依赖 | 当前来源 | 引入原因 | 合理性分析 | 是否建议消除 | 消除或优化建议 |
| --- | --- | --- | --- | --- | --- |
| `cmake` | `build/build_plugin.sh`、`build/build_ascendc_ops.sh`、`csrc/CMakeLists.txt` | 配置和构建 PyTorch 扩展、CANN 自定义算子 | 合理，C++/AscendC 项目通用构建工具 | 不建议消除 | 固定最低版本：插件 CMakeLists 为 3.10，ops CMakeLists 为 3.16，应文档化 |
| `gcc/g++` | Dockerfile 安装 gcc-9/g++-9/gcc-13/g++-13；文档提示安装 gcc/g++ | 编译 C++ 插件和 host 侧代码 | 合理且必要 | 不建议消除 | 明确推荐版本和 ABI；避免文档只写泛化 `gcc g++` |
| `bisheng` | `build/build_ascendc_ops.sh` | AscendC kernel 编译工具 | 合理且必要 | 不建议消除 | 在环境检查中提前验证 `which bisheng` |
| `msopgen` | `build/build_ops.sh`、`build/build_tik_ops.sh` | 生成 CANN 自定义算子工程，控制发布内容 | 合理，用于标准 CANN 自定义算子工程生成 | 不建议消除 | 将生成过程封装到构建容器，减少用户手动配置 |
| `ccache` | Dockerfile 安装；`build/build_ascendc_ops.sh` 可启用 | 加速重复编译 | 合理但非必需 | 可以作为可选依赖 | 没有 ccache 时继续构建，仅降低速度 |
| CANN `.run` 包 | Dockerfile 下载 Toolkit/910B ops/NNAL | 安装 CANN 构建与运行环境 | 合理，但网络下载和版本漂移风险较高 | 不建议消除 | 固化包 checksum；内网制品库缓存；区分构建镜像和运行镜像 |
| 环境变量 `ASCEND_TOOLKIT_HOME`、`ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` | `build/*`、`csrc/*` | 定位 CANN 安装目录、OPP 目录和运行库 | 合理，但易配置错误 | 不建议消除 | 提供 `scripts/check_env.sh` 类预检，输出缺失项和建议 |
| `USER_ABI_VERSION` | `build/build_plugin.sh`、`csrc/CMakeLists.txt` | 保持 PyTorch C++ ABI 一致 | 合理且必要 | 不建议消除 | 构建日志中显式打印 torch ABI、Python ABI、平台 tag |

### 3.4 其它依赖与约束

| 约束 | 当前证据 | 影响 | 改进建议 |
| --- | --- | --- | --- |
| 硬件型号 | `docs/zh/features/supported_matrix.md` 列出 Atlas 800I A2、Atlas 800I A3、Atlas 300I DUO | 不同模型和特性支持矩阵不同，量化/稀疏/并行能力并非所有硬件均可用 | 将硬件型号、NPU 数量、显存、支持特性形成机器可读矩阵 |
| 架构平台 | Dockerfile 和 PyTorch wheel 使用 `aarch64`、`manylinux_2_28_aarch64` | wheel 与系统 glibc/架构强绑定 | 发布 wheel 时按 Python ABI、平台、芯片族打 tag |
| 算子路径 | `ASCEND_CUSTOM_OPP_PATH` 指向 `mindiesd/ops/vendors/customize` 和 `mindiesd/ops/vendors/aie_ascendc` | 运行时必须能找到自定义 OPP | 安装后自动设置或提供入口脚本；错误时给出路径检查 |
| 设备挂载 | 安装文档中 Docker run 挂载 `/dev/davinci*`、driver、firmware | 容器内运行依赖宿主机驱动固件和设备权限 | 提供最小权限和推荐权限两套说明，解释 `rwm` 风险和必要性 |

### 3.5 部件2依赖合理性与消减建议

整体判断：

- 自定义算子/插件对 CANN、torch_npu、Ascend 硬件、C++ ABI 的强依赖是产品定位决定的，不能通过普通重构消除。
- 当前主要问题不是“依赖过多”，而是“构建链路暴露给用户过多、版本矩阵不够明确、源码构建和二进制交付边界不清”。

建议按以下方式降低依赖成本：

1. 默认交付预编译二进制 wheel。
   - 普通用户安装 wheel 即可获得 `mindiesd/plugin/*.so` 和 `mindiesd/ops/vendors/*`。
   - 源码构建作为开发者路径，要求完整 CANN/编译器工具链。
2. 引入构建环境预检。
   - 检查 Python、torch、torch_npu、CANN、`bisheng`、`msopgen`、CMake、GCC/G++、ABI 和 `ASCEND_*` 环境变量。
3. 建立算子可选开关。
   - 允许只构建/打包指定算子，例如只构建 `laser_attention` 或只启用 PyTorch 插件。
4. 固化二进制兼容矩阵。
   - 至少覆盖 OS、glibc、Python ABI、torch、torch_npu、CANN、芯片型号、wheel tag。
5. 明确运行态最小依赖。
   - 从构建工具链中剥离 `cmake`、`gcc`、`bisheng`、`msopgen` 等，仅在开发镜像保留。

## 4. 镜像依赖调研和改进建议

### 4.1 当前基础操作系统和容器软件版本约束

当前仓库存在两类镜像口径：

1. 仓库内开发镜像 Dockerfile：
   - 基础镜像：`ubuntu:24.04`
   - 架构：`aarch64`
   - Python：`3.11.4`
   - CANN：`8.5.1`
   - torch：`2.9.0`
   - torch_npu：`2.9.0.*`，Dockerfile 从 `pytorch_v2.9.0-7.3.0_py311.tar.gz` 中安装匹配 wheel
   - wheel 平台：`manylinux_2_28_aarch64`

2. 文档中的官方 MindIE 镜像示例：
   - 镜像名示例：`mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts`
   - OS 口径：openEuler 24.03 LTS
   - Python 口径：`py311`
   - 硬件口径：`800I-A2`

宿主机容器软件约束：

- 安装文档建议用户自行安装 Docker，版本要求/建议为 `>=24.x.x`。
- 容器运行需要宿主机已安装 NPU driver/firmware，并挂载 `/dev/davinci_manager`、`/dev/hisi_hdc`、`/dev/devmm_svm`、`/dev/davinci0`、`/usr/local/Ascend/driver`、`/usr/local/Ascend/firmware` 等。

### 4.2 系统当前提供的镜像现状

仓库当前提供：

- 开发镜像定义：`docker/Dockerfile_910b_aarch64.ubuntu`
- 镜像说明：`docker/README.md`
- 构建示例：`docker build --network=host -f docker/Dockerfile_910b_aarch64.ubuntu -t mindiesd:910b-aarch64-head .`
- 运行示例：`mindiesd:910b-aarch64-head`

文档当前引用：

- 官方 MindIE 镜像下载入口。
- 镜像示例 `mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts`。

现状判断：

- 仓库内镜像更像“源码构建/CI/开发镜像”，包含编译器、CANN 安装包、Python 构建链、OBS 工具和源码。
- 文档中的 MindIE 镜像更像“用户运行/交付镜像”，但仓库没有给出该镜像的 Dockerfile、层内容、SBOM 或完整版本矩阵。

### 4.3 系统当前验证的操作系统及版本现状

可从仓库证据确认的验证/适配状态：

| 环境 | 版本/口径 | 证据 | 结论 |
| --- | --- | --- | --- |
| 本地开发 Dockerfile | Ubuntu 24.04 aarch64 | `docker/Dockerfile_910b_aarch64.ubuntu` | 仓库内可复现构建目标 |
| 官方镜像示例 | openEuler 24.03 LTS, py311 | `docs/zh/installing_guide.md`、`docs/en/installing_guide.md` | 文档推荐运行镜像口径 |
| Python | `>=3.10`，Docker 使用 `3.11.4` | `pyproject.toml`、Dockerfile | 支持范围和验证版本未完全对齐 |
| PyTorch | 文档物理机写 `2.1.0`；requirements/Dockerfile 使用 `2.9.0` | `docs/zh/installing_guide.md`、`requirements.txt`、Dockerfile | 存在明显不一致，需要修正 |
| CANN | Dockerfile 固定 `8.5.1`；文档链接指向 CANN 安装指南 | Dockerfile、安装文档 | 仓库内实际构建使用 8.5.1 |

当前不足：

- 没有在仓库内形成明确的 OS 支持矩阵。
- Ubuntu 开发镜像与 openEuler 运行镜像并存，但边界未在顶层文档清晰解释。
- 物理机安装文档未写明已验证的 OS、CANN、driver、firmware、torch_npu 组合。

### 4.4 系统对操作系统的依赖分析

| OS/系统依赖 | 影响范围 | 分析 |
| --- | --- | --- |
| Linux aarch64 | Dockerfile、torch/torch_npu wheel、CANN 包 | 当前镜像和 wheel 下载都指向 aarch64，非 aarch64 环境不能直接复用 |
| glibc/manylinux_2_28 | torch wheel、torch_npu wheel、`.so` 插件 | 二进制兼容性受 glibc 和平台 tag 约束 |
| NPU driver/firmware | 运行态 | 容器不内置宿主驱动能力，必须挂载宿主机驱动、固件和设备节点 |
| CANN Toolkit/Kernels/NNAL | 构建态和运行态 | 构建自定义算子、加载 ACL/ACLNN、运行 OPP 均依赖 CANN |
| GCC/G++ 与 libstdc++ ABI | 构建态和二进制运行态 | PyTorch C++ ABI 通过 `USER_ABI_VERSION` 适配，系统 C++ 运行库版本影响 `.so` 加载 |
| Bash、coreutils、CMake、make | 构建态 | `setup.py` 直接调用 bash 构建脚本，非 Linux 或缺少构建工具会失败 |
| Docker 设备权限与 IPC/network | 容器运行态 | NPU 运行依赖 device 挂载，服务化/分布式场景依赖 host network、IPC、共享内存等配置 |

### 4.5 操作系统验证/镜像改进策略

建议按“开发镜像、运行镜像、官方镜像适配说明”三层治理：

1. 明确镜像类型。
   - 开发镜像：包含 GCC/G++、CMake、CANN Toolkit、`bisheng`、`msopgen`、测试工具，用于源码构建和 CI。
   - 运行镜像：只包含 Python、torch、torch_npu、CANN runtime/OPP、MindIE-SD wheel 和必要系统库。
   - 官方 MindIE 镜像：作为用户推荐入口，文档只描述如何选择匹配版本和如何安装本仓库产物。
2. 统一版本矩阵。
   - 建议新增表格：OS、架构、Python、CANN、driver/firmware、torch、torch_npu、MindIE-SD、硬件型号。
   - 将文档中的 PyTorch `2.1.0` 修正为与当前仓库实际一致的 `2.9.0`，或明确区分历史版本和当前版本。
3. 增加镜像可追溯性。
   - 为 Dockerfile 下载的 CANN、torch、torch_npu、obsutil 等外部包增加 checksum。
   - 生成 SBOM，记录 apt 包、pip 包、CANN 包、系统库版本。
4. 降低运行镜像体积。
   - 多阶段构建：第一阶段编译 wheel 和算子，第二阶段只拷贝 wheel、OPP、`.so` 和运行库。
   - 移除运行镜像中的编译器、源码、缓存、下载工具和不必要开发包。
5. 增加启动前验证。
   - 提供 `python -m mindiesd.env_check` 或 shell 脚本，检查 NPU 可见性、driver/firmware、CANN、torch_npu、`ASCEND_CUSTOM_OPP_PATH`、自定义算子加载。

## 5. 需优先修正的不一致点

| 问题 | 当前表现 | 风险 | 建议优先级 |
| --- | --- | --- | --- |
| `requirements.txt` 与 `setup.py` 依赖不一致 | `requirements.txt` 固定大量依赖和 `torch==2.9.0`；`setup.py` 只安装 `torch`、`torch_npu` | 不同安装方式得到不同运行环境；部分能力缺依赖 | 高 |
| `safetensors` 实际 import 但未声明 | `mindiesd/quantization/quantize.py` 使用 `safetensors.safe_open` | 使用量化接口时可能运行失败 | 高 |
| `pre-commit` 被放入运行依赖 | `requirements.txt` 有 `pre-commit==3.8.0`，`requirements-lint.txt` 又有 `pre-commit==4.1.0` | 生产安装冗余，且版本冲突 | 高 |
| PyTorch 文档版本与实际版本不一致 | 文档写 `2.1.0`，Docker/requirements 写 `2.9.0` | 用户按文档安装后可能 ABI 或 API 不兼容 | 高 |
| Ubuntu Dockerfile 与 openEuler 官方镜像示例并存 | Dockerfile 为 Ubuntu 24.04，文档示例为 openEuler 24.03 LTS | 用户不清楚哪个是验证交付环境 | 中 |
| 示例服务依赖未独立声明 | `examples/service` 使用 `ray/fastapi/uvicorn/pydantic/Pillow` 等 | 示例不可复现或污染核心依赖 | 中 |
| CANN/driver/torch_npu 兼容矩阵缺失 | 只有散落在 Dockerfile 和文档中的版本信息 | 环境问题定位困难 | 高 |

## 6. 综合改进路线

建议分三阶段推进：

### 阶段一：依赖声明收敛

- 拆分 `requirements-runtime.txt`、`requirements-model.txt`、`requirements-service.txt`、`requirements-dev.txt`。
- 移除运行依赖中的 `pre-commit`。
- 补齐 `safetensors`、`pyzmq` 等实际运行依赖。
- 统一 `pyproject.toml`、`setup.py`、文档中的依赖来源。

### 阶段二：二进制交付与环境矩阵

- 发布预编译 wheel，避免普通用户本地构建 CANN 自定义算子。
- 明确 wheel tag、Python ABI、torch/torch_npu/CANN/OS/芯片支持矩阵。
- 增加环境预检脚本，覆盖 NPU、CANN、OPP、自定义算子、ABI。

### 阶段三：镜像治理

- 分离开发镜像和运行镜像。
- 用多阶段构建生成最小运行镜像。
- 为镜像增加 checksum、SBOM、版本标签规范。
- 修正文档中 PyTorch、OS、镜像示例的版本口径，避免 Ubuntu/openEuler、2.1.0/2.9.0 混用造成误导。

## 7. 结论

MindIE-SD 的核心外部依赖与其产品定位基本匹配：它本质上是基于 PyTorch、torch_npu、CANN 和 Ascend 硬件的软件加速套件，因此核心依赖不应简单消除。当前最需要改进的是依赖治理和交付边界：把核心运行依赖、模型生态依赖、示例服务依赖、开发构建依赖拆开；把源码构建链路与预编译交付链路拆开；把开发镜像与运行镜像拆开。完成这些工作后，系统对外交付的可安装性、可复现性和环境问题定位效率都会明显提升。
