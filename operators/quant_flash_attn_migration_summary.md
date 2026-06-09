# QuantFlashAttn / Metadata 迁移总结

## 1. 迁移目标与边界

本次迁移目标是把 CANN `ops-transformer` 中与 `quant_flash_attn` 实际运行相关的能力同步到 MindIE-SD，并保持 MindIE-SD 现有的包结构、命名空间、构建方式和运行时环境设置。

最终纳入范围分为四类：

- `quant_flash_attn` AscendC kernel、tiling、op host、手写 ACLNN opapi。
- `quant_flash_attn_metadata` proto/L0 opapi/AICPU kernel。
- QFA 和 metadata 实际依赖的最小 `common` 闭包。
- MindIE-SD 自己的 PyTorch plugin、Python 测试入口、build/package/runtime env 适配。

本次没有迁移 `fused_infer_attention_score`。CANN new 中该算子更新较多，但 MindIE-SD 当前没有对应 `csrc/ops/fused_infer_attention_score` 目录，QFA、metadata、plugin 和测试也没有 active dependency。

本次也没有引入 `vector_common.h`。CANN new 中该文件有更新，但 MindIE-SD 当前没有该文件，QFA 依赖链没有 active include，仅 `copy_gm_to_ub.h` 的 `#if 0` 注释中提到。

## 2. CANN 到 MindIE-SD 的目录映射

QFA 在 CANN 中主要来自：

```text
attention/quant_flash_attn/
```

迁移到 MindIE-SD 后落在：

```text
csrc/ops/quant_flash_attn/
```

metadata 在 CANN 中主要来自：

```text
attention/quant_flash_attn_metadata/
```

迁移到 MindIE-SD 后落在：

```text
csrc/ops/quant_flash_attn_metadata/
```

CANN 中与 QFA 实际 include 相关的公共代码来自：

```text
attention/common/
```

MindIE-SD 中只保留实际依赖的最小文件集合，主要落在：

```text
csrc/ops/common/
```

CANN 的 Python wrapper 和 torch extension 不直接搬入 MindIE-SD 包结构，而是吸收其对外语义，落到 MindIE-SD 的 plugin 和测试里：

```text
csrc/plugin/
tests/ops/quant_flash_attn/
```

## 3. 算子源码同步内容

### 3.1 QuantFlashAttn kernel

QFA 的 AscendC kernel 以 CANN new 为准同步，核心范围集中在 `arch35` 和 `arch35/vf`。

本轮 CANN new 同步涉及 10 个 kernel 文件：

```text
csrc/ops/quant_flash_attn/op_kernel/arch35/quant_flash_attn_block_cube_dn.h
csrc/ops/quant_flash_attn/op_kernel/arch35/quant_flash_attn_block_vector_dn.h
csrc/ops/quant_flash_attn/op_kernel/arch35/quant_flash_attn_common_def.h
csrc/ops/quant_flash_attn/op_kernel/arch35/quant_flash_attn_kernel_dn.h
csrc/ops/quant_flash_attn/op_kernel/arch35/quant_flash_attn_template_tiling_key.h
csrc/ops/quant_flash_attn/op_kernel/arch35/vf/vf_attenOut_dn_mxfp4.h
csrc/ops/quant_flash_attn/op_kernel/arch35/vf/vf_computeScale_dn_mxfp4.h
csrc/ops/quant_flash_attn/op_kernel/arch35/vf/vf_nd2nz_indexes_dn_mxfp4.h
csrc/ops/quant_flash_attn/op_kernel/arch35/vf/vf_softmax_dn_cast_nz_mxfp4.h
csrc/ops/quant_flash_attn/op_kernel/arch35/vf/vf_updateScale_dn_mxfp4.h
```

### 3.2 QuantFlashAttnMetadata AICPU

metadata 是 AICPU-only 辅助算子，不走普通 AscendC kernel 路径。迁移内容包括：

- metadata proto / graph / op host 定义。
- metadata L0 ACLNN opapi。
- metadata AICPU kernel。
- metadata AICPU json。

本轮 CANN new 同步中，metadata AICPU 保留了新的 core 数覆盖：

```cpp
aicCoreNum_ = 1;
aivCoreNum_ = 2;
```

MindIE-SD 侧继续保留 `custom_soc_version` 适配，同时兼容读取 CANN 原始 `soc_version`。

### 3.3 Common 最小依赖

QFA kernel 中存在对 `attention/common` 的相对 include。MindIE-SD 只同步实际依赖文件，不迁入整个 common 目录。

本轮同步的 common 文件是：

```text
csrc/ops/common/op_kernel/memcopy/copy_ub_to_gm.h
csrc/ops/common/op_kernel/memcopy/offset_calculator_v2.h
csrc/ops/common/op_kernel/memcopy/parser.h
```

这类文件必须同时满足源码位置和构建时拷贝位置。QFA header 编译时位于：

```text
binary/<soc>/src/quant_flash_attn/arch35
```

其相对 include 会解析到：

```text
binary/<soc>/common/op_kernel/...
```

因此 MindIE-SD 的构建系统需要把 common helper 拷贝到 `binary/<soc>/common/op_kernel`，而不是 `binary/<soc>/src/common/op_kernel`。

## 4. MindIE-SD 侧接口适配

### 4.1 Torch namespace

CANN torch extension 暴露的是：

```text
torch.ops.npu_ops_transformer.*
```

MindIE-SD 统一暴露为：

```text
torch.ops.mindiesd.*
```

因此测试和 Python 入口只复用 CANN 的输入生成、golden 计算和精度比较语义，不保留 CANN 的 namespace。

### 4.2 QFA plugin wrapper

QFA wrapper 落在：

```text
csrc/plugin/quant_flash_attn.cpp
csrc/plugin/quant_flash_attn.h
```

适配重点是对齐 CANN torch extension 的调用语义，同时保留 MindIE-SD plugin 风格：

- schema 使用 `mindiesd::quant_flash_attn`。
- packed FP4/FP8 tensor 的真实 ACL dtype 通过 dtype override 传入。
- dtype 转换统一走 `ConvertToAclDataType` 等价路径。
- 输出 tensor 显式分配到 NPU device。
- 输出 format 继续复用 MindIE-SD 的 `empty_with_format` 和 `get_npu_format(query)`。
- ACLNN 调用中保持 `query/key/value` wrapper 顺序正确，`value` 传 `valueWrapper`。

### 4.3 Metadata plugin wrapper

metadata wrapper 落在：

```text
csrc/plugin/quant_flash_attn_metadata.cpp
csrc/plugin/quant_flash_attn_metadata.h
```

metadata 的输入 tensor 都是 optional。部分 BSND/连续 KV 场景下，`cu_seqlens_q`、`cu_seqlens_kv`、`seqused_q`、`seqused_kv` 可以全部为 `None`。这时 PyTorch dispatcher 没有 tensor 参数可用于推断 NPU backend。

MindIE-SD 因此同时注册：

- `PrivateUse1` 实现：有 NPU tensor 参数时正常走 NPU dispatch。
- `CatchAll` 实现：无 tensor 参数时仍进入同一个 metadata wrapper。

metadata 输出为固定 int32 buffer，对齐 CANN wrapper 的 metadata 输出约定。

### 4.4 Plugin 公共 helper

公共 helper 主要落在：

```text
csrc/plugin/pytorch_npu_helper.h
```

本次补齐了 CANN torch extension 中的 dtype 转换语义：

- `at::ScalarType` 到 `aclDataType` 通过 `ConvertToAclDataType`。
- QFA 的 FP4/FP8 packed uint8 tensor 继续通过 real dtype override 映射到 `ACL_FLOAT4_E2M1`、`ACL_FLOAT8_E8M0`、`ACL_HIFLOAT8`。
- 普通 tensor、scalar、scalar type 的 ACL 转换路径保持一致。

## 5. 构建系统梳理

MindIE-SD 构建分为两条主线：OPP/custom op 构建和 PyTorch plugin 构建。

### 5.1 OPP/custom op 构建

入口是：

```text
build/build_ops.sh
build/build_ascendc_ops.sh
csrc/ops/
```

这条链路负责生成并打包：

```text
mindiesd/ops/vendors/...
```

主要产物包括：

- `op_proto`。
- `op_host`。
- `op_api/lib/libcust_opapi.so`。
- `op_impl/ai_core/tbe`。
- metadata 的 `op_impl/cpu` AICPU 产物。

QFA 是 AscendC kernel + opapi 算子，metadata 是 AICPU-only 算子。构建系统需要区分这两类算子，不能假设每个算子都有普通 `op_kernel`。

### 5.2 `op_host_aclnnExc` 的作用

CANN QFA 和 metadata 源码自带手写 public ACLNN wrapper，例如：

```text
aclnnQuantFlashAttn*
aclnnQuantFlashAttnMetadata*
```

MindIE-SD open project 构建如果继续把对应 `*_def.cpp` 放进 `op_host_aclnn`，会自动生成同名 public ACLNN wrapper，最终和手写 opapi 产生 duplicate symbol。

最终处理方式是：

- 手写 opapi 源码继续进入 `opapi` / `libcust_opapi.so`。
- `quant_flash_attn_def.cpp` 放入 `op_host_aclnnExc`。
- `quant_flash_attn_metadata_def.cpp` 放入 `op_host_aclnnExc`。

这样仍然生成 proto / ops info，但不再生成同名 public ACLNN wrapper。

### 5.3 AICPU 构建与打包

metadata 引入了 MindIE-SD 原本不完整覆盖的 AICPU custom op 链路。

最终构建系统需要完成：

- 使用 ARM 交叉编译器编译 metadata AICPU cpp。
- 链接 CANN AICPU context/protobuf 相关静态库。
- 生成 `libtransformer_aicpu_kernels.so`。
- 安装 `cust_aicpu_kernel.json`。
- 将 AICPU so 和 json 放入 custom OPP 的 `op_impl/cpu` 目录。

AICPU so 是 ARM 目标文件，在 x86 构建环境中不能使用 host `strip`。因此 `build/build_ops.sh` 中需要跳过：

```text
op_impl/cpu/aicpu_kernel/impl/*.so
```

### 5.4 Vendor 目录与 runtime registry

CANN `ops-transformer` custom 包使用 transformer 风格 vendor 目录：

```text
vendors/<vendor>_transformer/
```

metadata AICPU registry 对 custom repository 名称和后缀有实际约束。MindIE-SD 因此需要把 metadata AICPU CPU 产物同步到：

```text
mindiesd/ops/vendors/customize_transformer/op_impl/cpu/
```

同时运行时要让该路径出现在 `ASCEND_CUSTOM_OPP_PATH` 中，并优先于普通 vendor 路径。

MindIE-SD 运行时环境入口是：

```text
mindiesd/env.py
```

它负责设置 custom OPP 搜索路径，让 plugin helper 可以找到：

```text
op_api/lib/libcust_opapi.so
op_impl/cpu/config/cust_aicpu_kernel.json
op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so
```

### 5.5 Plugin 构建

plugin 构建入口是：

```text
build/build_plugin.sh
csrc/plugin/
```

它负责生成：

```text
mindiesd/plugin/libPTAExtensionOPS.so
```

该动态库注册 `torch.ops.mindiesd.*`，并在执行时通过 helper 查找 custom opapi 中的 ACLNN 符号。

MindIE-SD 的运行链路因此是：

```text
Python test / user code
  -> torch.ops.mindiesd.*
  -> libPTAExtensionOPS.so
  -> ASCEND_CUSTOM_OPP_PATH
  -> libcust_opapi.so
  -> AscendC kernel / AICPU kernel
```

## 6. 测试与验证入口

QFA 测试位于：

```text
tests/ops/quant_flash_attn/
```

本轮按 CANN new 测试语义更新：

- `quant_flash_attn_golden.py`
- `mx_quant_fp4_tool.py`
- `flash_attention_cpu_golden.py`
- `binary_file_io.py`

删除更新后不再引用的旧 helper：

- `flash_attention_fp32_golden.py`
- `flash_attention_mxfp4_golden.py`

测试脚本保留 CANN 的输入生成、CPU golden、mask/varlen/PA 处理和精度比较逻辑；MindIE-SD 差异集中在：

- 调用 `mindiesd.env.set_environment_variables()`。
- 导入 `mindiesd.layers.register_ops` 触发 plugin 注册。
- 使用 `torch.ops.mindiesd.quant_flash_attn`。
- 使用 `torch.ops.mindiesd.quant_flash_attn_metadata`。

最小 NPU 验证入口：

```bash
python tests/ops/quant_flash_attn/quant_flash_attn_golden.py \
  --b 1 --n2 1 --g 1 --s1 128 --s2 128 --qk_d 128 \
  --q_quant_mode 3 --k_quant_mode 3 --v_quant_mode 3 \
  --q_dtype fp4_e2m1 --kv_dtype fp4_e2m1 \
  --input_layout BSND --kv_storage_mode continue --enable_softmax_lse
```

## 7. 当前最终状态

当前 MindIE-SD 侧已经完成：

- CANN new QFA kernel 同步。
- CANN new metadata AICPU 更新同步。
- QFA 实际依赖 common 文件同步。
- QFA plugin dtype、输出 device、ACLNN 参数顺序适配。
- metadata plugin 无 tensor 参数 dispatch 适配。
- AICPU build/package/runtime vendor 路径适配。
- QFA 测试脚本和 golden helper 更新。

静态验证项：

```bash
python -m compileall -q tests/ops/quant_flash_attn
git diff --check
rg -n "npu_ops_transformer|flash_attention_mxfp4_golden|flash_attention_fp32_golden|fused_infer_attention_score|vector_common" csrc tests
```

其中 `vector_common` 只剩 `copy_gm_to_ub.h` 中 `#if 0` 注释里的非 active mention。

目标环境构建和测试已确认通过。本次迁移最终保留的是 QFA / metadata 及其真实依赖闭包，没有扩大到 FIAS 或未使用 common 文件。
