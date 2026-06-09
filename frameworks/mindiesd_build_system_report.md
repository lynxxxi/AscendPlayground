# MindIE-SD 编译工程分析报告

## 1. 工程整体模型

MindIE-SD 的编译工程由三层组成：

```text
Python wheel 工程
  ├─ CANN custom OPP 工程：csrc/ops
  ├─ PyTorch / torch_npu plugin 工程：csrc/plugin + csrc/CMakeLists.txt
  └─ Python 包：mindiesd
```

这三层的职责不同：

- Python wheel 工程负责总调度和最终交付。
- CANN custom OPP 工程负责生成 CANN runtime 能识别的自定义算子包。
- torch_npu plugin 工程负责把 `torch.ops.mindiesd.*` 调用桥接到 CANN ACLNN 接口。

最终 wheel 中至少包含两类 native 产物：

```text
mindiesd/ops/vendors/**/*
mindiesd/plugin/libPTAExtensionOPS.so
```

运行时链路是：

```text
Python / tests
  -> torch.ops.mindiesd.*
  -> libPTAExtensionOPS.so
  -> dlsym custom aclnn* from libcust_opapi.so
  -> CANN ACLNN runtime
  -> AscendC kernel / AICPU kernel
```

## 2. 顶层 wheel 构建逻辑

入口文件：

```text
pyproject.toml
setup.py
```

`pyproject.toml` 声明 setuptools 构建后端，并把以下内容纳入包数据：

```text
mindiesd/plugin/*.so
mindiesd/ops/**/*
```

`setup.py` 中的 `CustomBuildPy.run()` 是总调度器。它的执行顺序是：

```text
1. 记录 project root 和 build dir
2. 调用 get_python_version()
3. 把 build/ 下脚本 chmod 为只读
4. 如果 csrc/ops 存在，执行 build/build_ops.sh
5. 如果 csrc/plugin 存在，执行 build/build_plugin.sh
6. 清理 build 临时目录
7. 合并 compile_commands.json
8. 从 build/build 拷贝 plugin .so 到 mindiesd/plugin
9. 调用 setuptools 原始 build_py
```

顶层会显式依赖：

```text
torch
torch_npu
```

这不是普通 Python 依赖而已。plugin 构建会读取 PyTorch ABI：

```python
torch.compiled_with_cxx11_abi()
```

然后把结果写入环境变量：

```text
USER_ABI_VERSION
```

CMake 再用它设置：

```text
-D_GLIBCXX_USE_CXX11_ABI=<0|1>
```

目的是保证 `libPTAExtensionOPS.so` 和当前 PyTorch / torch_npu ABI 一致。

## 3. `build/build_ops.sh`：OPP 总入口

`build_ops.sh` 是自定义算子 OPP 的外层总入口。

### 3.1 CANN toolkit 探测

脚本按顺序查找：

```text
ASCEND_TOOLKIT_HOME
/usr/local/Ascend/ascend-toolkit/latest
/home/slave1/Ascend/ascend-toolkit/latest
```

然后要求存在：

```text
${local_toolkit}/python/site-packages/bin/msopgen
```

`msopgen` 是后续 TIK custom project 生成器。

### 3.2 构建顺序

`build_ops()` 的逻辑是：

```text
1. cd build/
2. rm -rf vendors
3. source build_ascendc_ops.sh -n <op list> -c <soc list>
4. source build_tik_ops.sh
5. sync_aicpu_ops_to_transformer_vendor
6. 删除各 vendor 下的 bin 目录
7. strip 普通 .so，跳过 AICPU kernel so
8. copy_ops 到 mindiesd/ops/vendors
```

当前 AscendC open-project 编译算子列表：

```text
laser_attention
la_preprocess
ada_block_sparse_attention
sparse_block_estimate
quant_flash_attn
quant_flash_attn_metadata
```

当前目标芯片列表：

```text
ascend910
ascend910b
ascend910_93
ascend950
```

### 3.3 删除和复制逻辑

每次 OPP 构建前删除：

```text
build/vendors
```

构建后删除：

```text
build/vendors/aie_ascendc/bin
build/vendors/customize/bin
build/vendors/customize_transformer/bin
```

目的：这些 `bin` 是安装脚本或临时执行文件，不是 wheel 运行时必须内容。

随后扫描：

```bash
find "${current_script_dir}/vendors" -name "*.so"
```

对普通 `.so` 执行 `strip`，但跳过：

```text
*/op_impl/cpu/aicpu_kernel/impl/*.so
```

原因：AICPU kernel so 是 ARM 目标文件，在 x86 host 上用 host `strip` 会失败。

最后 `copy_ops()` 会清空并重建：

```text
mindiesd/ops/vendors
```

再把：

```text
build/vendors/*
```

逐个目录复制进去。

## 4. `build/build_ascendc_ops.sh`：AscendC open-project 构建

这个脚本是 `csrc/ops` 的 CMake 入口。

### 4.1 CANN 环境和编译器

脚本会从以下变量推导 CANN 根目录：

```text
ASCEND_HOME_PATH
ASCEND_OPP_PATH
$HOME/Ascend/ascend-toolkit/latest
/usr/local/Ascend/ascend-toolkit/latest
/usr/local/Ascend/latest
```

然后执行：

```bash
source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
```

并查找：

```text
bisheng
```

`bisheng` 是 AscendC 编译相关工具链入口。脚本还可以用 `ccache` 包装 `bisheng`，生成临时 wrapper：

```text
build/gen_bisheng_dir/bisheng
```

### 4.2 CMake 配置

核心配置命令是：

```bash
cmake -S csrc/ops -B build \
  -DBUILD_OPEN_PROJECT=ON \
  -DASCEND_COMPUTE_UNIT=<soc list> \
  -DASCEND_OP_NAME=<op list> \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=<cann path> \
  -DCHECK_COMPATIBLE=true \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

`BUILD_OPEN_PROJECT=ON` 表示使用 CANN open project 形态构建 custom OPP 包。

### 4.3 构建目标

脚本中的目标函数：

```text
build_package -> build custom_aicpu_kernels; build package
build_host    -> build_package
build_kernel  -> build ops_kernel
```

实际默认执行：

```text
build_package
build_and_install
clean_build_dirs
```

也就是：

```text
1. 先构建 AICPU 聚合 target
2. 再构建 package target
3. 找到 CANN-custom_ops-*.run
4. 安装到 build/ 目录
5. 保存 compile_commands_ascendc.json
6. 删除 build/build
```

## 5. `csrc/ops` CMake 工程设计

`csrc/ops` 是 CANN custom OPP 工程。它依赖 CANN 的 open-project CMake 模板、`op_build` 工具、AscendC util 脚本和 runtime 库。

### 5.1 顶层选项和路径

顶层定义：

```cmake
BUILD_OPEN_PROJECT = ON
ASCEND_COMPUTE_UNIT = ascend910b
ASCEND_OP_NAME = ALL
VENDOR_NAME = aie_ascendc
```

`cmake/config.cmake` 中设置关键路径：

```text
ASCEND_IMPL_OUT_DIR   = build/impl
ASCEND_BINARY_OUT_DIR = build/binary
ASCEND_AUTOGEN_DIR    = build/autogen
OP_BUILD_TOOL         = ${ASCEND_CANN_PACKAGE_PATH}/tools/opbuild/op_build
ASCEND_PROJECT_DIR    = CANN ascend_project 或 op_project_templates/ascendc/customize
ASCEND_CMAKE_DIR      = ${ASCEND_PROJECT_DIR}/cmake
```

安装路径设计为：

```text
packages/vendors/${VENDOR_NAME}/...
```

由于 `VENDOR_NAME=aie_ascendc`，主产物先进入：

```text
packages/vendors/aie_ascendc
```

### 5.2 prepare 阶段

配置阶段会执行：

```bash
cmake/scripts/prepare.sh
```

传入：

```text
source dir
build dir
CANN package path
autogen dir
binary out dir
impl out dir
op_build tool
ascend cmake dir
tiling keys
compile options
op debug config
ASCEND_OP_NAME
ASCEND_COMPUTE_UNIT
```

这个阶段为后续生成 ops-info、compile cmd、dynamic py 和 binary kernel 编译脚本做准备。

### 5.3 扫描算子目录

函数：

```cmake
op_add_subdirectory(OP_LIST OP_DIR_LIST)
```

扫描路径：

```text
csrc/ops/**/op_host/CMakeLists.txt
```

对每个匹配项，取父目录作为算子目录，目录名作为算子名。

它会按 `ASCEND_OP_NAME` 过滤：

- `ALL` / `all`：全部纳入。
- 指定列表：只纳入列表里的算子。

最终得到：

```text
OP_LIST
OP_DIR_LIST
```

当前目录形态大致是：

```text
ada_block_sparse_attention : op_host, op_kernel
laser_attention            : op_host, op_kernel
la_preprocess              : op_host, op_kernel
quant_flash_attn           : op_host, op_kernel, op_api
quant_flash_attn_metadata  : op_host, op_kernel, op_kernel_aicpu
sparse_block_estimate      : op_host, op_kernel
common                     : op_host, op_kernel, CMakeLists.txt
utils                      : CMakeLists.txt
tik                        : TIK 旧链路，不进 op_add_subdirectory
```

### 5.4 依赖目录扫描

函数：

```cmake
op_add_depend_directory(OP_LIST ... OP_DIR_LIST ...)
```

会查找变量：

```text
<op_name>_depends
```

如果某个算子声明了依赖目录，并且依赖目录存在：

```text
<depend>/op_host/CMakeLists.txt
```

则把依赖 op_host 也加入 CMake。

目的：比如某些算子依赖 `common/op_host` 的 proto、tiling helper 或公共注册逻辑，但 common 本身不是直接编译入口时，仍能参与 host 侧构建。

### 5.5 `op_host` 进入方式

扫描完成后：

```cmake
foreach (OP_DIR ${OP_DIR_LIST})
    add_subdirectory(${OP_DIR}/op_host)
endforeach ()
```

每个算子的 `op_host/CMakeLists.txt` 会把源文件加入对应 target，例如：

- `op_host_aclnn`
- `op_host_aclnnInner`
- `op_host_aclnnExc`
- `opapi`
- `opsproto`
- `optiling`

这决定了后续是否生成 public ACLNN wrapper、inner wrapper、proto、tiling 和手写 opapi。

### 5.6 AICPU 子目录扫描

AICPU 扫描逻辑：

```cmake
add_custom_target(custom_aicpu_kernels)

foreach (OP_DIR ${OP_DIR_LIST})
    if (EXISTS "${OP_DIR}/op_kernel_aicpu/CMakeLists.txt")
        add_subdirectory(${OP_DIR}/op_kernel_aicpu)
    endif ()
endforeach ()
```

目的：

- 发现 metadata 这种 AICPU-only 或 AICPU 辅助算子的 CPU kernel。
- 让每个 AICPU 子工程生成自己的 so/json。
- 统一挂到 `custom_aicpu_kernels`，方便 `build_package` 先构建。

## 6. `csrc/ops` 核心 target

### 6.1 `op_host_aclnn`

输入：各算子的 `*_def.cpp`。

作用：交给 CANN `op_build` 自动生成：

```text
autogen/aclnn_<op>.cpp
autogen/aclnn_<op>.h
autogen/<op>_proto.cpp
autogen/<op>_proto.h
```

适用场景：算子没有手写 public ACLNN wrapper，希望 CANN 自动生成。

### 6.2 `op_host_aclnnInner`

输入：inner def。

作用：生成：

```text
autogen/inner/aclnnInner_<op>.cpp
autogen/inner/<op>_proto.cpp
```

适用场景：算子有 public wrapper 调 inner wrapper，或者 CANN 两段式接口需要拆出内部执行逻辑。

### 6.3 `op_host_aclnnExc`

作用：只生成 proto / ops info，不生成 public ACLNN wrapper。

实现方式：调用 `op_build` 时使用：

```text
OPS_ACLNN_GEN=0
OPS_PROJECT_NAME=aclnnExc
```

适用场景：QFA、metadata 这类已经迁入 CANN 手写 public ACLNN wrapper 的算子。如果放进 `op_host_aclnn`，会自动生成同名 `aclnn*`，和手写版本重复。

### 6.4 `ops_aclnn`

这是一个 static library，收集自动生成的：

```text
aclnn_*.cpp
aclnnInner_*.cpp
```

然后以 whole-archive 方式链接进：

```text
opapi
```

### 6.5 `opapi`

目标输出名：

```text
libcust_opapi.so
```

安装路径：

```text
packages/vendors/${VENDOR_NAME}/op_api/lib
```

它包含：

- 自动生成的 ACLNN wrapper。
- 手写 ACLNN wrapper。
- 手写 inner opapi。
- CANN `-lopapi`、`nnopbase`、`profapi`、`ge_common_base` 等依赖。

运行时 plugin 会从这里 `dlsym`：

```text
aclnnXXXGetWorkspaceSize
aclnnXXX
InitHugeMemThreadLocal
UnInitHugeMemThreadLocal
ReleaseHugeMem
```

### 6.6 `opsproto`

目标输出名：

```text
libcust_opsproto_rt2.0.so
```

安装路径：

```text
packages/vendors/${VENDOR_NAME}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
```

作用：

- 注册算子 proto。
- 提供 infer shape / attr / dtype 等 host 侧定义。
- 为 CANN runtime 识别 custom op 提供元信息。

同时会安装生成的 proto header：

```text
packages/vendors/${VENDOR_NAME}/op_proto/inc
```

### 6.7 `optiling`

目标输出名：

```text
libcust_opmaster_rt2.0.so
```

安装路径：

```text
packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
```

同时创建兼容软链：

```text
op_impl/ai_core/tbe/op_tiling/liboptiling.so
```

作用：承载 tiling host 逻辑。

### 6.8 `generate_adapt_py`

调用 CANN 脚本：

```text
ascendc_impl_build.py
```

输入：

```text
autogen ops-info dir
autogen/inner ops-info dir
autogen/exc ops-info dir
```

输出 dynamic py：

```text
build/impl/dynamic/<op>.py
```

安装到：

```text
packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
```

这些 dynamic py 会被 kernel 编译脚本使用。

### 6.9 `generate_compile_cmd`

每个 compute unit 都会建一个 target：

```text
generate_compile_cmd_<soc>
```

调用 CANN 脚本：

```text
ascendc_bin_param_build.py
```

输入：

```text
aic-<soc>-ops-info.ini
inner/aic-<soc>-ops-info.ini
exc/aic-<soc>-ops-info.ini
```

输出：

```text
build/binary/<soc>/gen/*.sh
```

这些 `.sh` 是实际 AscendC binary 编译脚本。

### 6.10 `generate_ops_info`

每个 compute unit 都会建一个 target：

```text
generate_ops_info_<soc>
```

调用：

```text
parse_ini_to_json.py
```

输出：

```text
autogen/aic-<soc>-ops-info.json
```

并复制 / 安装到：

```text
packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/config/<soc>
```

### 6.11 `ops_kernel`

每个 compute unit 会调用：

```cmake
add_bin_compile_target(COMPUTE_UNIT <soc> OP_INFO ${OP_DIR_LIST})
```

它会扫描：

```text
<op_dir>/op_kernel/*
build/binary/<soc>/gen/*.sh
```

对于存在 `op_kernel` 文件的算子，建立：

```text
<op>_<soc>
<op>_<soc>_src_copy
<op>_<soc>_py_copy
<op>_<soc>_mkdir
<op>_<soc>_<index>
```

关键逻辑：

- `add_ops_src_copy` 把 `op_kernel` 拷贝到 `build/binary/<soc>/src/<op>`。
- 如果算子有 `<op>_depends`，会额外拷贝依赖算子的 `op_kernel`。
- 把 dynamic py 拷贝到 `build/binary/<soc>/src/<op>/<op_type>.py`。
- 执行 `build/binary/<soc>/gen/*.sh` 生成 binary。
- 生成 `binary_info_config.json`。
- 安装 binary 和 config 到 `op_impl/ai_core/tbe/kernel`。

### 6.12 `package`

CPack 配置：

```text
CPACK_GENERATOR = External
CPACK_PACKAGE_FILE_NAME = CANN-custom_ops-${CANN_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run
CPACK_EXTERNAL_PACKAGE_SCRIPT = ${ASCEND_CMAKE_DIR}/makeself.cmake
```

这会生成 CANN custom op `.run` 包。`build_ascendc_ops.sh` 随后把它安装到 `build/`，形成：

```text
build/vendors/aie_ascendc/...
```

## 7. AICPU 工程设计

metadata 的 AICPU CMake 在：

```text
csrc/ops/quant_flash_attn_metadata/op_kernel_aicpu/CMakeLists.txt
```

它扫描：

```text
op_kernel_aicpu/*.json
op_kernel_aicpu/*_aicpu*.cpp
```

如果没有 `*_aicpu*.cpp`，直接 return。

它生成：

```text
libtransformer_aicpu_kernels.so
cust_aicpu_kernel.json
```

使用的 CANN 编译器接口：

```text
${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
```

使用的 CANN include / lib 接口包括：

```text
${ASCEND_CANN_PACKAGE_PATH}/include
${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu
${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context
${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu_common/context
libaicpu_context.a
libbase_ascend_protobuf.a
```

安装到：

```text
packages/vendors/${VENDOR_NAME}/op_impl/cpu/aicpu_kernel/impl
packages/vendors/${VENDOR_NAME}/op_impl/cpu/config
```

然后 `build_ops.sh` 再把：

```text
vendors/aie_ascendc/op_impl/cpu
```

同步到：

```text
vendors/customize_transformer/op_impl/cpu
```

原因：CANN `ops-transformer` 类 AICPU custom registry 对 repository 名称/后缀有识别习惯，`customize_transformer` 更接近 CANN transformer custom 包形态。

## 8. TIK 工程设计

TIK 构建入口：

```text
build/build_tik_ops.sh
```

它不是 `csrc/ops/CMakeLists.txt` 的一部分，而是另起一套 CANN `msopgen` 临时工程。

流程：

```text
1. 删除 build/custom_project_tik
2. msopgen gen 生成空 custom project
3. 删除 demo 生成的 onnx/proto/tbe/ini 文件
4. 从 csrc/ops/tik 拷贝真实文件进去
5. 修改 build.sh 和 CMakeLists.txt
6. bash build.sh
7. 安装 .run 到 build/vendors
```

扫描 / 拷贝的源路径：

```text
csrc/ops/tik/framework/onnx_plugin
csrc/ops/tik/op_proto
csrc/ops/tik/tbe/impl
csrc/ops/tik/tbe/op_info_cfg/ai_core/ascend310p
```

产物 vendor：

```text
vendors/customize
```

所以 `customize` 与 `aie_ascendc` 的来源不同：

- `aie_ascendc`：`csrc/ops` open-project。
- `customize`：`msopgen` 生成的 TIK custom project。

## 9. Plugin 工程设计

plugin 构建入口：

```text
build/build_plugin.sh
csrc/CMakeLists.txt
csrc/plugin/*.cpp
```

### 9.1 构建流程

`build_plugin.sh` 做：

```text
1. 定位 ASCEND_INSTALL_PATH / ASCEND_HOME_PATH / ascend-toolkit/latest
2. source CANN setenv.bash
3. python3 查询 torch.compiled_with_cxx11_abi()
4. export USER_ABI_VERSION
5. rm -rf build
6. cmake -B build ../csrc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
7. cmake --build build -j
8. find build -name "*.so" 复制到 mindiesd/plugin
```

### 9.2 编译目标

`csrc/CMakeLists.txt` 只构建一个核心动态库：

```text
PTAExtensionOPS
```

输出：

```text
libPTAExtensionOPS.so
```

源文件包括：

```text
register_ops.cpp
find_op_path.cpp
la.cpp
adalayernorm.cpp
la_preprocess.cpp
rainfusionattention.cpp
ada_block_sparse_attention.cpp
sparse_block_estimate.cpp
layernorm.cpp
block_sparse_attention.cpp
quant_flash_attn.cpp
quant_flash_attn_metadata.cpp
```

### 9.3 torch_npu / PyTorch 暴露给 plugin 的接口

编译期路径来自 Python site-packages：

```text
torch/include
torch/include/torch/csrc/api/include
torch_npu/include
torch_npu/include/third_party/acl/inc
torch_npu/include/third_party/hccl/inc
torch/lib
torch_npu/lib
```

链接库：

```text
c10
torch
torch_cpu
torch_npu
```

代码里直接使用的 torch / torch_npu 接口主要包括：

```cpp
TORCH_LIBRARY
TORCH_LIBRARY_IMPL
at::Tensor
at::empty
at_npu::native::empty_with_format
at_npu::native::get_npu_format
at_npu::native::OpCommand
c10_npu::getCurrentNPUStream
torch_npu::utils::get_npu_device_type
```

这些接口的作用：

- `TORCH_LIBRARY`：定义 `torch.ops.mindiesd` namespace 和 schema。
- `TORCH_LIBRARY_IMPL(..., PrivateUse1)`：给 NPU backend 注册实现。
- `TORCH_LIBRARY_IMPL(..., CatchAll)`：给无 tensor 参数的调用提供 fallback。
- `empty_with_format` / `get_npu_format`：分配 NPU 输出 tensor 并保持 NPU format。
- `getCurrentNPUStream`：取 torch_npu 当前 stream，传给 ACLNN 第二段执行接口。
- `OpCommand::SetCustomHandler`：把自定义 ACLNN 调用塞进 torch_npu 的执行框架。
- `get_npu_device_type`：构造 NPU TensorOptions，尤其用于 metadata 这种无必选输入 tensor 的输出分配。

### 9.4 CANN 暴露给 plugin 的接口

编译期 include / lib：

```text
${ASCEND_PATH}/include
${ASCEND_PATH}/include/aclnn
${ASCEND_PATH}/lib64/libascendcl.so
${ASCEND_PATH}/lib64/libnnopbase.so
${ASCEND_PATH}/lib64/libopapi.so
```

代码里使用的 ACL / ACLNN 类型和函数包括：

```cpp
aclTensor
aclScalar
aclIntArray
aclBoolArray
aclTensorList
aclOpExecutor
aclrtStream
aclCreateTensor
aclCreateScalar
aclCreateIntArray
aclCreateTensorList
aclDestroyTensor
aclGetRecentErrMsg
```

plugin 不直接链接 custom `libcust_opapi.so`。它运行时通过：

```cpp
dlopen
dlsym
```

查找这些符号：

```text
aclnn<OpName>GetWorkspaceSize
aclnn<OpName>
InitHugeMemThreadLocal
UnInitHugeMemThreadLocal
ReleaseHugeMem
```

搜索顺序由 `find_op_path.cpp` / `pytorch_npu_helper.h` 控制：

```text
1. ASCEND_CUSTOM_OPP_PATH 中每个 vendor 的 op_api/lib/libcust_opapi.so
2. ASCEND_OPP_PATH/vendors/config.ini 中 load_priority 指定 vendor 的 libcust_opapi.so
3. 系统 libopapi.so
```

## 10. 为什么会生成这些文件

这些产物不是 MindIE-SD 任意设计的，而是由 CANN custom OPP 和 torch_npu plugin 两套机制共同决定。

### 10.1 `libcust_opapi.so`

来源 target：

```text
opapi
```

原因：CANN ACLNN custom op 需要一个承载 `aclnn*` 符号的 opapi 动态库。plugin 运行时通过 `dlsym` 查找它。

### 10.2 `libcust_opsproto_rt2.0.so`

来源 target：

```text
opsproto
```

原因：CANN runtime 需要 custom op 的 proto、shape inference、attr 定义和注册信息。

### 10.3 `libcust_opmaster_rt2.0.so` / `liboptiling.so`

来源 target：

```text
optiling
optiling_compat
```

原因：AscendC kernel 运行前需要 host tiling 逻辑计算 tiling data。兼容软链是为了满足不同 CANN runtime 查找布局。

### 10.4 `op_impl/ai_core/tbe/kernel/<soc>`

来源 target：

```text
ops_kernel
```

原因：AscendC kernel 需要按 SOC 编译成 binary，并生成对应 config。

### 10.5 `op_impl/ai_core/tbe/config/<soc>`

来源：

```text
generate_ops_info
ops_config
```

原因：CANN runtime 需要知道某个 SOC 下有哪些 custom op、kernel binary、tiling key、dtype/layout 支持等元信息。

### 10.6 `*_impl/dynamic/<op>.py`

来源：

```text
generate_adapt_py
```

原因：CANN AscendC binary 编译脚本需要 dynamic py 作为 op impl 入口，驱动 kernel build。

### 10.7 `libtransformer_aicpu_kernels.so` 和 `cust_aicpu_kernel.json`

来源：

```text
custom_aicpu_kernels
op_kernel_aicpu/CMakeLists.txt
```

原因：metadata 这类 AICPU 算子不生成 ai_core kernel binary，而是通过 CPU kernel so 和 registry json 被 CANN runtime 调度。

### 10.8 `libPTAExtensionOPS.so`

来源：

```text
csrc/CMakeLists.txt -> PTAExtensionOPS
```

原因：PyTorch 不会直接调用 CANN OPP。需要一个 torch extension 注册 `torch.ops.mindiesd.*`，负责参数转换、输出 tensor 分配、workspace 分配和 ACLNN 执行。

## 11. 运行时环境设计

MindIE-SD 的环境设置在：

```text
mindiesd/env.py
build/set_env.sh
```

核心变量：

```text
ASCEND_CUSTOM_OPP_PATH
```

当前顺序：

```text
mindiesd/ops/vendors/customize_transformer
mindiesd/ops/vendors/aie_ascendc
mindiesd/ops/vendors/customize
<old ASCEND_CUSTOM_OPP_PATH>
```

这个顺序的含义：

- 先让 transformer 风格 AICPU registry 生效。
- 再让主 AscendC custom OPP 生效。
- 再兼容 TIK `customize` 产物。
- 最后保留用户原有 custom OPP。

## 12. 构建产物关系

| 产物目录 | 来源 | 内容 | 使用方 |
|---|---|---|---|
| `mindiesd/ops/vendors/aie_ascendc` | `csrc/ops` open-project | opapi、proto、tiling、AscendC kernel、初始 AICPU CPU impl | CANN runtime / plugin dlsym |
| `mindiesd/ops/vendors/customize` | `build_tik_ops.sh` | TIK/TBE custom project 产物 | CANN runtime |
| `mindiesd/ops/vendors/customize_transformer` | 从 `aie_ascendc/op_impl/cpu` 同步 | AICPU registry json 和 CPU kernel so | CANN AICPU custom registry |
| `mindiesd/plugin/libPTAExtensionOPS.so` | `csrc/plugin` | torch dispatcher 实现和 ACLNN 调用桥 | PyTorch / torch_npu |
| `dist/mindiesd-*.whl` | setuptools | Python 包 + native 产物 | 用户安装 |

## 13. 一句话总结

MindIE-SD 编译工程的核心设计是：用 CANN 工具链生成 custom OPP，用 torch_npu plugin 把 PyTorch dispatcher 接到这些 custom ACLNN opapi，再由 Python wheel 把两者打包到同一个 `mindiesd` 分发件中。
