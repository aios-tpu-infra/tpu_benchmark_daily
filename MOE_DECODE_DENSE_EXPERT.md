# Dense-expert decode MoE 实验实现

已经实现并接到原来的 `fused_moe_func`，函数签名、输入权重布局和输出 `[tokens, hidden]` / BF16 均保持不变。当前是可切换的实验实现，默认仍为 `standard`。LLO 分析发现 activation scale 布局转换开销；已把 scale scratch 从 `[M,Kblocks,128]` 改为 `[Kblocks,M,128]`，接入同一入口并完成服务端到端验证。

完整设计、存储量、LLO 布局修复、正确性与端到端压测口径见单独文档：[Dense expert MoE：设计与性能验证](third_party/torchtpu-vllm/docs/developers_guide/dense_expert_moe.md)。

## 切换入口

在启动 Python/服务进程前设置：

```bash
export TPU_MOE_DECODE_IMPL=dense_expert  # 新实现
# export TPU_MOE_DECODE_IMPL=standard  # 原实现，默认值
```

开关在模块导入时固定，服务切换需要重启/重新编译，不支持在同一个已编译图上动态切换。上层 adapter、router/top-k、shared expert 和 DP/TP 通信均无需改调用。

UT runner 增加了同名选择参数，仍调用原入口：

```bash
TPU_SKIP_MDS_QUERY=true .venv/bin/python scripts/bench_moe_decode_ut.py \
  --output runs/moe-dense-standard-new --moe-impl standard \
  --pools 16 32 64 --seed 20260921 --iterations 200 --profile-steps 10

TPU_SKIP_MDS_QUERY=true .venv/bin/python scripts/bench_moe_decode_ut.py \
  --output runs/moe-dense-candidate-new --moe-impl dense_expert \
  --pools 16 32 64 --seed 20260921 --iterations 200 --profile-steps 10

.venv/bin/python scripts/analyze_moe_ut_trace.py runs/moe-dense-standard-new
.venv/bin/python scripts/analyze_moe_ut_trace.py runs/moe-dense-candidate-new
.venv/bin/python scripts/compare_moe_ut.py \
  runs/moe-dense-standard-new runs/moe-dense-candidate-new \
  --output runs/moe-dense-comparison-new.json
```

这些 TPU 命令顺序执行，使用独占的 8 个 chiplet。比较工具会先检查 hidden、router、top-k IDs/weights、专家 token 计数完全相同，再比较数值与设备时间。

## Kernel 流程

1. 原入口继续处理全局到本地 expert ID 的映射、非本地路由屏蔽和 BF16 top-k weight 转换。
2. 构造 `[token, local_expert]` 路由系数和 active bitmap；没有对 token 排序。
3. 单个 Pallas kernel 将 active bitmap 在 SMEM 中压成去重的本地专家列表。每个专家最多出现一次，相当于 token/top-k 遍历中的 visited 集合；以本地专家顺序执行。
4. hidden 只在进入 kernel 时加载，并预先按原 GMM 的 512-wide block 量化成 FP8；在所有专家之间复用。
5. 每个命中专家的 gate/up/down 权重只 DMA 一次。双缓冲使下一位专家的权重预取与当前专家的计算重叠；不加载未命中的专家权重。
6. 对全部 token 计算 gate/up、SiLU、down，然后按路由系数直接累加到 FP32 VMEM 输出；零系数的位置显式屏蔽。
7. 所有专家完成后，将累加输出转成 BF16，写回 HBM 一次。

单专家 intermediate 和量化后的 intermediate 均复用 VMEM scratch。不创建 `[expert, token, hidden]` 输出，也不把每专家输出写回 HBM。256×4096 的 FP32 累加 tensor 为 4 MiB；包含权重双缓冲、输入/输出、中间结果、量化 scale 等的显式 VMEM 占用约 34.6 MiB，另留编译器临时空间。此前测试使用 60 MiB 编译预算；当前已移除固定预算，由编译器判断资源是否足够。

保留 FP8 权重与动态 activation quantization，FP32 MXU dot 结果按原 GMM 路径转 BF16、应用 scale 并累加。最终专家加权累加使用 FP32。FP8/BF16 舍入和融合变化使它与原实现不是逐位相同。

## 当前支持范围与回退

新路径只在静态检查通过时选择：

- BF16 hidden，原生 `float8_e4m3fn` 权重；FP32 每输出通道 scale，支持入口归一化后的 `[E,1,1,N]` 布局。
- SiLU、无 expert bias、无 packed quantization；`rhs_quant_dtype` 为 `None` 或相同的 FP8 dtype。
- M/H/I/E 为正；M 是 16 的倍数，H/I 是 512 的倍数。无尺寸白名单和 M/E 上限。
- `skip_padded_tokens` 两种取值都支持；系数矩阵的专家维向上补齐到 128 的倍数。
- 不再检查固定 VMEM 容量，不设置固定 60 MiB 预算，由编译器判断资源。

由用户仅在 decode 服务上显式设置 `TPU_MOE_DECODE_IMPL=dense_expert`，prefill 服务保持 `standard`。不根据 M 自动推断阶段；去掉 M 上限后，混合服务的长 prefill 不再自动回退。
未启用开关或功能/对齐不支持时保留原 GMM。选中新实现后，编译资源错误直接传播，没有异常重试 fallback。

## 放宽形状限制的验证

新增 CPU 准入和 Pallas 解释模式验证覆盖 M=272、H/I=1536、E=129，以及专家 127/128、重复/无效 ID、零权重行和显式实现选择。解释模式仅用于测试，不作为运行时分支；资源异常传播测试确认不会重试原 GMM。
本轮未重新进行 TPU 编译或 E2E 压测，当前运行服务仍加载此前版本。下面的 203 项 TPU 测试及性能结果来自限制放宽前的 `3477a1f`。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  third_party/torchtpu-vllm/tests/kernels/test_moe_dense_expert_shapes.py
```

## 布局修复与新增验证

- `xs_ref` / `asc_ref` 的 K-block 维移到最前面，固定 K-block 后读取连续的 `[tokens,128]` 平面，算术顺序不变。
- pool32 的隔离实验输出逐位一致，kernel 490.57 → 158.65 μs；整层 XLA Ops 相比原 GMM 305.64 → 264.03 μs。详见 `runs/moe-llo-20260921T060341Z/ANALYSIS.md`。
- 集成后的 kernel AST 与测量实验完全一致（仅新增解释注释）；重新通过 203 项 TPU 测试。
- 新 seed 20260922 的全尺寸 pool16/32/64 参考数值校验全部通过，TP 副本逐位一致。
- 服务启动脚本可设置 `TPU_MOE_DECODE_IMPL`，新实现使用独立缓存目录，默认仍是 standard。
- 已完成无 prefix cache 的原 benchmark 脚本、有 prefix cache 的预填充复测，两种协议分别对照 standard / dense_expert。完整报告：[服务 E2E 对照](runs/moe-e2e-20260921T065541Z/README.md)。
- 无缓存单轮：峰值吞吐 5739 → 5995 tok/s（+4.46%），峰值窗口 TPOT 44.91 → 42.79 ms；两边峰值窗口均 248 路，均无持续满 256 路窗口。包含 prefill 的整轮吞吐下降 1.04%，不能据此声称整体请求收益。
- 有缓存各三轮：满 256 路区间平均吞吐中位数 5917.03 → 6294.73 tok/s（+6.38%），满 batch 窗口 TPOT p50 中位数 43.294 → 40.682 ms（−6.03%）；每个请求均命中 63360 token 缓存。包含剩余 prefill 的整轮吞吐中位数提高 4.28%。
- 用户追加的简单问答检查已执行：新实现两种配置及有缓存原实现各 8 题，覆盖 4 个 DP rank，回答文本全部一致，未发现乱码或异常重复；不替代完整模型精度评测。原实现无缓存基线在追加检查前已完成。
- 新实现有缓存服务保留运行；所有本轮测试未开启采集、未发布结果。

## 2026-09-21 布局修复前的配对测试（历史结果）

配置与前一轮一致：全局 256 token、DP4/TP2/EP8、top-k=10、固定种子 20260921、合成权重/输入；不是模型 checkpoint 的真实 activation replay。两次运行保存的输入和路由逐值相同。原路径三组输出的 SHA256 与前一轮 baseline 完全一致。

下表为 TPU:0 的 TensorCore XLA Ops 耗时之和，10 个采集 forward 的均值，单位 μs；不包含 Python dispatch，不重复累加 Async/SparseCore lane。

| 每 rank 实际命中专家 | 原实现 | 新实现 | 时延变化 | 相对原输出 L2 误差 |
|---|---:|---:|---:|---:|
| 16 | 238.09 | 356.05 | +49.5% | 0.268% |
| 32 | 305.86 | 594.80 | +94.5% | 0.267% |
| 63–64 | 438.47 | 1071.95 | +144.5% | 0.262% |

pool32 中新融合 kernel 本身为 490.47 μs，外部算子为 104.33 μs。原实现两个 GMM 合计 153.35 μs，其余算子为 152.50 μs。两种 kernel 的范围不同：新 kernel 已包含 activation、mask 和加权 combine，不能把这两个 kernel 时间直接当作同范围的 GMM 对比。

完整 device module 跨度另有记录：pool32 原实现为 351.06 μs，新实现为 646.93 μs，同样发生回退。同步 host benchmark 单独保留，不混入上表。

所有 8 个 rank、每组 10 个 forward 的 trace 检查通过：原路径每层 4 个 sort，新路径 0 个 sort，也不再出现独立的 blockwise one-hot unpermute kernel。虽然流程中的排序和独立还原阶段已去掉，当前融合实现的耗时增加仍超过其收益。**此表是布局修复前的历史回退结果。** 继续调优时需要检查融合 kernel 内部的 DMA/MXU/VPU 调度和计算开销。

三组最坏 token 的相对 L2 误差为 0.469%–0.484%，低于既有 UT 的门限（整体 0.8%、最坏 token 1.5%）；TP 副本逐位相同。这组历史测试当时尚未执行完整模型输出精度或服务吞吐测试；后续服务测试见上方 E2E 验证。

## 测试与产物

**限制放宽前的硬件测试：203 passed**（包含 13 项新 kernel 测试与 190 项原 MoE core 回归测试）；Ruff 和 diff whitespace 检查通过。三组 DP4/TP2 全尺寸 benchmark 校验也均通过。

Kernel 测试覆盖同一公共入口切换、非本地路由、单专家、空 rank、零权重、重复 expert ID，以及不支持的参数和原 M 上限回退；当前测试已改为验证对齐回退。硬件测试命令：

```bash
TPU_SKIP_MDS_QUERY=true TPU_MOE_OWNER_OUTPUT_MODE=on \
LIBTPU_INIT_ARGS='--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_scoped_vmem_limit_kib=65536' \
.venv/bin/python -m pytest -q \
  third_party/torchtpu-vllm/tests/kernels/test_moe_dense_expert.py \
  third_party/torchtpu-vllm/tests/layers/core/test_fused_moe_gmm.py
```

- Kernel：`third_party/torchtpu-vllm/src/vllm_torchtpu/kernels/megablox/moe_dense_expert.py`
- 入口：`third_party/torchtpu-vllm/src/vllm_torchtpu/layers/core/fused_moe_gmm.py`
- 开关：`third_party/torchtpu-vllm/src/vllm_torchtpu/envs.py`
- 配对结果：`runs/moe-dense-expert-20260921/comparison.json`
- 两套完整输入、输出、8-rank trace、每步时间和 StableHLO：同目录的 `standard/`、`dense_expert/`。
- 测试日志：`runs/moe-dense-expert-20260921/final_pytest.log`。

服务端到端验证使用 `runs/moe-e2e-20260921T065541Z/start_server.sh`；模型权重及默认 MoE 选择仍保持原配置。当前测试未发布日报。
