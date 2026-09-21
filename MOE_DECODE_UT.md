# Qwen3.5 decode MoE UT 基线（2026-09-21）

已在 8 个 TPU7x chiplet 上跑通原始 MoE 路径，用于后续“逐专家计算全部 token、直接累加输出”实现的对照。这里只建立基线，没有替换生产 kernel，也没有发布报告。

## 配置与范围

- 全局 256 个 decode token，DP4 / TP2 / EP8。每个 DP 输入 64 个不同 token，TP 两个成员持有副本；DP all-gather 后，每个 EP rank 可见全部 256 个 token。
- hidden=4096，512 个 routed expert，top-k=10；每 rank 64 个 expert，每 expert intermediate=1024。
- Routed expert 使用 FP8 权重、每输出通道 FP32 scale、BF16 activation；GMM 内部执行原有 512-wide 动态 FP8 activation quantization 和 BF16 累加。
- 保留 router projection、softmax/top-k、原始 sort/one-hot permute、GMM1/SiLU/GMM2、blockwise one-hot unpermute、shared expert/gate、DP reduce-scatter 和 TP all-reduce。
- Shared expert intermediate=1024，TP2 后每 rank 512。
- 使用 serving 对应的 `torchtpu-vllm` main revision：`eaf9756f357564f886d8597528383a57e09fb3bf`。
- **输入和权重为固定种子 20260921 生成的合成数据，没有加载模型 checkpoint。** 之前服务 trace 没有保存 activation 和 top-k IDs，不能声称逐值复现原请求。

通过 router logits bias 将候选专家限制到每 rank 的前 16 / 32 / 64 个。最后一种不限制候选池。三个 case 共用同样的 hidden、router 和专家权重；各自保存实际 top-k IDs/weights 和专家 token 计数。这样可以区分专家命中数量与 kernel 优化的影响。

## 设备结果

以下是 **TPU:0 的 TensorCore XLA Ops 耗时之和**，每个 UT case 采集 10 次 forward 后取均值，单位 μs。与之前服务分析使用同一种算子计时口径；不重复计入 Async/SparseCore lane。

| case | 每 rank 实际命中专家 | GMM1 | GMM2 | 两个 GMM | 其他算子 | 算子总计 |
|---|---|---:|---:|---:|---:|---:|
| pool16 | 16 | 55.80 | 30.68 | 86.48 | 152.32 | 238.80 |
| pool32 | 32 | 96.71 | 56.53 | 153.23 | 152.54 | 305.78 |
| pool64 | 63–64 | 178.51 | 107.79 | 286.30 | 151.68 | 437.98 |
| 原服务真实路由，60 层均值 | 未记录 | 87.52 | 53.42 | 140.94 | 170.17 | 311.12 |

“其他算子”包含路由、重排/还原、shared expert 和通信的暴露等待。**pool32 接近之前的总耗时，不代表真实模型每 rank 命中了 32 个专家。** 单层 JAX 编译边界与完整 Torch 模型图不同，融合/调度也会影响其余算子。pool64 是近乎所有本地专家均被访问的合成基线，不能直接要求其耗时等于原服务 trace。

同时记录完整 device module 的起止跨度：TPU:0 均值分别为 283.57 / 353.48 / 479.34 μs。它包含算子之间的间隔和分布式启动偏差；不要与上表的算子之和混用。同步 host benchmark（20 次预热、200 次计时）另存于 `summary.json`，包含 Python/JAX dispatch 开销，不用于上表。全部 8 个 rank 的每步明细与中位数/均值均已保存。

已核对两个 GMM 的尺寸和 tiling 与原服务 trace 相同：

- GMM1：`g_64-m_2560-k_4096-act_silu-n_2048-tm_512-tk_4096-tn_1024`
- GMM2：`g_64-m_2560-k_1024-act_None-n_4096-tm_512-tk_1024-tn_4096`

trace 中 `bytes_accessed` 是静态 cost estimate，包含全部专家，不能拿它当作实际 HBM 访问量计算 MBU。

## 正确性

硬件 UT：`3 passed`。三个 case 均满足：

- 输出 shape 为 `[256,4096]` 且有限；TP 两个副本逐位相同。
- 每个 token 恰好 10 个不同专家；全局共 2560 条 route，归一化权重和约等于 1。
- 使用同样 GMM 算术、逐专家处理全部 token 后直接加权累加的参考实现，三组与 baseline **逐位一致**（relative L2=0）。参考实现没有调用生产 sort/permute/unpermute 路径。
- 另一个独立 dense contraction 参考实现的 relative L2 为 1.54%–1.57%，最坏 token 为 2.99%–3.08%。它与 Pallas 在 BF16 中间舍入和后续动态 FP8 量化上存在数值差异，不能要求逐位相同。代码分别设置 dense 及同算术路由参考的误差门限。

## 重跑

从项目根目录执行。TPU 需由 UT 独占；本次已停止原空闲服务，恢复脚本保留在 `runs/profile-20260921T030746Z/start_server.sh`。

```bash
TPU_SKIP_MDS_QUERY=true .venv/bin/python scripts/bench_moe_decode_ut.py \
  --output runs/moe-ut-baseline-new \
  --pools 16 32 64 --seed 20260921 --iterations 200 --profile-steps 10

.venv/bin/python scripts/analyze_moe_ut_trace.py runs/moe-ut-baseline-new

RUN_MOE_TPU_UT=1 .venv/bin/python -m pytest -q tests/test_moe_decode_tpu_baseline.py
```

当前环境：JAX/JAXLIB 0.10.2，libtpu 0.0.47，torch 2.13.0+cpu，torch-tpu 0.1.1.dev20260912075415。测试使用 pytest 9.1.1，离线 trace 转换使用 xprof 2.23.2。未修改 serving 的这些核心包版本。

当前 JAX profiler 的 plugin ABI 与 libtpu 不兼容。脚本先以一个微小 TorchTPU allocation 初始化设备 profiler，再使用 TorchTPU profiler 采集 JAX 执行的同一组 TPU；只调用 lazy_init 会产生空 trace。编译/预热/参考计算都在正式采集区间之外。`--profile-steps 0` 可跳过采集。

当前 libtpu 导出包含 async-update 的 optimized HLO 会报错，故保存 StableHLO；实际 kernel 名称和 tiling 从设备 trace 核验。

## 本次产物

目录：`runs/moe-ut-baseline-20260921/`。

- `summary.json`：配置、revision、各 case 数值误差、专家命中数、host timing、所有 rank 的 device timing。
- `pool*/fixture.npz`：hidden、router weight/bias、top-k IDs/weights、expert_counts、baseline 和两个参考输出。完整专家权重由固定种子重建，未重复保存多份大 tensor。
- `pool*/device_steps.csv`、`device_timing.json`：8 rank × 10 steps 的时间明细与汇总。
- `pool*/trace/`、`trace.json.gz`：XPlane 与可打开的 Chrome trace。分析脚本选择最新采集，自动拒绝空设备 trace 或不完整的 forward 数量。
- `pool*/baseline.stablehlo.txt`：编译图；`pytest.log`：硬件 UT 结果。

后续比较应保持 seed、权重生成、输入、路由、候选池、通信边界与测量口径相同，同时对照 GMM、其余算子和完整 device module 时间。真实模型收益最终仍需回到同一批请求的服务 benchmark 验证。

## 实验实现

已增加保持同一入口的 `dense_expert` 实验开关及配对对比工具；实现范围、数值误差、历史回退及后续布局修复收益见 [新算子说明](MOE_DECODE_DENSE_EXPERT.md)。默认仍使用原路径。
