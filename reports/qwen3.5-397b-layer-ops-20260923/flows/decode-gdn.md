# DP4/TP2 decode — layer 1（gdn）的编译后算子依赖

[返回计算流总览](../COMPUTE-FLOWS.md)

本表按最终 HLO 的 operand 依赖排列，编号用于引用上游算子，不是 trace 时间或 kernel launch 编号。同一算子产生多个输出时仍只有一个编号。

保留 ≥1 µs 的计算与布局算子；通信按完整逻辑调用保留，短 start/update/done 不独立列项。低于阈值的数值转换、包装节点和 DMA 被穿透，直接连到上游 producer。while 的循环体合并到循环节点。

表中列出的是数据依赖，不代表所有无边节点一定并发；资源、通信与缓存别名仍可引入执行约束。布局列保留实际维度，省略物理 tiling；完整 shape 与通信 groups 见同名 JSON。
异步算子用短名表示完整调用，省略 cloned/call-done 等包装后缀；JSON 保留承接结果的完整 HLO 名称及行号。

| 编号 | 输入来自 | 编译后算子 | 作用 | 输出 shape |
|---|---|---|---|---|
| N01 | 权重/scale | `copy.1108` | 数据复制/布局物化 | `f32[6144,1,4]` |
| N02 | 权重/scale | `copy_.316` | FP32 中间计算 | `f32[128]` |
| N03 | 上游层输出/统计量, 权重/scale, 跨层共享准备 | `abs_reduce_fusion.177` | FP8 激活量化 | `bf16[64], bf16[64,4096]` |
| N04 | N03 | `compare_select_fusion.301` | FP8 量化 scale | `f32[64]` |
| N05 | N03, N04 | `clamp_convert_fusion.132` | FP8 激活量化 | `f8e4m3fn[64,4096]` |
| N06 | N04, N05, 权重/scale | `fusion.46` | GDN 输入投影 | `bf16[64,10240]` |
| N07 | N06 | `slice_convert_fusion.43` | GDN 输入准备 | `f32[64,6144]` |
| N08 | N07 | `reshape.6684` | 数据复制/布局物化 | `f32[64,1,6144]` |
| N09 | N03, 权重/scale | `fusion.1994` | GDN 门控投影 | `bf16[64,64]` |
| N10 | N09 | `fusion.1990` | GDN B/A 拆分 | `bf16[64,32], bf16[64,32]` |
| N11 | N10 | `pad_convert_fusion.87` | GDN 输入准备 | `f32[64,128]` |
| N12 | N10 | `pad_convert_fusion.86` | GDN 输入准备 | `f32[64,128]` |
| N13 | N01, N08, N11, N12, 上游层输出/统计量, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_conv1d_gdn_batched.91` | GDN 核心 | `bf16[64,32,128], f8e4m3fn[42339,2,64,4,128]` |
| N14 | N01, N08, N11, N12, N13, 权重/scale, 跨层共享准备 | `fused_conv1d_gdn_per_seq.91` | GDN 核心 | `bf16[64,32,128], f8e4m3fn[42339,2,64,4,128]` |
| N15 | N14 | `fusion.784` | RMSNorm/residual | `f32[64,32]` |
| N16 | N15 | `add_rsqrt_fusion.58` | RMSNorm/residual | `f32[64,32]` |
| N17 | N06 | `sparse-core-data-format-call.73` | SparseCore 布局转换 | `bf16[8,8,80,128]` |
| N18 | N02, N14, N16, N17 | `abs_reduce_fusion.58` | FP8 激活量化 | `bf16[64], bf16[64,32,128]` |
| N19 | N18 | `convert.1306` | FP32 中间计算 | `f32[8,8,32,128]` |
| N20 | N18 | `compare_select_fusion.300` | FP8 量化 scale | `f32[64]` |
| N21 | N19, N20 | `fusion.1038` | FP8 激活量化 | `f8e4m3fn[64,4096]` |
| N22 | N20, N21, 权重/scale | `fusion.301` | attention 输出投影 | `bf16[64,4096]` |
| N23 | N22 | `all-reduce.370` | TP all-reduce | `bf16[64,4096]` |
| N24 | N23, 上游层输出/统计量, 跨层共享准备 | `multiply_reduce_fusion.117` | RMSNorm/residual | `f32[64], bf16[64,4096]` |
| N25 | N24 | `add_rsqrt_fusion.192` | RMSNorm/residual | `f32[64]` |
| N26 | N23, N24, N25, 权重/scale | `abs_reduce_fusion.176` | FP8 激活量化 | `bf16[64], bf16[64,4096]` |
| N27 | N26, 权重/scale | `fusion.1995` | Router 打分 | `bf16[64,512]` |
| N28 | N27 | `all-gather.125` | EP dispatch / all-gather | `bf16[256,512]` |
| N29 | N28 | `reduce.1221` | Router FP32 softmax | `bf16[256]` |
| N30 | N29 | `convert.747` | Router FP32 softmax | `f32[256]` |
| N31 | N28, N30 | `fusion.1333` | Router FP32 softmax | `f32[256]` |
| N32 | N28, N30, N31 | `fusion.1332` | Router FP32 softmax | `f32[256,512]` |
| N33 | N32 | `router_topk.121` | Router top-k | `f32[256,10], s32[256,10]` |
| N34 | N33 | `reduce.1223` | 索引/张量辅助 | `f32[256]` |
| N35 | N34 | `compare_select_fusion.493` | Router top-k 归一化保护 | `f32[256]` |
| N36 | N33, N35, 运行时输入/缓存 | `fusion.1454` | MoE 路由系数准备 | `f32[256,10]` |
| N37 | N36 | `slice_reduce_fusion.1178` | MoE top-k 拆列 | `f32[256], f32[256], f32[256], f32[256] …（共 10 个 tuple 分量）` |
| N38 | N33, 运行时输入/缓存 | `subtract_select_fusion.58` | MoE 专家 ID 本地化 | `s32[256,10]` |
| N39 | N38 | `slice_reduce_fusion.1179` | MoE top-k 拆列 | `s32[256], s32[256], s32[256], s32[256] …（共 10 个 tuple 分量）` |
| N40 | N37, N39 | `compare_reduce_fusion.58` | MoE 本地路由系数 | `pred[64], f32[256,64]` |
| N41 | N40 | `pad.381` | MoE 路由系数准备 | `f32[256,128]` |
| N42 | N40 | `convert_element_type.2675` | MoE 活跃专家 mask | `s32[64]` |
| N43 | N26 | `all-gather.127` | EP dispatch / all-gather | `bf16[256,4096]` |
| N44 | N43 | `copy.1534` | 数据复制/布局物化 | `bf16[256,4096]` |
| N45 | N41, N42, N44, 权重/scale | `dense_expert_moe-e_64-m_256-h_4096-i_1024-buffers_3.241` | 路由专家计算 | `bf16[256,4096]` |
| N46 | N26 | `compare_select_fusion.299` | FP8 量化 scale | `f32[64]` |
| N47 | N26, N46 | `clamp_convert_fusion.131` | FP8 激活量化 | `f8e4m3fn[64,4096]` |
| N48 | N46, N47, 权重/scale | `fusion.541` | 共享专家 gate/up | `bf16[64,1024]` |
| N49 | N48 | `abs_reduce_fusion.238` | FP8 激活量化 | `bf16[64]` |
| N50 | N49 | `compare_select_fusion.298` | FP8 量化 scale | `f32[64]` |
| N51 | N26, 权重/scale | `fusion.776` | 共享专家 gate | `bf16[64]` |
| N52 | N48, N50 | `clamp_convert_fusion.193` | FP8 激活量化 | `f8e4m3fn[64,512]` |
| N53 | N45 | `reduce-scatter.243` | EP combine / DP reduce-scatter | `bf16[64,4096]` |
| N54 | N50, N51, N52, N53, 权重/scale | `fusion.477` | 共享专家 down/合并 | `bf16[64,4096]` |
| N55 | N54 | `all-reduce.372` | TP all-reduce | `bf16[64,4096]` |
| N56 | N23, N24, N55 | `multiply_reduce_fusion.116` | RMSNorm/residual | `f32[64]` |
| N57 | N56 | `add_rsqrt_fusion.191` | RMSNorm/residual | `f32[64]` |

[完整依赖与 HLO 行号（JSON）](decode-gdn.json)
