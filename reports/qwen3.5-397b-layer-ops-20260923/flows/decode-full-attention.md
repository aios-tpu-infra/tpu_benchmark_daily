# DP4/TP2 decode — layer 3（full-attention）的编译后算子依赖

[返回计算流总览](../COMPUTE-FLOWS.md)

本表按最终 HLO 的 operand 依赖排列，编号用于引用上游算子，不是 trace 时间或 kernel launch 编号。同一算子产生多个输出时仍只有一个编号。

保留 ≥1 µs 的计算与布局算子；通信按完整逻辑调用保留，短 start/update/done 不独立列项。低于阈值的数值转换、包装节点和 DMA 被穿透，直接连到上游 producer。while 的循环体合并到循环节点。

表中列出的是数据依赖，不代表所有无边节点一定并发；资源、通信与缓存别名仍可引入执行约束。布局列保留实际维度，省略物理 tiling；完整 shape 与通信 groups 见同名 JSON。
异步算子用短名表示完整调用，省略 cloned/call-done 等包装后缀；JSON 保留承接结果的完整 HLO 名称及行号。

| 编号 | 输入来自 | 编译后算子 | 作用 | 输出 shape |
|---|---|---|---|---|
| N01 | 运行时输入/缓存 | `copy.563` | 数据复制/布局物化 | `bf16[1048576,64]` |
| N02 | 上游层输出/统计量, 权重/scale | `abs_reduce_fusion.173` | FP8 激活量化 | `bf16[64], bf16[64,4096]` |
| N03 | N02 | `compare_select_fusion.293` | FP8 量化 scale | `f32[64]` |
| N04 | N02, N03 | `clamp_convert_fusion.128` | FP8 激活量化 | `f8e4m3fn[64,4096]` |
| N05 | N03, N04, 权重/scale | `fusion.119` | 全注意力输入投影 | `bf16[64,8704]` |
| N06 | N05 | `fusion.1673` | FP8 KV 与 Q/K 准备 | `f8e4m3fn[64,1,256], f32[64]` |
| N07 | N05 | `as_strided.28` | 数据复制/布局物化 | `bf16[64,17,512]` |
| N08 | N06 | `add_rsqrt_fusion.188` | RMSNorm/residual | `f32[64]` |
| N09 | N05, N08, 权重/scale | `multiply_add_fusion.165` | Q/K RMSNorm | `bf16[64,256]` |
| N10 | N09 | `slice_bitcast_fusion.14` | RoPE | `bf16[64,1,192]` |
| N11 | N07 | `fusion.792` | RMSNorm/residual | `f32[64,16]` |
| N12 | N11 | `add_rsqrt_fusion.56` | RMSNorm/residual | `f32[64,16]` |
| N13 | N09, 跨层共享准备 | `subtract_bitcast_fusion.14` | RoPE | `bf16[64,1,32], bf16[64,32], bf16[64,32]` |
| N14 | N09, N13 | `add_bitcast_fusion.14` | RoPE | `bf16[64,1,32]` |
| N15 | N07, N12, 权重/scale | `multiply_add_fusion.128` | Q/K RMSNorm | `bf16[64,16,256]` |
| N16 | N13, N15 | `multiply_subtract_fusion.14` | RoPE | `bf16[64,16,32], bf16[64,16,32]` |
| N17 | N15 | `as_strided.637` | RoPE | `bf16[64,16,192]` |
| N18 | N10, N13, N14 | `select_convert_fusion.29` | FP8 KV 准备 | `f8e4m3fn[64,1,256]` |
| N19 | N06, N18 | `pad_maximum_fusion.15` | FP8 KV 准备 | `f8e4m3fn[64,1,512]` |
| N20 | N16, N17 | `pad_maximum_fusion` | RoPE | `bf16[64,16,256]` |
| N21 | N20 | `transpose_reshape_reshape.30` | 数据复制/布局物化 | `bf16[1,64,8,2,256]` |
| N22 | N19 | `pad.385.clone` | 索引/张量辅助 | `f8e4m3fn[128,2,256]` |
| N23 | N21, N22, 上游层输出/统计量, 跨层共享准备, 运行时输入/缓存 | `RPAd-snh-p128-b8-q1-k3328.45` | 全注意力 RPA | `bf16[1,64,8,2,256], f8e4m3fn[42339,2,64,4,128]` |
| N24 | N22, N23, 跨层共享准备, 运行时输入/缓存 | `RPAm-snh-p128-b2-q256-k256.45` | 全注意力 RPA | `bf16[1,64,8,2,256], f8e4m3fn[42339,2,64,4,128]` |
| N25 | N24 | `reshape_transpose_reshape.30` | 数据复制/布局物化 | `bf16[64,16,256]` |
| N26 | N07, N25 | `abs_reduce_fusion.56` | FP8 激活量化 | `bf16[64], f32[64,16,256]` |
| N27 | N26 | `compare_select_fusion.292` | FP8 量化 scale | `f32[64]` |
| N28 | N26 | `sparse-core-data-format-call.70` | SparseCore 布局转换 | `f32[8,8,16,256]` |
| N29 | N27, N28 | `clamp_convert_fusion.127` | FP8 激活量化 | `f8e4m3fn[64,4096]` |
| N30 | N27, N29, 权重/scale | `fusion.303` | attention 输出投影 | `bf16[64,4096]` |
| N31 | N30 | `all-reduce.378` | TP all-reduce | `bf16[64,4096]` |
| N32 | N31, 上游层输出/统计量 | `multiply_reduce_fusion.113` | RMSNorm/residual | `f32[64], bf16[64,4096]` |
| N33 | N32 | `add_rsqrt_fusion.187` | RMSNorm/residual | `f32[64]` |
| N34 | N31, N32, N33, 权重/scale | `abs_reduce_fusion.172` | FP8 激活量化 | `bf16[64], bf16[64,4096]` |
| N35 | N34, 权重/scale | `fusion.1998` | Router 打分 | `bf16[64,512]` |
| N36 | N35 | `all-gather.133` | EP dispatch / all-gather | `bf16[256,512]` |
| N37 | N36 | `reduce.1244` | Router FP32 softmax | `bf16[256]` |
| N38 | N37 | `convert.753` | Router FP32 softmax | `f32[256]` |
| N39 | N36, N38 | `fusion.1329` | Router FP32 softmax | `f32[256]` |
| N40 | N36, N38, N39 | `fusion.1328` | Router FP32 softmax | `f32[256,512]` |
| N41 | N40 | `router_topk.123` | Router top-k | `f32[256,10], s32[256,10]` |
| N42 | N41 | `reduce.1246` | 索引/张量辅助 | `f32[256]` |
| N43 | N42 | `compare_select_fusion.365` | Router top-k 归一化保护 | `f32[256]` |
| N44 | N41, N43, 运行时输入/缓存 | `fusion.1452` | MoE 路由系数准备 | `f32[256,10]` |
| N45 | N44 | `slice_reduce_fusion.1138` | MoE top-k 拆列 | `f32[256], f32[256], f32[256], f32[256] …（共 10 个 tuple 分量）` |
| N46 | N41, 运行时输入/缓存 | `subtract_select_fusion.56` | MoE 专家 ID 本地化 | `s32[256,10]` |
| N47 | N46 | `slice_reduce_fusion.1139` | MoE top-k 拆列 | `s32[256], s32[256], s32[256], s32[256] …（共 10 个 tuple 分量）` |
| N48 | N45, N47 | `compare_reduce_fusion.56` | MoE 本地路由系数 | `pred[64], f32[256,64]` |
| N49 | N48 | `pad.386` | MoE 路由系数准备 | `f32[256,128]` |
| N50 | N48 | `convert_element_type.2711` | MoE 活跃专家 mask | `s32[64]` |
| N51 | N34 | `all-gather.135` | EP dispatch / all-gather | `bf16[256,4096]` |
| N52 | N51 | `copy.1585` | 数据复制/布局物化 | `bf16[256,4096]` |
| N53 | N49, N50, N52, 权重/scale | `dense_expert_moe-e_64-m_256-h_4096-i_1024-buffers_3.243` | 路由专家计算 | `bf16[256,4096]` |
| N54 | N34 | `compare_select_fusion.291` | FP8 量化 scale | `f32[64]` |
| N55 | N34, N54 | `clamp_convert_fusion.126` | FP8 激活量化 | `f8e4m3fn[64,4096]` |
| N56 | N54, N55, 权重/scale | `fusion.543` | 共享专家 gate/up | `bf16[64,1024]` |
| N57 | N56 | `abs_reduce_fusion.236` | FP8 激活量化 | `bf16[64]` |
| N58 | N57 | `compare_select_fusion.290` | FP8 量化 scale | `f32[64]` |
| N59 | N34, 权重/scale | `fusion.770` | 共享专家 gate | `bf16[64]` |
| N60 | N56, N58 | `clamp_convert_fusion.191` | FP8 激活量化 | `f8e4m3fn[64,512]` |
| N61 | N53 | `reduce-scatter.247` | EP combine / DP reduce-scatter | `bf16[64,4096]` |
| N62 | N58, N59, N60, N61, 权重/scale | `fusion.473` | 共享专家 down/合并 | `bf16[64,4096]` |
| N63 | N62 | `all-reduce.380` | TP all-reduce | `bf16[64,4096]` |
| N64 | N31, N32, N63 | `multiply_reduce_fusion.112` | RMSNorm/residual | `f32[64]` |
| N65 | N64 | `add_rsqrt_fusion.186` | RMSNorm/residual | `f32[64]` |

[完整依赖与 HLO 行号（JSON）](decode-full-attention.json)
