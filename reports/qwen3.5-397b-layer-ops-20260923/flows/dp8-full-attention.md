# DP8 prefill — layer 3（full-attention）的编译后算子依赖

[返回计算流总览](../COMPUTE-FLOWS.md)

本表按最终 HLO 的 operand 依赖排列，编号用于引用上游算子，不是 trace 时间或 kernel launch 编号。同一算子产生多个输出时仍只有一个编号。

保留 ≥1 µs 的计算与布局算子；通信按完整逻辑调用保留，短 start/update/done 不独立列项。低于阈值的数值转换、包装节点和 DMA 被穿透，直接连到上游 producer。while 的循环体合并到循环节点。

表中列出的是数据依赖，不代表所有无边节点一定并发；资源、通信与缓存别名仍可引入执行约束。布局列保留实际维度，省略物理 tiling；完整 shape 与通信 groups 见同名 JSON。
异步算子用短名表示完整调用，省略 cloned/call-done 等包装后缀；JSON 保留承接结果的完整 HLO 名称及行号。

| 编号 | 输入来自 | 编译后算子 | 作用 | 输出 shape |
|---|---|---|---|---|
| N01 | 上游层输出/统计量, 权重/scale | `abs_reduce_fusion.173` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N02 | N01 | `clamp_convert_fusion.128` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N03 | N01, N02, 权重/scale | `multiply_convert_fusion.59` | 全注意力输入投影 | `bf16[4096,17408]` |
| N04 | N03 | `sparse-core-data-format-call.95` | SparseCore 布局转换 | `bf16[4096,17408]` |
| N05 | N04 | `fusion.556` | RMSNorm/residual | `f32[4096,2], f8e4m3fn[4096,2,256]` |
| N06 | N04, N05, 权重/scale | `multiply_add_fusion.225` | Q/K RMSNorm | `bf16[4096,2,256]` |
| N07 | N04 | `fusion.12` | RMSNorm/residual | `f32[4096,32]` |
| N08 | N04, N07, 权重/scale | `multiply_add_fusion.14` | Q/K RMSNorm | `bf16[4096,32,256]` |
| N09 | N08 | `as_strided.725` | RoPE | `bf16[4096,32,192]` |
| N10 | N08, 跨层共享准备 | `multiply_subtract_fusion.14` | RoPE | `bf16[4096,32,32], bf16[4096,32,32]` |
| N11 | N09, N10 | `pad_maximum_fusion` | RoPE | `bf16[4096,32,256]` |
| N12 | N06, 跨层共享准备 | `select_convert_fusion.89` | FP8 KV 准备 | `f8e4m3fn[4096,2,256]` |
| N13 | N05, N12 | `pad_maximum_fusion.15` | FP8 KV 准备 | `f8e4m3fn[4096,2,512]` |
| N14 | N13 | `reshape_reshape.184` | 数据复制/布局物化 | `f8e4m3fn[4096,4,64,4]` |
| N15 | N11 | `sparse-core-data-format-call.94` | SparseCore 布局转换 | `bf16[4096,2,8,2,256]` |
| N16 | N14, N15, 上游层输出/统计量, 跨层共享准备, 运行时输入/缓存 | `RPAd-p128-b8-q1-k1536.45` | 全注意力 RPA | `bf16[2,4096,8,2,256], f8e4m3fn[15642,4,64,4,128]` |
| N17 | N14, N16, 跨层共享准备, 运行时输入/缓存 | `RPAm-p128-b1-q256-k256.45` | 全注意力 RPA | `bf16[2,4096,8,2,256], f8e4m3fn[15642,4,64,4,128]` |
| N18 | N17 | `reshape.7177` | 数据复制/布局物化 | `bf16[2,4096,16,256]` |
| N19 | N18 | `sparse-core-data-format-call.93` | SparseCore 布局转换 | `bf16[2,4096,16,256]` |
| N20 | N19 | `sparse-core-data-format-call.92` | SparseCore 布局转换 | `bf16[4096,32,256]` |
| N21 | N04, N20 | `abs_reduce_fusion.56` | FP8 激活量化 | `bf16[4096], bf16[4096,32,256]` |
| N22 | N21 | `bitcast_convert_fusion.14` | FP8 激活量化 | `f32[4096,8192]` |
| N23 | N22 | `sparse-core-data-format-call.91` | SparseCore 布局转换 | `f32[4096,8192]` |
| N24 | N21, N23 | `clamp_convert_fusion.14` | FP8 激活量化 | `f8e4m3fn[4096,8192]` |
| N25 | N21, N24, 上游层输出/统计量, 权重/scale | `multiply_reduce_fusion.113` | attention 输出投影 | `f32[4096], bf16[4096,4096], bf16[4096,4096]` |
| N26 | N25, 权重/scale | `abs_reduce_fusion.172` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N27 | N26, 权重/scale | `fusion.3144` | Router 打分 | `bf16[4096], bf16[4096,512]` |
| N28 | N27 | `copy.6058` | 数据复制/布局物化 | `bf16[4096,512]` |
| N29 | N27, N28 | `fusion.789` | MoE 路由/通信计划 | `f32[4096]` |
| N30 | N27, N28, N29 | `is-finite_reduce_fusion.56` | MoE 路由/通信计划 | `pred[4096], f32[4096,512]` |
| N31 | N30 | `shard_map.183` | Router top-k | `f32[4096,10], s32[4096,10]` |
| N32 | N31 | `fusion.829` | MoE 路由/通信计划 | `s32[4096,10]` |
| N33 | N32 | `broadcast_in_dim_reshape.371` | 数据复制/布局物化 | `s32[160,256]` |
| N34 | N32 | `reshape_reshape.185` | 数据复制/布局物化 | `s32[40960]` |
| N35 | N33 | `convert_reduce_fusion.227` | MoE 路由/通信计划 | `s32[160,512], s32[512]` |
| N36 | N26 | `fusion.499` | MoE 激活量化 | `f8e4m3fn[512,8,32,128]` |
| N37 | N36 | `copy.8020` | 数据复制/布局物化 | `f8e4m3fn[512,8,32,128]` |
| N38 | 权重/scale | `copy.8806` | 数据复制/布局物化 | `f32[64,1,1,4096]` |
| N39 | N31 | `reduce.4574` | MoE 路由/通信计划 | `f32[4096]` |
| N40 | N35 | `copy.6060` | 数据复制/布局物化 | `s32[512,2,128]` |
| N41 | N26, 权重/scale | `convolution_reduce_fusion.56` | 共享专家 gate | `bf16[4096]` |
| N42 | N26 | `clamp_convert_fusion.127` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N43 | N26, N42, 权重/scale | `multiply_convert_fusion.356` | 共享专家 gate/up | `bf16[4096,2048]` |
| N44 | N43 | `abs_reduce_fusion.236` | FP8 激活量化 | `bf16[4096]` |
| N45 | N33, 跨层共享准备 | `convert_reduce_fusion.353` | MoE 路由/通信计划 | `s32[160,256]` |
| N46 | N43, N44 | `clamp_convert_fusion.191` | FP8 激活量化 | `f8e4m3fn[4096,1024]` |
| N47 | N26, N34, N35 | `all-gather.307` | EP dispatch / all-gather | `s32[8,1,45568]` |
| N48 | N47 | `copy.6061` | EP 全收集 | `s32[8,1,45568]` |
| N49 | N48 | `slice.3035` | MoE 路由/通信计划 | `s32[8,40960]` |
| N50 | N48 | `reduce-window.1388` | MoE 路由/通信计划 | `s32[8,4,128]` |
| N51 | N48 | `fusion.3116` | MoE 路由/通信计划 | `s32[8], s32[512,8], s32[512,8]` |
| N52 | N51, 运行时输入/缓存 | `and_reduce_fusion.56` | MoE 路由/通信计划 | `s32[512]` |
| N53 | N33, N35, N40, N48, N50, N52, 运行时输入/缓存 | `select_reduce_fusion.112` | MoE 路由/通信计划 | `s32[160,256], s32[160,256]` |
| N54 | N45, N53 | `reshape.7210` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N55 | N45, N53 | `reshape.7211` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N56 | N54, N55 | `fusion.903` | MoE 路由/通信计划 | `s32[4096,10,1]` |
| N57 | N56 | `reshape.9604` | 数据复制/布局物化 | `s32[40960]` |
| N58 | N49 | `copy.6067` | 数据复制/布局物化 | `s32[8,40960]` |
| N59 | N58 | `reshape.7198` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N60 | N59 | `copy.6068` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N61 | N60, 跨层共享准备 | `convert_reduce_fusion.225` | MoE 路由/通信计划 | `s32[1280,64], s32[1280,8,64]` |
| N62 | N61 | `reduce-window.1394` | MoE 路由/通信计划 | `s32[8,64,2,128]` |
| N63 | N61, 跨层共享准备 | `select_reduce_fusion.176` | MoE 路由/通信计划 | `s32[1280,8,64]` |
| N64 | N60, 跨层共享准备 | `convert_reduce_fusion.352` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N65 | N48, N51, N60, N61, N62, N63, 跨层共享准备, 运行时输入/缓存 | `fusion.303` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N66 | N64, N65 | `add.22256` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N67 | N66 | `copy.6077` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N68 | N67 | `reshape.7203` | 数据复制/布局物化 | `s32[327680]` |
| N69 | N48, 运行时输入/缓存 | `compare_and_fusion.56` | MoE 路由/通信计划 | `pred[8,40960]` |
| N70 | N69 | `copy.6066` | 数据复制/布局物化 | `pred[8,40960]` |
| N71 | N48 | `broadcast_in_dim.3768` | MoE 路由/通信计划 | `s32[32768,10]` |
| N72 | N71 | `reshape.10014` | 数据复制/布局物化 | `s32[327680]` |
| N73 | N48, N51, 跨层共享准备, 运行时输入/缓存 | `multiply_reduce_fusion.416` | MoE 路由/通信计划 | `s32[8], s32[8], s32[8], s32[8] …（共 8 个 tuple 分量）` |
| N74 | N48, 运行时输入/缓存 | `convert_reduce_fusion.416` | MoE 路由/通信计划 | `s32[64]` |
| N75 | N37 | `all-gather.187` | EP dispatch / all-gather | `f8e4m3fn[32768,32,128]` |
| N76 | N68, N70, 运行时输入/缓存 | `compare_select_fusion.292` | MoE 返回地址/索引 | `s32[327680], s32[327680]` |
| N77 | N76, 跨层共享准备 | `scatter_offload.6` | SparseCore scatter（完整调用） | `s32[328320]` |
| N78 | N77 | `broadcast_clamp_fusion.56` | MoE 路由/通信计划 | `s32[328320]` |
| N79 | N78 | `pad.600` | MoE 路由/通信计划 | `s32[328576]` |
| N80 | N72, N76, 跨层共享准备 | `scatter_offload.7` | SparseCore scatter（完整调用） | `s32[328448]` |
| N81 | N80 | `bitcast-convert_bitcast_fusion.56` | MoE 路由/通信计划 | `f32[2566,128]` |
| N82 | N38, N48, N50, N51, N73, N74, N75, N79, N81, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_ep_moe_v2_g64_c128_nb3.183` | 路由专家计算 | `f8e4m3fn[41472,32,128], f32[1350,128], f8e4m3fn[328320,32,128], f32[2695,128]` |
| N83 | N57, N82 | `gather_offload.5` | SparseCore gather（完整调用） | `f32[40960]` |
| N84 | N83 | `reshape.9607` | 数据复制/布局物化 | `f32[4096,10]` |
| N85 | N30, N31, N39, N84 | `select_multiply_fusion.56` | MoE 路由/通信计划 | `f32[4096,10]` |
| N86 | N85 | `reshape.7212` | 数据复制/布局物化 | `f32[40960]` |
| N87 | N45, N53, N82, N86 | `moe_v2_combine_t4096_k10_b64.183` | 路由专家合并 | `bf16[4096,4096]` |
| N88 | N25, N41, N44, N46, N87, 权重/scale | `multiply_reduce_fusion.112` | 共享专家 down/合并 | `f32[4096], bf16[4096,4096]` |

[完整依赖与 HLO 行号（JSON）](dp8-full-attention.json)
