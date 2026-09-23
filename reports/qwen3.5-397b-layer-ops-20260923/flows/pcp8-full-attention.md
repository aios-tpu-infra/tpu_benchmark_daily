# PCP8 prefill — layer 3（full-attention）的编译后算子依赖

[返回计算流总览](../COMPUTE-FLOWS.md)

本表按最终 HLO 的 operand 依赖排列，编号用于引用上游算子，不是 trace 时间或 kernel launch 编号。同一算子产生多个输出时仍只有一个编号。

保留 ≥1 µs 的计算与布局算子；通信按完整逻辑调用保留，短 start/update/done 不独立列项。低于阈值的数值转换、包装节点和 DMA 被穿透，直接连到上游 producer。while 的循环体合并到循环节点。

表中列出的是数据依赖，不代表所有无边节点一定并发；资源、通信与缓存别名仍可引入执行约束。布局列保留实际维度，省略物理 tiling；完整 shape 与通信 groups 见同名 JSON。
异步算子用短名表示完整调用，省略 cloned/call-done 等包装后缀；JSON 保留承接结果的完整 HLO 名称及行号。

| 编号 | 输入来自 | 编译后算子 | 作用 | 输出 shape |
|---|---|---|---|---|
| N01 | 上游层输出/统计量, 权重/scale | `abs_reduce_fusion.131` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N02 | N01 | `clamp_convert_fusion.131` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N03 | N01, N02, 权重/scale | `multiply_convert_fusion.14` | 全注意力输入投影 | `bf16[4096,17408]` |
| N04 | N03 | `fusion.428` | RMSNorm/residual | `f32[4096,2], f8e4m3fn[4096,2,256]` |
| N05 | N03, N04, 权重/scale | `multiply_add_fusion.225` | Q/K RMSNorm | `bf16[4096,2,256]` |
| N06 | N03 | `fusion.7` | RMSNorm/residual | `f32[4096,32]` |
| N07 | N03, N06, 权重/scale | `multiply_add_fusion.14` | Q/K RMSNorm | `bf16[4096,32,256]` |
| N08 | N07 | `as_strided.542` | RoPE | `bf16[4096,32,192]` |
| N09 | N07, 跨层共享准备 | `multiply_subtract_fusion.14` | RoPE | `bf16[4096,32,32], bf16[4096,32,32]` |
| N10 | N08, N09 | `pad_maximum_fusion` | RoPE | `bf16[4096,32,256]` |
| N11 | N05 | `slice` | 异步切片（完整调用） | `bf16[4096,2,192]` |
| N12 | N05, N11, 跨层共享准备 | `select_convert_fusion.89` | FP8 KV 准备 | `f8e4m3fn[4096,2,256]` |
| N13 | N04, N12 | `pad_maximum_fusion.15` | FP8 KV 准备 | `f8e4m3fn[4096,2,512]` |
| N14 | N13 | `reshape.22996` | 数据复制/布局物化 | `f8e4m3fn[4096,4,64,4]` |
| N15 | N10, 跨层共享准备 | `sparse-core-data-format-call.58` | SparseCore 布局转换 | `bf16[4096,32,256]` |
| N16 | N14, N15, 上游层输出/统计量, 跨层共享准备 | `pcp_streaming_attention_current_state_page_groups_multi_head.45` | PCP 当前片段 attention | `f32[2,4096,16,128], f32[2,4096,16,128], f32[2,4096,16,256], f8e4m3fn[4,64,4,32768]` |
| N17 | N16, 上游层输出/统计量, 跨层共享准备 | `pcp_write_captured_kv_to_local_cache.45` | PCP KV 写回 | `f8e4m3fn[7764,4,64,4,256]` |
| N18 | N15, N16, N17, 跨层共享准备 | `pcp_streaming_attention_history_output_page_groups_multi_head.45` | PCP 历史 attention | `bf16[4096,2,16,256]` |
| N19 | N16, N18, 跨层共享准备 | `pcp_streaming_attention_zero_history_output_multi_head.45` | PCP 无历史输出 | `bf16[4096,2,16,256]` |
| N20 | N19, 跨层共享准备 | `broadcast_select_fusion.21` | PCP 输出选择 | `bf16[4096,2,16,256]` |
| N21 | N20 | `sparse-core-data-format-call.57` | SparseCore 布局转换 | `bf16[4096,32,256]` |
| N22 | N03, N21 | `abs_reduce_fusion.14` | FP8 激活量化 | `bf16[4096], bf16[4096,32,256]` |
| N23 | N22 | `bitcast_convert_fusion.14` | FP8 激活量化 | `f32[4096,8192]` |
| N24 | N23 | `sparse-core-data-format-call.56` | SparseCore 布局转换 | `f32[4096,8192]` |
| N25 | N22, N24 | `clamp_convert_fusion.56` | FP8 激活量化 | `f8e4m3fn[4096,8192]` |
| N26 | N22, N25, 上游层输出/统计量, 权重/scale | `multiply_reduce_fusion.158` | attention 输出投影 | `f32[4096], bf16[4096,4096], bf16[4096,4096]` |
| N27 | N26, 权重/scale | `abs_reduce_fusion.130` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N28 | N27, 权重/scale | `fusion.3381` | Router 打分 | `bf16[4096], bf16[4096,512]` |
| N29 | N28 | `copy.15318` | 数据复制/布局物化 | `bf16[4096,512]` |
| N30 | N28, N29 | `fusion.661` | MoE 路由/通信计划 | `f32[4096]` |
| N31 | N28, N29, N30 | `is-finite_reduce_fusion.56` | MoE 路由/通信计划 | `pred[4096], f32[4096,512]` |
| N32 | N31 | `shard_map.275` | Router top-k | `f32[4096,10], s32[4096,10]` |
| N33 | N32 | `fusion.791` | MoE 路由/通信计划 | `s32[4096,10]` |
| N34 | N33 | `broadcast_in_dim_reshape.378` | 数据复制/布局物化 | `s32[160,256]` |
| N35 | N33 | `reshape_reshape.123` | 数据复制/布局物化 | `s32[40960]` |
| N36 | N34 | `convert_reduce_fusion.227` | MoE 路由/通信计划 | `s32[160,512], s32[512]` |
| N37 | N27 | `fusion.371` | MoE 激活量化 | `f8e4m3fn[512,8,32,128]` |
| N38 | N37 | `copy.20649` | 数据复制/布局物化 | `f8e4m3fn[512,8,32,128]` |
| N39 | 权重/scale | `copy.21573` | 数据复制/布局物化 | `f32[64,1,1,4096]` |
| N40 | N32 | `reduce.5649` | MoE 路由/通信计划 | `f32[4096]` |
| N41 | N27, 权重/scale | `convolution_reduce_fusion.56` | 共享专家 gate | `bf16[4096]` |
| N42 | N27 | `clamp_convert_fusion.130` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N43 | N27, N42, 权重/scale | `multiply_convert_fusion.311` | 共享专家 gate/up | `bf16[4096,2048]` |
| N44 | N43 | `abs_reduce_fusion.191` | FP8 激活量化 | `bf16[4096]` |
| N45 | N34, 跨层共享准备 | `convert_reduce_fusion.353` | MoE 路由/通信计划 | `s32[160,256]` |
| N46 | N43, N44 | `clamp_convert_fusion.191` | FP8 激活量化 | `f8e4m3fn[4096,1024]` |
| N47 | N27, N35, N36 | `all-gather.307` | EP dispatch / all-gather | `s32[8,1,45568]` |
| N48 | N47 | `copy.15321` | EP 全收集 | `s32[8,1,45568]` |
| N49 | N48 | `slice.10916` | MoE 路由/通信计划 | `s32[8,40960]` |
| N50 | N48 | `fusion.2754` | MoE 路由/通信计划 | `s32[8], s32[512], s32[8], s32[8,512] …（共 6 个 tuple 分量）` |
| N51 | N50 | `reduce-window.1948` | MoE 路由/通信计划 | `s32[8,4,128]` |
| N52 | N48, N50 | `fusion.3209` | MoE 路由/通信计划 | `s32[8], s32[512,8], s32[512,8]` |
| N53 | N52, 运行时输入/缓存 | `and_reduce_fusion.60` | MoE 路由/通信计划 | `s32[512]` |
| N54 | N34, N36, N50, N51, N53, 运行时输入/缓存 | `select_reduce_fusion.112` | MoE 路由/通信计划 | `s32[160,256], s32[160,256]` |
| N55 | N45, N54 | `reshape.23111` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N56 | N45, N54 | `reshape.23112` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N57 | N55, N56 | `fusion.865` | MoE 路由/通信计划 | `s32[4096,10,1]` |
| N58 | N57 | `reshape.27180` | 数据复制/布局物化 | `s32[40960]` |
| N59 | N49 | `copy.15327` | 数据复制/布局物化 | `s32[8,40960]` |
| N60 | N59 | `reshape.23099` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N61 | N60 | `copy.15328` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N62 | N61, 跨层共享准备 | `convert_reduce_fusion.225` | MoE 路由/通信计划 | `s32[1280,64], s32[1280,8,64]` |
| N63 | N62 | `reduce-window.1954` | MoE 路由/通信计划 | `s32[8,64,2,128]` |
| N64 | N50, N52, N62, N63, 跨层共享准备, 运行时输入/缓存 | `fusion.1266` | MoE 路由/通信计划 | `s32[8,160,64]` |
| N65 | N62, 跨层共享准备 | `select_reduce_fusion.176` | MoE 路由/通信计划 | `s32[1280,8,64]` |
| N66 | N61, 跨层共享准备 | `convert_reduce_fusion.352` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N67 | N61, N64, N65, 跨层共享准备 | `fusion.175` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N68 | N66, N67 | `copy.15337` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N69 | N68 | `reshape.23104` | 数据复制/布局物化 | `s32[327680]` |
| N70 | N48, 运行时输入/缓存 | `compare_and_fusion.57` | MoE 路由/通信计划 | `pred[8,40960]` |
| N71 | N70 | `copy.15326` | 数据复制/布局物化 | `pred[8,40960]` |
| N72 | N48 | `broadcast_in_dim.4782` | MoE 路由/通信计划 | `s32[32768,10]` |
| N73 | N72 | `reshape.27690` | 数据复制/布局物化 | `s32[327680]` |
| N74 | N50, 运行时输入/缓存 | `reduce-window.1952` | MoE 路由/通信计划 | `s32[64,8]` |
| N75 | N50, 跨层共享准备, 运行时输入/缓存 | `multiply_reduce_fusion.392` | MoE 路由/通信计划 | `s32[8], s32[64,8]` |
| N76 | N50, N52, 跨层共享准备, 运行时输入/缓存 | `multiply_reduce_fusion.462` | MoE 路由/通信计划 | `s32[8], s32[8], s32[8], s32[8] …（共 8 个 tuple 分量）` |
| N77 | N50, N51, 运行时输入/缓存 | `fusion.1533` | MoE 路由/通信计划 | `s32[64,8]` |
| N78 | N38 | `all-gather.187` | EP dispatch / all-gather | `f8e4m3fn[32768,32,128]` |
| N79 | N69, N71, 运行时输入/缓存 | `compare_select_fusion.295` | MoE 返回地址/索引 | `s32[327680], s32[327680]` |
| N80 | N50, 跨层共享准备, 运行时输入/缓存 | `broadcast_subtract_fusion.56` | MoE 路由/通信计划 | `s32[64]` |
| N81 | N79, 跨层共享准备 | `scatter_offload.10` | SparseCore scatter（完整调用） | `s32[328320]` |
| N82 | N81 | `broadcast_clamp_fusion.56` | MoE 路由/通信计划 | `s32[328320]` |
| N83 | N82 | `pad.1520` | MoE 路由/通信计划 | `s32[328576]` |
| N84 | N73, N79, 跨层共享准备 | `scatter_offload.11` | SparseCore scatter（完整调用） | `s32[328448]` |
| N85 | N84 | `bitcast-convert_bitcast_fusion.56` | MoE 路由/通信计划 | `f32[2566,128]` |
| N86 | N39, N50, N52, N74, N75, N76, N77, N78, N80, N83, N85, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_ep_moe_v2_g64_c128_nb3.183` | 路由专家计算 | `f8e4m3fn[41472,32,128], f32[1350,128], f8e4m3fn[328320,32,128], f32[2695,128]` |
| N87 | N58, N86 | `gather_offload.6` | SparseCore gather（完整调用） | `f32[40960]` |
| N88 | N87 | `reshape.27183` | 数据复制/布局物化 | `f32[4096,10]` |
| N89 | N31, N32, N40, N88 | `select_multiply_fusion.56` | MoE 路由/通信计划 | `f32[4096,10]` |
| N90 | N89 | `reshape.23113` | 数据复制/布局物化 | `f32[40960]` |
| N91 | N45, N54, N86, N90 | `moe_v2_combine_t4096_k10_b64.183` | 路由专家合并 | `bf16[4096,4096]` |
| N92 | N26, N41, N44, N46, N91, 权重/scale | `multiply_reduce_fusion.157` | 共享专家 down/合并 | `f32[4096], bf16[4096,4096]` |

[完整依赖与 HLO 行号（JSON）](pcp8-full-attention.json)
