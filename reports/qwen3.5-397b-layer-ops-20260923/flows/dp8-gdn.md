# DP8 prefill — layer 1（gdn）的编译后算子依赖

[返回计算流总览](../COMPUTE-FLOWS.md)

本表按最终 HLO 的 operand 依赖排列，编号用于引用上游算子，不是 trace 时间或 kernel launch 编号。同一算子产生多个输出时仍只有一个编号。

保留 ≥1 µs 的计算与布局算子；通信按完整逻辑调用保留，短 start/update/done 不独立列项。低于阈值的数值转换、包装节点和 DMA 被穿透，直接连到上游 producer。while 的循环体合并到循环节点。

表中列出的是数据依赖，不代表所有无边节点一定并发；资源、通信与缓存别名仍可引入执行约束。布局列保留实际维度，省略物理 tiling；完整 shape 与通信 groups 见同名 JSON。
异步算子用短名表示完整调用，省略 cloned/call-done 等包装后缀；JSON 保留承接结果的完整 HLO 名称及行号。

| 编号 | 输入来自 | 编译后算子 | 作用 | 输出 shape |
|---|---|---|---|---|
| N01 | 上游层输出/统计量, 权重/scale, 跨层共享准备 | `abs_reduce_fusion.177` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N02 | N01 | `clamp_convert_fusion.132` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N03 | N01, N02, 权重/scale | `multiply_convert_fusion.43` | GDN 输入投影 | `bf16[4096,20480]` |
| N04 | N03 | `slice_convert_fusion.43` | GDN 输入准备 | `f32[4096,12288]` |
| N05 | N04 | `reshape.9234` | 数据复制/布局物化 | `f32[4096,1,12288]` |
| N06 | N01, 权重/scale | `bitcast_convert_fusion.57` | GDN 门控投影 | `f32[4096,1,128], bf16[4096,128]` |
| N07 | 常量/调度 | `broadcast_in_dim.3673.clone.1` | 索引/张量辅助 | `bf16[4096,64,128]` |
| N08 | N05, N06, N07, 上游层输出/统计量, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_conv1d_gdn_per_seq.91` | GDN 核心 | `bf16[4096,64,128], f8e4m3fn[15642,4,64,4,128]` |
| N09 | N08 | `fusion.4` | RMSNorm/residual | `f32[4096,64]` |
| N10 | N03, N08, N09, 权重/scale | `abs_reduce_fusion.58` | FP8 激活量化 | `bf16[4096], bf16[4096,64,128]` |
| N11 | N10 | `convert.1119` | FP32 中间计算 | `f32[512,8,64,128]` |
| N12 | N10, N11 | `fusion.256` | FP8 激活量化 | `f8e4m3fn[4096,8192]` |
| N13 | N10, N12, 上游层输出/统计量, 权重/scale, 跨层共享准备 | `multiply_reduce_fusion.117` | attention 输出投影 | `f32[4096], bf16[4096,4096], bf16[4096,4096]` |
| N14 | N13, 权重/scale | `abs_reduce_fusion.176` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N15 | N14, 权重/scale | `fusion.3142` | Router 打分 | `bf16[4096], bf16[4096,512]` |
| N16 | N15 | `copy.5999` | 数据复制/布局物化 | `bf16[4096,512]` |
| N17 | N15, N16 | `fusion.793` | MoE 路由/通信计划 | `f32[4096]` |
| N18 | N15, N16, N17 | `is-finite_reduce_fusion.58` | MoE 路由/通信计划 | `pred[4096], f32[4096,512]` |
| N19 | N18 | `shard_map.181` | Router top-k | `f32[4096,10], s32[4096,10]` |
| N20 | N19 | `fusion.827` | MoE 路由/通信计划 | `s32[4096,10]` |
| N21 | N20 | `broadcast_in_dim_reshape.365` | 数据复制/布局物化 | `s32[160,256]` |
| N22 | N20 | `reshape_reshape.181` | 数据复制/布局物化 | `s32[40960]` |
| N23 | N21 | `convert_reduce_fusion.235` | MoE 路由/通信计划 | `s32[160,512], s32[512]` |
| N24 | N14 | `fusion.497` | MoE 激活量化 | `f8e4m3fn[512,8,32,128]` |
| N25 | N24 | `copy.8010` | 数据复制/布局物化 | `f8e4m3fn[512,8,32,128]` |
| N26 | 权重/scale | `copy.8808` | 数据复制/布局物化 | `f32[64,1,1,4096]` |
| N27 | N19 | `reduce.4499` | MoE 路由/通信计划 | `f32[4096]` |
| N28 | N14, 权重/scale | `convolution_reduce_fusion.58` | 共享专家 gate | `bf16[4096]` |
| N29 | N14 | `clamp_convert_fusion.131` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N30 | N14, N29, 权重/scale | `multiply_convert_fusion.358` | 共享专家 gate/up | `bf16[4096,2048]` |
| N31 | N30 | `abs_reduce_fusion.238` | FP8 激活量化 | `bf16[4096]` |
| N32 | N21, 跨层共享准备 | `convert_reduce_fusion.357` | MoE 路由/通信计划 | `s32[160,256]` |
| N33 | N30, N31 | `clamp_convert_fusion.193` | FP8 激活量化 | `f8e4m3fn[4096,1024]` |
| N34 | N14, N22, N23 | `all-gather.303` | EP dispatch / all-gather | `s32[8,1,45568]` |
| N35 | N34 | `copy.6002` | EP 全收集 | `s32[8,1,45568]` |
| N36 | N35 | `slice.3015` | MoE 路由/通信计划 | `s32[8,40960]` |
| N37 | N35 | `reduce-window.1368` | MoE 路由/通信计划 | `s32[8,4,128]` |
| N38 | N37 | `slice_reduce_fusion.58` | MoE 路由/通信计划 | `s32[8,4]` |
| N39 | N35, N37, N38 | `add_subtract_fusion.58` | MoE 路由/通信计划 | `s32[8,8,64]` |
| N40 | N35 | `fusion.3128` | MoE 路由/通信计划 | `s32[8], s32[512,8], s32[512,8]` |
| N41 | N40, 运行时输入/缓存 | `and_reduce_fusion.58` | MoE 路由/通信计划 | `s32[512]` |
| N42 | N21, N23, N39, N41, 运行时输入/缓存 | `select_reduce_fusion.116` | MoE 路由/通信计划 | `s32[160,256], s32[160,256]` |
| N43 | N32, N42 | `reshape.7138` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N44 | N32, N42 | `reshape.7139` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N45 | N43, N44 | `fusion.893` | MoE 路由/通信计划 | `s32[4096,10,1]` |
| N46 | N45 | `reshape.9584` | 数据复制/布局物化 | `s32[40960]` |
| N47 | N36 | `copy.6008` | 数据复制/布局物化 | `s32[8,40960]` |
| N48 | N47 | `reshape.7126` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N49 | N48 | `copy.6009` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N50 | N49, 跨层共享准备 | `convert_reduce_fusion.233` | MoE 路由/通信计划 | `s32[1280,64], s32[1280,8,64]` |
| N51 | N50 | `reduce-window.1374` | MoE 路由/通信计划 | `s32[8,64,2,128]` |
| N52 | N50, 跨层共享准备 | `select_reduce_fusion.178` | MoE 路由/通信计划 | `s32[1280,8,64]` |
| N53 | N49, 跨层共享准备 | `convert_reduce_fusion.356` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N54 | N35, N40, N49, N50, N51, N52, 跨层共享准备, 运行时输入/缓存 | `fusion.301` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N55 | N53, N54 | `add.21151` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N56 | N55 | `copy.6018` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N57 | N56 | `reshape.7131` | 数据复制/布局物化 | `s32[327680]` |
| N58 | N35, 运行时输入/缓存 | `compare_and_fusion.58` | MoE 路由/通信计划 | `pred[8,40960]` |
| N59 | N58 | `copy.6007` | 数据复制/布局物化 | `pred[8,40960]` |
| N60 | N35 | `broadcast_in_dim.3719` | MoE 路由/通信计划 | `s32[32768,10]` |
| N61 | N60 | `reshape.9982` | 数据复制/布局物化 | `s32[327680]` |
| N62 | N35, N40, 跨层共享准备, 运行时输入/缓存 | `multiply_reduce_fusion.418` | MoE 路由/通信计划 | `s32[8], s32[8], s32[8], s32[8] …（共 8 个 tuple 分量）` |
| N63 | N35, 运行时输入/缓存 | `convert_reduce_fusion.418` | MoE 路由/通信计划 | `s32[64]` |
| N64 | N25 | `all-gather.183` | EP dispatch / all-gather | `f8e4m3fn[32768,32,128]` |
| N65 | N57, N59, 运行时输入/缓存 | `compare_select_fusion.296` | MoE 返回地址/索引 | `s32[327680], s32[327680]` |
| N66 | N65, 跨层共享准备 | `scatter_offload.2` | SparseCore scatter（完整调用） | `s32[328320]` |
| N67 | N66 | `broadcast_clamp_fusion.58` | MoE 路由/通信计划 | `s32[328320]` |
| N68 | N67 | `pad.593` | MoE 路由/通信计划 | `s32[328576]` |
| N69 | N61, N65, 跨层共享准备 | `scatter_offload.3` | SparseCore scatter（完整调用） | `s32[328448]` |
| N70 | N69 | `bitcast-convert_bitcast_fusion.58` | MoE 路由/通信计划 | `f32[2566,128]` |
| N71 | N26, N35, N39, N40, N62, N63, N64, N68, N70, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_ep_moe_v2_g64_c128_nb3.181` | 路由专家计算 | `f8e4m3fn[41472,32,128], f32[1350,128], f8e4m3fn[328320,32,128], f32[2695,128]` |
| N72 | N46, N71 | `gather_offload.2` | SparseCore gather（完整调用） | `f32[40960]` |
| N73 | N72 | `reshape.9587` | 数据复制/布局物化 | `f32[4096,10]` |
| N74 | N18, N19, N27, N73 | `select_multiply_fusion.58` | MoE 路由/通信计划 | `f32[4096,10]` |
| N75 | N74 | `reshape.7140` | 数据复制/布局物化 | `f32[40960]` |
| N76 | N32, N42, N71, N75 | `moe_v2_combine_t4096_k10_b64.181` | 路由专家合并 | `bf16[4096,4096]` |
| N77 | N13, N28, N31, N33, N76, 权重/scale | `multiply_reduce_fusion.116` | 共享专家 down/合并 | `f32[4096], bf16[4096,4096]` |

[完整依赖与 HLO 行号（JSON）](dp8-gdn.json)
