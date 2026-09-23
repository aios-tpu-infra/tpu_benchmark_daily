# PCP8 prefill — layer 1（gdn）的编译后算子依赖

[返回计算流总览](../COMPUTE-FLOWS.md)

本表按最终 HLO 的 operand 依赖排列，编号用于引用上游算子，不是 trace 时间或 kernel launch 编号。同一算子产生多个输出时仍只有一个编号。

保留 ≥1 µs 的计算与布局算子；通信按完整逻辑调用保留，短 start/update/done 不独立列项。低于阈值的数值转换、包装节点和 DMA 被穿透，直接连到上游 producer。while 的循环体合并到循环节点。

表中列出的是数据依赖，不代表所有无边节点一定并发；资源、通信与缓存别名仍可引入执行约束。布局列保留实际维度，省略物理 tiling；完整 shape 与通信 groups 见同名 JSON。
异步算子用短名表示完整调用，省略 cloned/call-done 等包装后缀；JSON 保留承接结果的完整 HLO 名称及行号。

| 编号 | 输入来自 | 编译后算子 | 作用 | 输出 shape |
|---|---|---|---|---|
| N01 | 权重/scale, 跨层共享准备 | `reduce.10635` | PCP head/token 布局准备 | `bf16[12288,4]` |
| N02 | N01 | `copy.21638` | 数据复制/布局物化 | `bf16[12288,4]` |
| N03 | N02 | `reshape.30646` | 数据复制/布局物化 | `bf16[8,2,3072]` |
| N04 | N03 | `all_to_all.357` | PCP all-to-all | `bf16[8,2,3072]` |
| N05 | N04 | `reshape.30647` | 数据复制/布局物化 | `bf16[12288,1,4]` |
| N06 | 权重/scale | `all_to_all.359` | PCP all-to-all | `bf16[8,8,1]` |
| N07 | 权重/scale | `all_to_all.358` | PCP all-to-all | `f32[8,8,1]` |
| N08 | 上游层输出/统计量, 权重/scale, 跨层共享准备 | `multiply_add_fusion.133` | RMSNorm/residual | `bf16[4096,4096]` |
| N09 | N08, 权重/scale | `fusion.3439` | GDN 门控投影 | `bf16[4096,128]` |
| N10 | N09 | `copy.15233` | PCP head/token 布局准备 | `bf16[4096,64,2]` |
| N11 | N10 | `reshape.22899` | PCP all-to-all | `bf16[4096,8,16]` |
| N12 | N11 | `all_to_all.347` | PCP all-to-all | `bf16[4096,8,16]` |
| N13 | N12 | `copy.17135` | PCP all-to-all | `bf16[8,4096,16]` |
| N14 | N13, 跨层共享准备 | `scatter_custom_fusion.10` | PCP head/token 布局准备 | `bf16[32768,16]` |
| N15 | N14 | `copy.15236` | PCP head/token 布局准备 | `bf16[32768,16]` |
| N16 | N15 | `reshape.22900` | PCP head/token 布局准备 | `bf16[32768,8,2]` |
| N17 | N16 | `fusion.3367` | PCP head/token 布局准备 | `bf16[32768,8,1], bf16[32768,8,1]` |
| N18 | N17 | `copy.17588` | 数据复制/布局物化 | `bf16[32768,8,1]` |
| N19 | N18 | `pad_convert_fusion.87` | PCP head/token 布局准备 | `f32[32768,128]` |
| N20 | N17 | `copy.17589` | 数据复制/布局物化 | `bf16[32768,8,1]` |
| N21 | N20 | `pad_convert_fusion.86` | PCP head/token 布局准备 | `f32[32768,128]` |
| N22 | 常量/调度 | `broadcast_in_dim.4668.clone.1` | PCP head/token 布局准备 | `bf16[32768,8,128]` |
| N23 | 常量/调度 | `broadcast_in_dim.4669.clone.1` | PCP head/token 布局准备 | `bf16[4096,8,8,128]` |
| N24 | N05, N06, N07, N08, N19, N21, N22, N23, 上游层输出/统计量, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_pcp_qkvz_projection_gdn_per_seq_compact_qkv_c256_p8_pooled.91` | PCP GDN 核心 | `bf16[32768,8,128], bf16[4096,8,8,128], f8e4m3fn[64,1,4,64,4,256], f8e4m3fn[64,2,4,64,4,256] …（共 7 个 tuple 分量）` |
| N25 | N24 | `multiply_reduce_fusion.43` | RMSNorm/residual | `f32[4096,8,8]` |
| N26 | N25 | `copy.15244` | 数据复制/布局物化 | `f32[4096,8,8]` |
| N27 | N24 | `convert.1215` | FP32 中间计算 | `f32[512,8,64,128]` |
| N28 | N26 | `broadcast.97467` | RMSNorm/residual | `f32[4096,64,128]` |
| N29 | N24, N27, N28, 权重/scale | `abs_reduce_fusion.58` | FP8 激活量化 | `bf16[4096], bf16[4096,8192]` |
| N30 | N29 | `clamp_convert_fusion.58` | FP8 激活量化 | `f8e4m3fn[4096,8192]` |
| N31 | N29, N30, 上游层输出/统计量, 权重/scale, 跨层共享准备 | `multiply_reduce_fusion.162` | attention 输出投影 | `f32[4096], bf16[4096,4096], bf16[4096,4096]` |
| N32 | N31, 权重/scale | `abs_reduce_fusion.133` | FP8 激活量化 | `bf16[4096], bf16[4096,4096]` |
| N33 | N32, 权重/scale | `fusion.3379` | Router 打分 | `bf16[4096], bf16[4096,512]` |
| N34 | N33 | `copy.15246` | 数据复制/布局物化 | `bf16[4096,512]` |
| N35 | N33, N34 | `fusion.665` | MoE 路由/通信计划 | `f32[4096]` |
| N36 | N33, N34, N35 | `is-finite_reduce_fusion.58` | MoE 路由/通信计划 | `pred[4096], f32[4096,512]` |
| N37 | N36 | `shard_map.273` | Router top-k | `f32[4096,10], s32[4096,10]` |
| N38 | N37 | `fusion.789` | MoE 路由/通信计划 | `s32[4096,10]` |
| N39 | N38 | `broadcast_in_dim_reshape.372` | 数据复制/布局物化 | `s32[160,256]` |
| N40 | N38 | `reshape_reshape.121` | 数据复制/布局物化 | `s32[40960]` |
| N41 | N39 | `convert_reduce_fusion.235` | MoE 路由/通信计划 | `s32[160,512], s32[512]` |
| N42 | N32 | `fusion.369` | MoE 激活量化 | `f8e4m3fn[512,8,32,128]` |
| N43 | N42 | `copy.20632` | 数据复制/布局物化 | `f8e4m3fn[512,8,32,128]` |
| N44 | 权重/scale | `copy.21575` | 数据复制/布局物化 | `f32[64,1,1,4096]` |
| N45 | N37 | `reduce.5567` | MoE 路由/通信计划 | `f32[4096]` |
| N46 | N24, 上游层输出/统计量, 跨层共享准备, 运行时输入/缓存 | `while.3717` | PCP GDN 状态/调度循环；循环体 `conditional.542` ×4, `bitcast_dynamic-update-slice_fusion.2` ×4 | `s32[], f8e4m3fn[7764,4,64,4,256], s32[], s32[65] …（共 15 个 tuple 分量）` |
| N47 | N24, N46, 跨层共享准备 | `while.3718` | PCP GDN 状态/调度循环；循环体 `conditional.543` ×4, `bitcast_dynamic-update-slice_fusion.3` ×4 | `s32[], f8e4m3fn[7764,4,64,4,256], s32[], s32[65] …（共 15 个 tuple 分量）` |
| N48 | N32, 权重/scale | `convolution_reduce_fusion.58` | 共享专家 gate | `bf16[4096]` |
| N49 | N32 | `clamp_convert_fusion.133` | FP8 激活量化 | `f8e4m3fn[4096,4096]` |
| N50 | N32, N49, 权重/scale | `multiply_convert_fusion.313` | 共享专家 gate/up | `bf16[4096,2048]` |
| N51 | N50 | `abs_reduce_fusion.193` | FP8 激活量化 | `bf16[4096]` |
| N52 | N39, 跨层共享准备 | `convert_reduce_fusion.357` | MoE 路由/通信计划 | `s32[160,256]` |
| N53 | N50, N51 | `clamp_convert_fusion.193` | FP8 激活量化 | `f8e4m3fn[4096,1024]` |
| N54 | N32, N40, N41 | `all-gather.303` | EP dispatch / all-gather | `s32[8,1,45568]` |
| N55 | N54 | `copy.15249` | EP 全收集 | `s32[8,1,45568]` |
| N56 | N55 | `slice.10891` | MoE 路由/通信计划 | `s32[8,40960]` |
| N57 | N55 | `fusion.2748` | MoE 路由/通信计划 | `s32[8], s32[512], s32[8], s32[8,512] …（共 6 个 tuple 分量）` |
| N58 | N57 | `reduce-window.1909` | MoE 路由/通信计划 | `s32[8,4,128]` |
| N59 | N55, N57 | `fusion.3221` | MoE 路由/通信计划 | `s32[8], s32[512,8], s32[512,8]` |
| N60 | N39, N41, N57, N58, N59, 运行时输入/缓存 | `select_reduce_fusion.116` | MoE 路由/通信计划 | `s32[160,256], s32[160,256]` |
| N61 | N52, N60 | `reshape.22944` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N62 | N52, N60 | `reshape.22945` | 数据复制/布局物化 | `s32[4096,10,1]` |
| N63 | N61, N62 | `fusion.855` | MoE 路由/通信计划 | `s32[4096,10,1]` |
| N64 | N63 | `reshape.27144` | 数据复制/布局物化 | `s32[40960]` |
| N65 | N56 | `copy.15255` | 数据复制/布局物化 | `s32[8,40960]` |
| N66 | N65 | `reshape.22932` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N67 | N66 | `copy.15256` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N68 | N67, 跨层共享准备 | `convert_reduce_fusion.233` | MoE 路由/通信计划 | `s32[1280,64], s32[1280,8,64]` |
| N69 | N68 | `reduce-window.1915` | MoE 路由/通信计划 | `s32[8,64,2,128]` |
| N70 | N57, N59, N68, N69, 跨层共享准备, 运行时输入/缓存 | `fusion.1268` | MoE 路由/通信计划 | `s32[8,160,64]` |
| N71 | N68, 跨层共享准备 | `select_reduce_fusion.178` | MoE 路由/通信计划 | `s32[1280,8,64]` |
| N72 | N67, 跨层共享准备 | `convert_reduce_fusion.356` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N73 | N67, N70, N71, 跨层共享准备 | `fusion.173` | MoE 路由/通信计划 | `s32[1280,8,32]` |
| N74 | N72, N73 | `copy.15265` | 数据复制/布局物化 | `s32[1280,8,32]` |
| N75 | N74 | `reshape.22937` | 数据复制/布局物化 | `s32[327680]` |
| N76 | N55, 运行时输入/缓存 | `compare_and_fusion.59` | MoE 路由/通信计划 | `pred[8,40960]` |
| N77 | N76 | `copy.15254` | 数据复制/布局物化 | `pred[8,40960]` |
| N78 | N55 | `broadcast_in_dim.4717` | MoE 路由/通信计划 | `s32[32768,10]` |
| N79 | N78 | `reshape.27634` | 数据复制/布局物化 | `s32[327680]` |
| N80 | N57, 运行时输入/缓存 | `reduce-window.1913` | MoE 路由/通信计划 | `s32[64,8]` |
| N81 | N57, 跨层共享准备, 运行时输入/缓存 | `multiply_reduce_fusion.400` | MoE 路由/通信计划 | `s32[8], s32[64,8]` |
| N82 | N57, N59, 跨层共享准备, 运行时输入/缓存 | `multiply_reduce_fusion.464` | MoE 路由/通信计划 | `s32[8], s32[8], s32[8], s32[8] …（共 8 个 tuple 分量）` |
| N83 | N57, 运行时输入/缓存 | `select_reduce_fusion.239` | MoE 路由/通信计划 | `s32[64]` |
| N84 | N57, N58, 运行时输入/缓存 | `fusion.1515` | MoE 路由/通信计划 | `s32[64,8]` |
| N85 | N43 | `all-gather.183` | EP dispatch / all-gather | `f8e4m3fn[32768,32,128]` |
| N86 | N75, N77, 运行时输入/缓存 | `compare_select_fusion.299` | MoE 返回地址/索引 | `s32[327680], s32[327680]` |
| N87 | N57, 跨层共享准备, 运行时输入/缓存 | `broadcast_subtract_fusion.58` | MoE 路由/通信计划 | `s32[64]` |
| N88 | N86, 跨层共享准备 | `scatter_offload.4` | SparseCore scatter（完整调用） | `s32[328320]` |
| N89 | N88 | `broadcast_clamp_fusion.58` | MoE 路由/通信计划 | `s32[328320]` |
| N90 | N89 | `pad.1510` | MoE 路由/通信计划 | `s32[328576]` |
| N91 | N79, N86, 跨层共享准备 | `scatter_offload.5` | SparseCore scatter（完整调用） | `s32[328448]` |
| N92 | N91 | `bitcast-convert_bitcast_fusion.58` | MoE 路由/通信计划 | `f32[2566,128]` |
| N93 | N44, N57, N59, N80, N81, N82, N83, N84, N85, N87, N90, N92, 权重/scale, 跨层共享准备, 运行时输入/缓存 | `fused_ep_moe_v2_g64_c128_nb3.181` | 路由专家计算 | `f8e4m3fn[41472,32,128], f32[1350,128], f8e4m3fn[328320,32,128], f32[2695,128]` |
| N94 | N64, N93 | `gather_offload.3` | SparseCore gather（完整调用） | `f32[40960]` |
| N95 | N94 | `reshape.27147` | 数据复制/布局物化 | `f32[4096,10]` |
| N96 | N36, N37, N45, N95 | `select_multiply_fusion.58` | MoE 路由/通信计划 | `f32[4096,10]` |
| N97 | N96 | `reshape.22946` | 数据复制/布局物化 | `f32[40960]` |
| N98 | N52, N60, N93, N97 | `moe_v2_combine_t4096_k10_b64.181` | 路由专家合并 | `bf16[4096,4096]` |
| N99 | N31, N48, N51, N53, N98, 权重/scale | `multiply_reduce_fusion.161` | 共享专家 down/合并 | `f32[4096], bf16[4096,4096]` |

[完整依赖与 HLO 行号（JSON）](pcp8-gdn.json)
