# vLLM 无需开发算子

共 8 个算子，均为 compile_wrapper 类。判断原则：自身不包含可独立开发的计算逻辑的算子无需 FlagGems-vllm 开发。

## compile_wrapper（8）

通过 `direct_register_custom_op` 注册为 `torch.ops.vllm.*` custom op，
使用 `@eager_break_during_capture` 装饰器实现 `torch.compile` graph break，
函数体仅从 `forward_context.no_compile_layers[layer_name]` 获取 layer 实例后
dispatch 到其方法，不包含任何独立 tensor 计算。

| # | 算子名 |
|---|--------|
| 1 | qwen_gdn_attention_core |
| 2 | qwen_gdn_attention_core_fused_norm_packed |
| 3 | qwen_gdn_attention_core_fused_norm_rocm |
| 4 | olmo_hybrid_gdn_full_forward |
| 5 | mamba_mixer |
| 6 | mamba_mixer2 |
| 7 | short_conv_forward |
| 8 | minimax_linear_attn_core |
