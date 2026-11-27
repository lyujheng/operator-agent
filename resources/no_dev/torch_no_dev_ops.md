# PyTorch 无需开发算子

共 214 个算子，按功能类别分组。判断原则：不在 device 端启动 compute kernel 的算子无需 FlagGems 开发。

## 视图/形状变换（56）

纯 metadata 操作，只修改 stride/shape/offset，不启动任何 device kernel。

| # | 算子名 |
|---|--------|
| 1 | _conj |
| 2 | _mkldnn_reshape |
| 3 | _mkldnn_transpose |
| 4 | _mkldnn_transpose_ |
| 5 | _reshape_alias |
| 6 | _reshape_from_tensor |
| 7 | _unsafe_view |
| 8 | alias |
| 9 | as_strided |
| 10 | as_strided_ |
| 11 | broadcast_tensors |
| 12 | broadcast_to |
| 13 | chunk |
| 14 | dsplit |
| 15 | empty_permuted |
| 16 | expand |
| 17 | expand_as |
| 18 | flatten |
| 19 | hsplit |
| 20 | lift |
| 21 | lift_fresh |
| 22 | moveaxis |
| 23 | movedim |
| 24 | narrow |
| 25 | permute |
| 26 | ravel |
| 27 | real |
| 28 | reshape |
| 29 | reshape_as |
| 30 | select |
| 31 | slice |
| 32 | slice_inverse |
| 33 | split |
| 34 | split_with_sizes |
| 35 | squeeze |
| 36 | squeeze_ |
| 37 | swapaxes |
| 38 | swapaxes_ |
| 39 | swapdims |
| 40 | swapdims_ |
| 41 | t_ |
| 42 | tensor_split |
| 43 | transpose |
| 44 | transpose_ |
| 45 | unbind |
| 46 | unflatten |
| 47 | unfold |
| 48 | unsafe_chunk |
| 49 | unsafe_split |
| 50 | unsafe_split_with_sizes |
| 51 | unsqueeze |
| 52 | unsqueeze_ |
| 53 | view |
| 54 | view_as |
| 55 | view_as_complex |
| 56 | vsplit |

## 属性查询（44）

只返回 tensor 属性或运行时状态，纯 host 读取。

| # | 算子名 |
|---|--------|
| 1 | _debug_has_internal_overlap |
| 2 | _has_compatible_shallow_copy_type |
| 3 | _is_zerotensor |
| 4 | _nnpack_available |
| 5 | _use_cudnn_ctc_loss |
| 6 | _use_cudnn_rnn_flatten_weight |
| 7 | _use_miopen_ctc_loss |
| 8 | cudnn_is_acceptable |
| 9 | get_device |
| 10 | indices |
| 11 | is_coalesced |
| 12 | is_complex |
| 13 | is_conj |
| 14 | is_contiguous |
| 15 | is_distributed |
| 16 | is_floating_point |
| 17 | is_inference |
| 18 | is_leaf |
| 19 | is_neg |
| 20 | is_non_overlapping_and_dense |
| 21 | is_pinned |
| 22 | is_set_to |
| 23 | is_signed |
| 24 | is_vulkan_available |
| 25 | matrix_H |
| 26 | mH |
| 27 | mT |
| 28 | numel |
| 29 | numpy_T |
| 30 | output_nr |
| 31 | promote_types |
| 32 | q_per_channel_axis |
| 33 | q_per_channel_scales |
| 34 | q_per_channel_zero_points |
| 35 | q_scale |
| 36 | q_zero_point |
| 37 | qscheme |
| 38 | result_type |
| 39 | retains_grad |
| 40 | row_indices |
| 41 | size |
| 42 | storage_offset |
| 43 | stride |
| 44 | values |

## 平台专属实现（37）

绑定 MKL-DNN / MPS / FBGEMM / NNPACK / CPU fallback / sparse 后端，与 Triton GPU backend 无关。

| # | 算子名 |
|---|--------|
| 1 | _convert_weight_to_int4pack_for_cpu |
| 2 | _grid_sampler_2d_cpu_fallback |
| 3 | _grid_sampler_2d_cpu_fallback_backward |
| 4 | _lstm_mps |
| 5 | _mps_convolution |
| 6 | _mps_convolution_transpose |
| 7 | _nnpack_spatial_convolution |
| 8 | _scaled_dot_product_flash_attention_for_cpu |
| 9 | _scaled_dot_product_flash_attention_for_cpu_backward |
| 10 | _sparse_mm_reduce_impl |
| 11 | _sparse_mm_reduce_impl_backward |
| 12 | fbgemm_linear_fp16_weight |
| 13 | fbgemm_linear_fp16_weight_fp32_activation |
| 14 | fbgemm_linear_int8_weight |
| 15 | fbgemm_linear_int8_weight_fp32_activation |
| 16 | fbgemm_linear_quantize_weight |
| 17 | fbgemm_pack_gemm_matrix_fp16 |
| 18 | fbgemm_pack_quantized_matrix |
| 19 | lstm_mps_backward |
| 20 | mkldnn_adaptive_avg_pool2d |
| 21 | mkldnn_adaptive_avg_pool2d_backward |
| 22 | mkldnn_convolution |
| 23 | mkldnn_linear |
| 24 | mkldnn_linear_backward |
| 25 | mkldnn_linear_backward_input |
| 26 | mkldnn_linear_backward_weights |
| 27 | mkldnn_max_pool2d |
| 28 | mkldnn_max_pool2d_backward |
| 29 | mkldnn_max_pool3d |
| 30 | mkldnn_max_pool3d_backward |
| 31 | mkldnn_reorder_conv2d_weight |
| 32 | mkldnn_reorder_conv3d_weight |
| 33 | mkldnn_rnn_layer_backward |
| 34 | mps_convolution_backward |
| 35 | mps_convolution_transpose_backward |
| 36 | to_mkldnn |
| 37 | to_mkldnn_backward |

## Copy 变体（18）

view 操作的 copy 版本，行为是 view + contiguous copy。

| # | 算子名 |
|---|--------|
| 1 | _nested_view_from_buffer_copy |
| 2 | _nested_view_from_jagged_copy |
| 3 | _reshape_alias_copy |
| 4 | alias_copy |
| 5 | fw_primal_copy |
| 6 | indices_copy |
| 7 | lift_fresh_copy |
| 8 | make_dual_copy |
| 9 | narrow_copy |
| 10 | slice_copy |
| 11 | split_with_sizes_copy |
| 12 | squeeze_copy |
| 13 | unbind_copy |
| 14 | unsqueeze_copy |
| 15 | values_copy |
| 16 | view_as_complex_copy |
| 17 | view_as_real_copy |
| 18 | view_copy |

## 测试/内部工具（15）

框架内部工具函数和测试桩。

| # | 算子名 |
|---|--------|
| 1 | _foobar |
| 2 | _print |
| 3 | _test_ambiguous_defaults |
| 4 | _test_autograd_multiple_dispatch |
| 5 | _test_autograd_multiple_dispatch_view |
| 6 | _test_autograd_multiple_dispatch_view_copy |
| 7 | _test_check_tensor |
| 8 | _test_functorch_fallback |
| 9 | _test_optional_filled_intlist |
| 10 | _test_optional_floatlist |
| 11 | _test_optional_intlist |
| 12 | _test_parallel_materialize |
| 13 | _test_string_default |
| 14 | _test_warn_in_autograd |
| 15 | unflatten_dense_tensors |

## 符号/断言（12）

编译器/符号追踪辅助，纯 host 逻辑。

| # | 算子名 |
|---|--------|
| 1 | _assert_scalar |
| 2 | _assert_tensor_metadata |
| 3 | _functional_assert_async |
| 4 | _functional_assert_scalar |
| 5 | _functional_sym_constrain_range |
| 6 | _functional_sym_constrain_range_for_size |
| 7 | sym_constrain_range |
| 8 | sym_constrain_range_for_size |
| 9 | sym_is_contiguous |
| 10 | sym_numel |
| 11 | sym_storage_offset |
| 12 | sym_stride |

## 稀疏张量构造（12）

稀疏格式的构造和 resize，不涉及计算。

| # | 算子名 |
|---|--------|
| 1 | _sparse_bsc_tensor_unsafe |
| 2 | _sparse_bsr_tensor_unsafe |
| 3 | _sparse_compressed_tensor_unsafe |
| 4 | _sparse_compressed_tensor_with_dims |
| 5 | _sparse_coo_tensor_unsafe |
| 6 | _sparse_coo_tensor_with_dims |
| 7 | _sparse_coo_tensor_with_dims_and_tensors |
| 8 | _sparse_csc_tensor_unsafe |
| 9 | _sparse_csr_tensor_unsafe |
| 10 | sparse_resize_ |
| 11 | sparse_resize_and_clear_ |
| 12 | spdiags |

## 内存管理（10）

显存分配 / memcpy / 标量读取，无 compute kernel。

| # | 算子名 |
|---|--------|
| 1 | _copy_from |
| 2 | _copy_from_and_resize |
| 3 | _lazy_clone |
| 4 | _local_scalar_dense |
| 5 | _make_dep_token |
| 6 | _to_cpu |
| 7 | clone |
| 8 | copy |
| 9 | empty |
| 10 | resize |

## cuFFT 运行时管理（4）

cuFFT plan cache 管理，纯 host 配置操作。

| # | 算子名 |
|---|--------|
| 1 | _cufft_clear_plan_cache |
| 2 | _cufft_get_plan_cache_max_size |
| 3 | _cufft_get_plan_cache_size |
| 4 | _cufft_set_plan_cache_max_size |

## Autograd 控制（4）

自动微分框架控制面，纯 host 操作。

| # | 算子名 |
|---|--------|
| 1 | record_stream |
| 2 | requires_grad_ |
| 3 | retain_grad |
| 4 | set_data |

## Nested Tensor 视图（2）

| # | 算子名 |
|---|--------|
| 1 | _nested_view_from_buffer |
| 2 | _nested_view_from_jagged |
