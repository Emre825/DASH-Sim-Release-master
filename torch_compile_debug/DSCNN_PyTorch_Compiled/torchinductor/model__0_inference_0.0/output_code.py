# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_emre/cr/ccrmanhy7q3sd4evl2nuzcujh6ox6lylwcgdehqco2dqgu5t5j6f.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[1, 3, 25, 5][375, 125, 5, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [5, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1125, 'x': 1500}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 125
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 125*y0), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/cx/ccxh7lxc6bmfnkaqtcz7y2ddaw7adynmpzgwziu4ruwj5vpuyon2.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 3, 10, 4][120, 40, 4, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [5, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 61440, 'x': 30720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 40
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 40*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 120*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/fi/cfimjrtcnonqcvqksktmojbfpxtt4d75viobrj2xkkkrtn7hblx7.py
# Topologically Sorted Source Nodes: [conv2d, batch_norm, x], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   conv2d => convolution
#   x => relu
# Graph fragment:
#   %buf2 : Tensor "f32[1, 64, 13, 3][2496, 1, 192, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [5, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 31232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/yw/cyw3dtd4qrbp6t7cj5yy2dfleiymtfekc4ibcrkg6s26cwkuh4ic.py
# Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7, conv2d_7, batch_norm_7, x_8, conv2d_8, batch_norm_8, x_9, x_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.avg_pool2d]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   batch_norm_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   batch_norm_2 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   batch_norm_3 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   batch_norm_4 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   batch_norm_5 => add_10, add_11, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   batch_norm_6 => add_12, add_13, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   batch_norm_7 => add_14, add_15, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   batch_norm_8 => add_16, add_17, mul_24, mul_25, mul_26, reciprocal_8, sqrt_8, sub_8, unsqueeze_64, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, unsqueeze_71
#   conv2d => convolution
#   conv2d_1 => convolution_1
#   conv2d_2 => convolution_2
#   conv2d_3 => convolution_3
#   conv2d_4 => convolution_4
#   conv2d_5 => convolution_5
#   conv2d_6 => convolution_6
#   conv2d_7 => convolution_7
#   conv2d_8 => convolution_8
#   x => relu
#   x_11 => avg_pool2d
#   x_2 => relu_1
#   x_3 => relu_2
#   x_4 => relu_3
#   x_5 => relu_4
#   x_6 => relu_5
#   x_7 => relu_6
#   x_8 => relu_7
#   x_9 => relu_8
# Graph fragment:
#   %relu_8 : Tensor "f32[1, 64, 13, 3][2496, 1, 192, 64]cuda:0" = PlaceHolder[target=relu_8]
#   %convolution : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [5, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg9_1, %arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg13_1, %arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_5 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %add_10 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_10,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_11 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg17_1, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_48 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_12 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_13 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_14 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_15 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg21_1, %arg22_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %sub_8 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %add_16 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_8,), kwargs = {})
#   %mul_24 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_24, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %mul_25 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, -1), kwargs = {})
#   %mul_26 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, -1), kwargs = {})
#   %add_17 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : Tensor "f32[1, 64, 13, 3][2496, 39, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   %avg_pool2d : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_8, [12, 2], [12, 2]), kwargs = {})
#   return %avg_pool2d
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 24, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6656}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (384 + x0), xmask)
    tmp9 = tl.load(in_ptr0 + (448 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (576 + x0), xmask)
    tmp13 = tl.load(in_ptr0 + (640 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp17 = tl.load(in_ptr0 + (832 + x0), xmask)
    tmp19 = tl.load(in_ptr0 + (960 + x0), xmask)
    tmp21 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp23 = tl.load(in_ptr0 + (1152 + x0), xmask)
    tmp25 = tl.load(in_ptr0 + (1216 + x0), xmask)
    tmp27 = tl.load(in_ptr0 + (1344 + x0), xmask)
    tmp29 = tl.load(in_ptr0 + (1408 + x0), xmask)
    tmp31 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp33 = tl.load(in_ptr0 + (1600 + x0), xmask)
    tmp35 = tl.load(in_ptr0 + (1728 + x0), xmask)
    tmp37 = tl.load(in_ptr0 + (1792 + x0), xmask)
    tmp39 = tl.load(in_ptr0 + (1920 + x0), xmask)
    tmp41 = tl.load(in_ptr0 + (1984 + x0), xmask)
    tmp43 = tl.load(in_ptr0 + (2112 + x0), xmask)
    tmp45 = tl.load(in_ptr0 + (2176 + x0), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp40 = tmp39 + tmp38
    tmp42 = tmp41 + tmp40
    tmp44 = tmp43 + tmp42
    tmp46 = tmp45 + tmp44
    tmp47 = 0.041666666666666664
    tmp48 = tmp46 * tmp47
    tl.store(out_ptr0 + (x0), tmp48, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 3, 10, 4), (120, 40, 4, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (1, 3, 25, 5), (375, 125, 5, 1))
        assert_size_stride(arg3_1, (64, ), (1, ))
        assert_size_stride(arg4_1, (64, ), (1, ))
        assert_size_stride(arg5_1, (64, ), (1, ))
        assert_size_stride(arg6_1, (64, ), (1, ))
        assert_size_stride(arg7_1, (64, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg8_1, (64, ), (1, ))
        assert_size_stride(arg9_1, (64, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg10_1, (64, ), (1, ))
        assert_size_stride(arg11_1, (64, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg12_1, (64, ), (1, ))
        assert_size_stride(arg13_1, (64, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg14_1, (64, ), (1, ))
        assert_size_stride(arg15_1, (64, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg16_1, (64, ), (1, ))
        assert_size_stride(arg17_1, (64, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg18_1, (64, ), (1, ))
        assert_size_stride(arg19_1, (64, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg20_1, (64, ), (1, ))
        assert_size_stride(arg21_1, (64, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg22_1, (64, ), (1, ))
        assert_size_stride(arg23_1, (10, 64), (64, 1))
        assert_size_stride(arg24_1, (10, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 3, 25, 5), (375, 1, 15, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 3, 125, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 3, 10, 4), (120, 1, 12, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 40, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(5, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf3, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg1_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:5
            buf4 = extern_kernels.convolution(buf3, arg7_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
            assert_size_stride(buf4, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg7_1
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf5, arg8_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg8_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:7
            buf6 = extern_kernels.convolution(buf5, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf6, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg9_1
            del buf5
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf7, arg10_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg10_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:9
            buf8 = extern_kernels.convolution(buf7, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
            assert_size_stride(buf8, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg11_1
            del buf7
            buf9 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf9, arg12_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg12_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:11
            buf10 = extern_kernels.convolution(buf9, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf10, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg13_1
            del buf9
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:12
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf11, arg14_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg14_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:13
            buf12 = extern_kernels.convolution(buf11, arg15_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
            assert_size_stride(buf12, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg15_1
            del buf11
            buf13 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:14
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf13, arg16_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg16_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:15
            buf14 = extern_kernels.convolution(buf13, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf14, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg17_1
            del buf13
            buf15 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:16
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf15, arg18_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg18_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7, conv2d_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:17
            buf16 = extern_kernels.convolution(buf15, arg19_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
            assert_size_stride(buf16, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg19_1
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7, conv2d_7, batch_norm_7, x_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:18
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf17, arg20_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg20_1
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7, conv2d_7, batch_norm_7, x_8, conv2d_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:19
            buf18 = extern_kernels.convolution(buf17, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf18, (1, 64, 13, 3), (2496, 1, 192, 64), 'torch.ops.aten.convolution.default')
            del arg21_1
            del buf17
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7, conv2d_7, batch_norm_7, x_8, conv2d_8, batch_norm_8, x_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2:20
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf19, arg22_1, arg3_1, arg4_1, arg5_1, arg6_1, 2496, stream=stream0)
            del arg22_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf20 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d, batch_norm, x, conv2d_1, batch_norm_1, x_2, conv2d_2, batch_norm_2, x_3, conv2d_3, batch_norm_3, x_4, conv2d_4, batch_norm_4, x_5, conv2d_5, batch_norm_5, x_6, conv2d_6, batch_norm_6, x_7, conv2d_7, batch_norm_7, x_8, conv2d_8, batch_norm_8, x_9, x_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.avg_pool2d]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_3:21
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_3.run(buf19, buf20, 64, stream=stream0)
            del buf19
            buf21 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:22
            extern_kernels.addmm(arg24_1, reinterpret_tensor(buf20, (1, 64), (64, 1), 0), reinterpret_tensor(arg23_1, (64, 10), (1, 64), 0), alpha=1, beta=1, out=buf21)
            del arg23_1
            del arg24_1
            del buf20
        return (buf21, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 10, 4), (120, 40, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 25, 5), (375, 125, 5, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((10, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
