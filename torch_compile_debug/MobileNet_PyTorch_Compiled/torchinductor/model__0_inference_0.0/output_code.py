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


# kernel path: /tmp/torchinductor_emre/yg/cygfmqol4iwogcussdtlyqvuxcx6sql73mvp2rpzakg4haevlugt.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[1, 3, 96, 96][27648, 9216, 96, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 82944, 'x': 110592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 9216*y0), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/6v/c6vh6v7rhpvc3zoklwxrmnryeiftvfeswi6jktds3zsighympeif.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[8, 3, 3, 3][27, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1728, 'x': 864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/75/c75ujuj474wic2c7rpyr4v5ht343rm2ne4ulvfcfvtc7cfczutrr.py
# Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x => relu
# Graph fragment:
#   %convolution : Tensor "f32[1, 8, 48, 48][18432, 1, 384, 8]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[8, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[1, 8, 48, 48][18432, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 221312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/u6/cu6qpzvi5w7jecbdwxabxm35m7ffnkzuj2jdsoce2vqa3hg36du7.py
# Topologically Sorted Source Nodes: [batch_norm_2, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   x_2 => relu_2
# Graph fragment:
#   %convolution_2 : Tensor "f32[1, 16, 48, 48][36864, 1, 768, 16]cuda:0" = PlaceHolder[target=convolution_2]
#   %arg12_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %unsqueeze_16 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[1, 16, 48, 48][36864, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 48, 48][36864, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[1, 16, 48, 48][36864, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 48, 48][36864, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 48, 48][36864, 2304, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   return %relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 442624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/wi/cwibgphf26mqkyb2trgod2h3gq647tvyfb7acoc3lhemiodca7w4.py
# Topologically Sorted Source Nodes: [batch_norm_3, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   x_3 => relu_3
# Graph fragment:
#   %convolution_3 : Tensor "f32[1, 16, 24, 24][9216, 1, 384, 16]cuda:0" = PlaceHolder[target=convolution_3]
#   %arg17_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %unsqueeze_24 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 16, 24, 24][9216, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[1, 16, 24, 24][9216, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 16, 24, 24][9216, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[1, 16, 24, 24][9216, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 24, 24][9216, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   return %relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 110848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/em/cemyoegc25lcduoxip5chw3zc7exgigt5psiodpbo5bhgmfiqvse.py
# Topologically Sorted Source Nodes: [batch_norm_4, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_4 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   x_4 => relu_4
# Graph fragment:
#   %convolution_4 : Tensor "f32[1, 32, 24, 24][18432, 1, 768, 32]cuda:0" = PlaceHolder[target=convolution_4]
#   %arg22_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %unsqueeze_32 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[1, 32, 24, 24][18432, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[1, 32, 24, 24][18432, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[1, 32, 24, 24][18432, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[1, 32, 24, 24][18432, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 24, 24][18432, 576, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
#   return %relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 221696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/st/cst7pgxf4e7bsjojwn4aticycbbmsxavmmlctljzybif7xibdbjc.py
# Topologically Sorted Source Nodes: [batch_norm_7, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_7 => add_14, add_15, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   x_7 => relu_7
# Graph fragment:
#   %convolution_7 : Tensor "f32[1, 32, 12, 12][4608, 1, 384, 32]cuda:0" = PlaceHolder[target=convolution_7]
#   %arg37_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %unsqueeze_56 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[1, 32, 12, 12][4608, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_14 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[1, 32, 12, 12][4608, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 12, 12][4608, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_15 : Tensor "f32[1, 32, 12, 12][4608, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : Tensor "f32[1, 32, 12, 12][4608, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   return %relu_7
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 55808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/je/cjepvdcpyr7cickilp3rsjb2esatgdyyfmb47nqv4sdycpjsx5ra.py
# Topologically Sorted Source Nodes: [batch_norm_8, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_8 => add_16, add_17, mul_24, mul_25, mul_26, reciprocal_8, sqrt_8, sub_8, unsqueeze_64, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, unsqueeze_71
#   x_8 => relu_8
# Graph fragment:
#   %convolution_8 : Tensor "f32[1, 64, 12, 12][9216, 1, 768, 64]cuda:0" = PlaceHolder[target=convolution_8]
#   %arg42_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %arg43_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %arg44_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %unsqueeze_64 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %sub_8 : Tensor "f32[1, 64, 12, 12][9216, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %add_16 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg43_1, 1e-05), kwargs = {})
#   %sqrt_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_8,), kwargs = {})
#   %mul_24 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_24, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %mul_25 : Tensor "f32[1, 64, 12, 12][9216, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg44_1, -1), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, -1), kwargs = {})
#   %mul_26 : Tensor "f32[1, 64, 12, 12][9216, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg45_1, -1), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, -1), kwargs = {})
#   %add_17 : Tensor "f32[1, 64, 12, 12][9216, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : Tensor "f32[1, 64, 12, 12][9216, 144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   return %relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 111616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/cj/ccjxtsxrcoo6z7qum34gpyay34voydryaokarnflae6gxivco4s4.py
# Topologically Sorted Source Nodes: [batch_norm_11, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_22, add_23, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   x_11 => relu_11
# Graph fragment:
#   %convolution_11 : Tensor "f32[1, 64, 6, 6][2304, 1, 384, 64]cuda:0" = PlaceHolder[target=convolution_11]
#   %arg57_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %arg59_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %unsqueeze_88 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[1, 64, 6, 6][2304, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_22 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_22,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[1, 64, 6, 6][2304, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[1, 64, 6, 6][2304, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_23 : Tensor "f32[1, 64, 6, 6][2304, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_11 : Tensor "f32[1, 64, 6, 6][2304, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
#   return %relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 28672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/kv/ckvt5ifet5wo5jct4xfss6mwxavnrjjebqc7uqaiybjogopdmncr.py
# Topologically Sorted Source Nodes: [batch_norm_12, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_12 => add_24, add_25, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   x_12 => relu_12
# Graph fragment:
#   %convolution_12 : Tensor "f32[1, 128, 6, 6][4608, 1, 768, 128]cuda:0" = PlaceHolder[target=convolution_12]
#   %arg62_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %arg63_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %unsqueeze_96 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[1, 128, 6, 6][4608, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_24 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_24,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[1, 128, 6, 6][4608, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[1, 128, 6, 6][4608, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_25 : Tensor "f32[1, 128, 6, 6][4608, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : Tensor "f32[1, 128, 6, 6][4608, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
#   return %relu_12
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 57344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/mt/cmtn7hwd63gxihr6v5exlemrup5m2dc2mqprvrfyeenm3tcsyadq.py
# Topologically Sorted Source Nodes: [batch_norm_23, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_23 => add_46, add_47, mul_69, mul_70, mul_71, reciprocal_23, sqrt_23, sub_23, unsqueeze_184, unsqueeze_185, unsqueeze_186, unsqueeze_187, unsqueeze_188, unsqueeze_189, unsqueeze_190, unsqueeze_191
#   x_23 => relu_23
# Graph fragment:
#   %convolution_23 : Tensor "f32[1, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=convolution_23]
#   %arg117_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg117_1]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %arg119_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg119_1]
#   %arg120_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg120_1]
#   %unsqueeze_184 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg117_1, -1), kwargs = {})
#   %unsqueeze_185 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_184, -1), kwargs = {})
#   %sub_23 : Tensor "f32[1, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %add_46 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg118_1, 1e-05), kwargs = {})
#   %sqrt_23 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_46,), kwargs = {})
#   %reciprocal_23 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_23,), kwargs = {})
#   %mul_69 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_23, 1), kwargs = {})
#   %unsqueeze_186 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_69, -1), kwargs = {})
#   %unsqueeze_187 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_186, -1), kwargs = {})
#   %mul_70 : Tensor "f32[1, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %unsqueeze_188 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg119_1, -1), kwargs = {})
#   %unsqueeze_189 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_188, -1), kwargs = {})
#   %mul_71 : Tensor "f32[1, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_189), kwargs = {})
#   %unsqueeze_190 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg120_1, -1), kwargs = {})
#   %unsqueeze_191 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_190, -1), kwargs = {})
#   %add_47 : Tensor "f32[1, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_191), kwargs = {})
#   %relu_23 : Tensor "f32[1, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   return %relu_23
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/pf/cpfyw3y432wdfhwtm2fgkctatslxp5pdmugli73wtl2epaynokxo.py
# Topologically Sorted Source Nodes: [batch_norm_24, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_24 => add_48, add_49, mul_72, mul_73, mul_74, reciprocal_24, sqrt_24, sub_24, unsqueeze_192, unsqueeze_193, unsqueeze_194, unsqueeze_195, unsqueeze_196, unsqueeze_197, unsqueeze_198, unsqueeze_199
#   x_24 => relu_24
# Graph fragment:
#   %convolution_24 : Tensor "f32[1, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=convolution_24]
#   %arg122_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg122_1]
#   %arg123_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %arg124_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg124_1]
#   %arg125_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %unsqueeze_192 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg122_1, -1), kwargs = {})
#   %unsqueeze_193 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_192, -1), kwargs = {})
#   %sub_24 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %add_48 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, 1e-05), kwargs = {})
#   %sqrt_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_48,), kwargs = {})
#   %reciprocal_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_24,), kwargs = {})
#   %mul_72 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_24, 1), kwargs = {})
#   %unsqueeze_194 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_72, -1), kwargs = {})
#   %unsqueeze_195 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_194, -1), kwargs = {})
#   %mul_73 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %unsqueeze_196 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg124_1, -1), kwargs = {})
#   %unsqueeze_197 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_196, -1), kwargs = {})
#   %mul_74 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %unsqueeze_198 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg125_1, -1), kwargs = {})
#   %unsqueeze_199 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_198, -1), kwargs = {})
#   %add_49 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_49,), kwargs = {})
#   return %relu_24
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 31744}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/q4/cq4oujxxsywdnq2ruwvz63vfzsf4k5r7vy3c4srrk5kys4r6clys.py
# Topologically Sorted Source Nodes: [batch_norm_26, x_26, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   batch_norm_26 => add_52, add_53, mul_78, mul_79, mul_80, reciprocal_26, sqrt_26, sub_26, unsqueeze_208, unsqueeze_209, unsqueeze_210, unsqueeze_211, unsqueeze_212, unsqueeze_213, unsqueeze_214, unsqueeze_215
#   x_26 => relu_26
#   x_27 => mean
# Graph fragment:
#   %convolution_26 : Tensor "f32[1, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=convolution_26]
#   %arg132_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg132_1]
#   %arg133_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg133_1]
#   %arg134_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg134_1]
#   %arg135_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg135_1]
#   %buf55 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=buf55]
#   %unsqueeze_208 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg132_1, -1), kwargs = {})
#   %unsqueeze_209 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_208, -1), kwargs = {})
#   %sub_26 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %add_52 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg133_1, 1e-05), kwargs = {})
#   %sqrt_26 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_52,), kwargs = {})
#   %reciprocal_26 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_26,), kwargs = {})
#   %mul_78 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_26, 1), kwargs = {})
#   %unsqueeze_210 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_78, -1), kwargs = {})
#   %unsqueeze_211 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_210, -1), kwargs = {})
#   %mul_79 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %unsqueeze_212 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg134_1, -1), kwargs = {})
#   %unsqueeze_213 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_212, -1), kwargs = {})
#   %mul_80 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %unsqueeze_214 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg135_1, -1), kwargs = {})
#   %unsqueeze_215 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_214, -1), kwargs = {})
#   %add_53 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : Tensor "f32[1, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
#   %mean : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_26, [-1, -2], True), kwargs = {})
#   return %buf55,%mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_12 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15360, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 9
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = 9.0
    tmp23 = (tmp21 / tmp22)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (1, 3, 96, 96), (27648, 9216, 96, 1))
        assert_size_stride(arg2_1, (8, ), (1, ))
        assert_size_stride(arg3_1, (8, ), (1, ))
        assert_size_stride(arg4_1, (8, ), (1, ))
        assert_size_stride(arg5_1, (8, ), (1, ))
        assert_size_stride(arg6_1, (8, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg7_1, (8, ), (1, ))
        assert_size_stride(arg8_1, (8, ), (1, ))
        assert_size_stride(arg9_1, (8, ), (1, ))
        assert_size_stride(arg10_1, (8, ), (1, ))
        assert_size_stride(arg11_1, (16, 8, 1, 1), (8, 1, 1, 1))
        assert_size_stride(arg12_1, (16, ), (1, ))
        assert_size_stride(arg13_1, (16, ), (1, ))
        assert_size_stride(arg14_1, (16, ), (1, ))
        assert_size_stride(arg15_1, (16, ), (1, ))
        assert_size_stride(arg16_1, (16, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg17_1, (16, ), (1, ))
        assert_size_stride(arg18_1, (16, ), (1, ))
        assert_size_stride(arg19_1, (16, ), (1, ))
        assert_size_stride(arg20_1, (16, ), (1, ))
        assert_size_stride(arg21_1, (32, 16, 1, 1), (16, 1, 1, 1))
        assert_size_stride(arg22_1, (32, ), (1, ))
        assert_size_stride(arg23_1, (32, ), (1, ))
        assert_size_stride(arg24_1, (32, ), (1, ))
        assert_size_stride(arg25_1, (32, ), (1, ))
        assert_size_stride(arg26_1, (32, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg27_1, (32, ), (1, ))
        assert_size_stride(arg28_1, (32, ), (1, ))
        assert_size_stride(arg29_1, (32, ), (1, ))
        assert_size_stride(arg30_1, (32, ), (1, ))
        assert_size_stride(arg31_1, (32, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(arg32_1, (32, ), (1, ))
        assert_size_stride(arg33_1, (32, ), (1, ))
        assert_size_stride(arg34_1, (32, ), (1, ))
        assert_size_stride(arg35_1, (32, ), (1, ))
        assert_size_stride(arg36_1, (32, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg37_1, (32, ), (1, ))
        assert_size_stride(arg38_1, (32, ), (1, ))
        assert_size_stride(arg39_1, (32, ), (1, ))
        assert_size_stride(arg40_1, (32, ), (1, ))
        assert_size_stride(arg41_1, (64, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(arg42_1, (64, ), (1, ))
        assert_size_stride(arg43_1, (64, ), (1, ))
        assert_size_stride(arg44_1, (64, ), (1, ))
        assert_size_stride(arg45_1, (64, ), (1, ))
        assert_size_stride(arg46_1, (64, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg47_1, (64, ), (1, ))
        assert_size_stride(arg48_1, (64, ), (1, ))
        assert_size_stride(arg49_1, (64, ), (1, ))
        assert_size_stride(arg50_1, (64, ), (1, ))
        assert_size_stride(arg51_1, (64, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg52_1, (64, ), (1, ))
        assert_size_stride(arg53_1, (64, ), (1, ))
        assert_size_stride(arg54_1, (64, ), (1, ))
        assert_size_stride(arg55_1, (64, ), (1, ))
        assert_size_stride(arg56_1, (64, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg57_1, (64, ), (1, ))
        assert_size_stride(arg58_1, (64, ), (1, ))
        assert_size_stride(arg59_1, (64, ), (1, ))
        assert_size_stride(arg60_1, (64, ), (1, ))
        assert_size_stride(arg61_1, (128, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg62_1, (128, ), (1, ))
        assert_size_stride(arg63_1, (128, ), (1, ))
        assert_size_stride(arg64_1, (128, ), (1, ))
        assert_size_stride(arg65_1, (128, ), (1, ))
        assert_size_stride(arg66_1, (128, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg67_1, (128, ), (1, ))
        assert_size_stride(arg68_1, (128, ), (1, ))
        assert_size_stride(arg69_1, (128, ), (1, ))
        assert_size_stride(arg70_1, (128, ), (1, ))
        assert_size_stride(arg71_1, (128, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg72_1, (128, ), (1, ))
        assert_size_stride(arg73_1, (128, ), (1, ))
        assert_size_stride(arg74_1, (128, ), (1, ))
        assert_size_stride(arg75_1, (128, ), (1, ))
        assert_size_stride(arg76_1, (128, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg77_1, (128, ), (1, ))
        assert_size_stride(arg78_1, (128, ), (1, ))
        assert_size_stride(arg79_1, (128, ), (1, ))
        assert_size_stride(arg80_1, (128, ), (1, ))
        assert_size_stride(arg81_1, (128, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg82_1, (128, ), (1, ))
        assert_size_stride(arg83_1, (128, ), (1, ))
        assert_size_stride(arg84_1, (128, ), (1, ))
        assert_size_stride(arg85_1, (128, ), (1, ))
        assert_size_stride(arg86_1, (128, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg87_1, (128, ), (1, ))
        assert_size_stride(arg88_1, (128, ), (1, ))
        assert_size_stride(arg89_1, (128, ), (1, ))
        assert_size_stride(arg90_1, (128, ), (1, ))
        assert_size_stride(arg91_1, (128, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg92_1, (128, ), (1, ))
        assert_size_stride(arg93_1, (128, ), (1, ))
        assert_size_stride(arg94_1, (128, ), (1, ))
        assert_size_stride(arg95_1, (128, ), (1, ))
        assert_size_stride(arg96_1, (128, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg97_1, (128, ), (1, ))
        assert_size_stride(arg98_1, (128, ), (1, ))
        assert_size_stride(arg99_1, (128, ), (1, ))
        assert_size_stride(arg100_1, (128, ), (1, ))
        assert_size_stride(arg101_1, (128, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg102_1, (128, ), (1, ))
        assert_size_stride(arg103_1, (128, ), (1, ))
        assert_size_stride(arg104_1, (128, ), (1, ))
        assert_size_stride(arg105_1, (128, ), (1, ))
        assert_size_stride(arg106_1, (128, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg107_1, (128, ), (1, ))
        assert_size_stride(arg108_1, (128, ), (1, ))
        assert_size_stride(arg109_1, (128, ), (1, ))
        assert_size_stride(arg110_1, (128, ), (1, ))
        assert_size_stride(arg111_1, (128, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg112_1, (128, ), (1, ))
        assert_size_stride(arg113_1, (128, ), (1, ))
        assert_size_stride(arg114_1, (128, ), (1, ))
        assert_size_stride(arg115_1, (128, ), (1, ))
        assert_size_stride(arg116_1, (128, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg117_1, (128, ), (1, ))
        assert_size_stride(arg118_1, (128, ), (1, ))
        assert_size_stride(arg119_1, (128, ), (1, ))
        assert_size_stride(arg120_1, (128, ), (1, ))
        assert_size_stride(arg121_1, (256, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg122_1, (256, ), (1, ))
        assert_size_stride(arg123_1, (256, ), (1, ))
        assert_size_stride(arg124_1, (256, ), (1, ))
        assert_size_stride(arg125_1, (256, ), (1, ))
        assert_size_stride(arg126_1, (256, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg127_1, (256, ), (1, ))
        assert_size_stride(arg128_1, (256, ), (1, ))
        assert_size_stride(arg129_1, (256, ), (1, ))
        assert_size_stride(arg130_1, (256, ), (1, ))
        assert_size_stride(arg131_1, (256, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg132_1, (256, ), (1, ))
        assert_size_stride(arg133_1, (256, ), (1, ))
        assert_size_stride(arg134_1, (256, ), (1, ))
        assert_size_stride(arg135_1, (256, ), (1, ))
        assert_size_stride(arg136_1, (2, 256), (256, 1))
        assert_size_stride(arg137_1, (2, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 3, 96, 96), (27648, 1, 288, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 3, 9216, stream=stream0)
            del arg1_1
            buf1 = empty_strided_cuda((8, 3, 3, 3), (27, 1, 9, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 24, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 8, 48, 48), (18432, 1, 384, 8), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 18432, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            # Topologically Sorted Source Nodes: [batch_norm, x, conv2d_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:5
            buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
            assert_size_stride(buf4, (1, 8, 48, 48), (18432, 1, 384, 8), 'torch.ops.aten.convolution.default')
            del arg6_1
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_1, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_2:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 18432, stream=stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            # Topologically Sorted Source Nodes: [batch_norm_1, x_1, conv2d_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:7
            buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf6, (1, 16, 48, 48), (36864, 1, 768, 16), 'torch.ops.aten.convolution.default')
            del arg11_1
            del buf5
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_2, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_3:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, 36864, stream=stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            # Topologically Sorted Source Nodes: [batch_norm_2, x_2, conv2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:9
            buf8 = extern_kernels.convolution(buf7, arg16_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
            assert_size_stride(buf8, (1, 16, 24, 24), (9216, 1, 384, 16), 'torch.ops.aten.convolution.default')
            del arg16_1
            del buf7
            buf9 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_3, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_4:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, arg17_1, arg18_1, arg19_1, arg20_1, 9216, stream=stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            # Topologically Sorted Source Nodes: [batch_norm_3, x_3, conv2d_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:11
            buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf10, (1, 32, 24, 24), (18432, 1, 768, 32), 'torch.ops.aten.convolution.default')
            del arg21_1
            del buf9
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_4, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_5:12
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf11, arg22_1, arg23_1, arg24_1, arg25_1, 18432, stream=stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            # Topologically Sorted Source Nodes: [batch_norm_4, x_4, conv2d_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:13
            buf12 = extern_kernels.convolution(buf11, arg26_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
            assert_size_stride(buf12, (1, 32, 24, 24), (18432, 1, 768, 32), 'torch.ops.aten.convolution.default')
            del arg26_1
            del buf11
            buf13 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_5, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_5:14
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf13, arg27_1, arg28_1, arg29_1, arg30_1, 18432, stream=stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [batch_norm_5, x_5, conv2d_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:15
            buf14 = extern_kernels.convolution(buf13, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf14, (1, 32, 24, 24), (18432, 1, 768, 32), 'torch.ops.aten.convolution.default')
            del arg31_1
            del buf13
            buf15 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_6, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_5:16
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf15, arg32_1, arg33_1, arg34_1, arg35_1, 18432, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            # Topologically Sorted Source Nodes: [batch_norm_6, x_6, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:17
            buf16 = extern_kernels.convolution(buf15, arg36_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
            assert_size_stride(buf16, (1, 32, 12, 12), (4608, 1, 384, 32), 'torch.ops.aten.convolution.default')
            del arg36_1
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_7, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_6:18
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf17, arg37_1, arg38_1, arg39_1, arg40_1, 4608, stream=stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            # Topologically Sorted Source Nodes: [batch_norm_7, x_7, conv2d_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:19
            buf18 = extern_kernels.convolution(buf17, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf18, (1, 64, 12, 12), (9216, 1, 768, 64), 'torch.ops.aten.convolution.default')
            del arg41_1
            del buf17
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_8, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_7:20
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf19, arg42_1, arg43_1, arg44_1, arg45_1, 9216, stream=stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            # Topologically Sorted Source Nodes: [batch_norm_8, x_8, conv2d_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:21
            buf20 = extern_kernels.convolution(buf19, arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
            assert_size_stride(buf20, (1, 64, 12, 12), (9216, 1, 768, 64), 'torch.ops.aten.convolution.default')
            del arg46_1
            del buf19
            buf21 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_9, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_7:22
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf21, arg47_1, arg48_1, arg49_1, arg50_1, 9216, stream=stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            # Topologically Sorted Source Nodes: [batch_norm_9, x_9, conv2d_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:23
            buf22 = extern_kernels.convolution(buf21, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf22, (1, 64, 12, 12), (9216, 1, 768, 64), 'torch.ops.aten.convolution.default')
            del arg51_1
            del buf21
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_10, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_7:24
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf23, arg52_1, arg53_1, arg54_1, arg55_1, 9216, stream=stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            # Topologically Sorted Source Nodes: [batch_norm_10, x_10, conv2d_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:25
            buf24 = extern_kernels.convolution(buf23, arg56_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
            assert_size_stride(buf24, (1, 64, 6, 6), (2304, 1, 384, 64), 'torch.ops.aten.convolution.default')
            del arg56_1
            del buf23
            buf25 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_11, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_8:26
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf25, arg57_1, arg58_1, arg59_1, arg60_1, 2304, stream=stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            # Topologically Sorted Source Nodes: [batch_norm_11, x_11, conv2d_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:27
            buf26 = extern_kernels.convolution(buf25, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg61_1
            del buf25
            buf27 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_12, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:28
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf27, arg62_1, arg63_1, arg64_1, arg65_1, 4608, stream=stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            # Topologically Sorted Source Nodes: [batch_norm_12, x_12, conv2d_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:29
            buf28 = extern_kernels.convolution(buf27, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf28, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg66_1
            del buf27
            buf29 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_13, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:30
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf29, arg67_1, arg68_1, arg69_1, arg70_1, 4608, stream=stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            # Topologically Sorted Source Nodes: [batch_norm_13, x_13, conv2d_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:31
            buf30 = extern_kernels.convolution(buf29, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf30, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg71_1
            del buf29
            buf31 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_14, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:32
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf31, arg72_1, arg73_1, arg74_1, arg75_1, 4608, stream=stream0)
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            # Topologically Sorted Source Nodes: [batch_norm_14, x_14, conv2d_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:33
            buf32 = extern_kernels.convolution(buf31, arg76_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf32, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg76_1
            del buf31
            buf33 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_15, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:34
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf33, arg77_1, arg78_1, arg79_1, arg80_1, 4608, stream=stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            # Topologically Sorted Source Nodes: [batch_norm_15, x_15, conv2d_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:35
            buf34 = extern_kernels.convolution(buf33, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf34, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg81_1
            del buf33
            buf35 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_16, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:36
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf35, arg82_1, arg83_1, arg84_1, arg85_1, 4608, stream=stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            # Topologically Sorted Source Nodes: [batch_norm_16, x_16, conv2d_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:37
            buf36 = extern_kernels.convolution(buf35, arg86_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf36, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg86_1
            del buf35
            buf37 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_17, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:38
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf37, arg87_1, arg88_1, arg89_1, arg90_1, 4608, stream=stream0)
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            # Topologically Sorted Source Nodes: [batch_norm_17, x_17, conv2d_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:39
            buf38 = extern_kernels.convolution(buf37, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf38, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg91_1
            del buf37
            buf39 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_18, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:40
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf39, arg92_1, arg93_1, arg94_1, arg95_1, 4608, stream=stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            # Topologically Sorted Source Nodes: [batch_norm_18, x_18, conv2d_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:41
            buf40 = extern_kernels.convolution(buf39, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf40, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg96_1
            del buf39
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_19, x_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:42
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf41, arg97_1, arg98_1, arg99_1, arg100_1, 4608, stream=stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            # Topologically Sorted Source Nodes: [batch_norm_19, x_19, conv2d_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:43
            buf42 = extern_kernels.convolution(buf41, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf42, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg101_1
            del buf41
            buf43 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_20, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:44
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf43, arg102_1, arg103_1, arg104_1, arg105_1, 4608, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            del arg105_1
            # Topologically Sorted Source Nodes: [batch_norm_20, x_20, conv2d_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:45
            buf44 = extern_kernels.convolution(buf43, arg106_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf44, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg106_1
            del buf43
            buf45 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_21, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:46
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf45, arg107_1, arg108_1, arg109_1, arg110_1, 4608, stream=stream0)
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [batch_norm_21, x_21, conv2d_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:47
            buf46 = extern_kernels.convolution(buf45, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf46, (1, 128, 6, 6), (4608, 1, 768, 128), 'torch.ops.aten.convolution.default')
            del arg111_1
            del buf45
            buf47 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_22, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:48
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf47, arg112_1, arg113_1, arg114_1, arg115_1, 4608, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            # Topologically Sorted Source Nodes: [batch_norm_22, x_22, conv2d_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:49
            buf48 = extern_kernels.convolution(buf47, arg116_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf48, (1, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution.default')
            del arg116_1
            del buf47
            buf49 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_23, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_10:50
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf49, arg117_1, arg118_1, arg119_1, arg120_1, 1152, stream=stream0)
            del arg117_1
            del arg118_1
            del arg119_1
            del arg120_1
            # Topologically Sorted Source Nodes: [batch_norm_23, x_23, conv2d_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:51
            buf50 = extern_kernels.convolution(buf49, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf50, (1, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution.default')
            del arg121_1
            del buf49
            buf51 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_24, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_11:52
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf51, arg122_1, arg123_1, arg124_1, arg125_1, 2304, stream=stream0)
            del arg122_1
            del arg123_1
            del arg124_1
            del arg125_1
            # Topologically Sorted Source Nodes: [batch_norm_24, x_24, conv2d_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:53
            buf52 = extern_kernels.convolution(buf51, arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
            assert_size_stride(buf52, (1, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution.default')
            del arg126_1
            del buf51
            buf53 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_25, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_11:54
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf53, arg127_1, arg128_1, arg129_1, arg130_1, 2304, stream=stream0)
            del arg127_1
            del arg128_1
            del arg129_1
            del arg130_1
            # Topologically Sorted Source Nodes: [batch_norm_25, x_25, conv2d_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:55
            buf54 = extern_kernels.convolution(buf53, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf54, (1, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution.default')
            del arg131_1
            del buf53
            buf55 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            buf56 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_26, x_26, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_no_training_mean_relu_12:56
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training_mean_relu_12.run(buf56, buf54, arg132_1, arg133_1, arg134_1, arg135_1, 256, 9, stream=stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            del arg135_1
            del buf54
            buf57 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm_26, x_26, x_27, x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:57
            extern_kernels.addmm(arg137_1, reinterpret_tensor(buf56, (1, 256), (0, 1), 0), reinterpret_tensor(arg136_1, (256, 2), (1, 256), 0), alpha=1, beta=1, out=buf57)
            del arg136_1
            del arg137_1
            del buf56
        return (buf57, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 3, 96, 96), (27648, 9216, 96, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
