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


# kernel path: /tmp/torchinductor_emre/e2/ce2qv3kqynayct2dwhs3jztuwbhl7vjurhsg7gu3tpca6oefgorp.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 3456, 'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_emre/rg/crgy52sy6fj6gu4xf5n64hpdcvb5ximq4fklrrzbmhlwuquvlhur.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[1, 3, 32, 32][3072, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 9216, 'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/qa/cqajtdpkrxejeomqdthfwsvgpy4l6e3bzblxl45i4dud5n5wuikc.py
# Topologically Sorted Source Nodes: [batch_norm, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   out => relu
# Graph fragment:
#   %convolution : Tensor "f32[1, 16, 32, 32][16384, 1, 512, 16]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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


# kernel path: /tmp/torchinductor_emre/3j/c3jwyx4lxeucl2yq5srypqujthfmqpkpdiqmxd42bbfgpezdtnra.py
# Topologically Sorted Source Nodes: [batch_norm_1, out_1, conv2d_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   conv2d_2 => convolution_2
#   out_1 => relu_1
# Graph fragment:
#   %arg11_1 : Tensor "f32[16, 16, 3, 3][144, 9, 3, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %unsqueeze_8 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 18432, 'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/7x/c7xwyzyy5w7lpgw2e2tveudg2z6dmsxehotxfaujwazskebazild.py
# Topologically Sorted Source Nodes: [out_2, out_3, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_2 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   out_3 => add_6
#   out_4 => relu_2
# Graph fragment:
#   %convolution_2 : Tensor "f32[1, 16, 32, 32][16384, 1, 512, 16]cuda:0" = PlaceHolder[target=convolution_2]
#   %arg12_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %relu : Tensor "f32[1, 16, 32, 32][16384, 1, 512, 16]cuda:0" = PlaceHolder[target=relu]
#   %unsqueeze_16 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %add_6 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %relu), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 32, 32][16384, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
#   return %relu_2
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp16 = tl.load(in_ptr4 + (x2), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ku/ckuso7ijbcp4weqpqpz3uscrfvpwpnkof2beqxr5cmwjdc3k7ksc.py
# Topologically Sorted Source Nodes: [batch_norm_3, out_5, conv2d_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, add_8, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   conv2d_4 => convolution_4
#   out_5 => relu_3
# Graph fragment:
#   %arg21_1 : Tensor "f32[32, 32, 3, 3][288, 9, 3, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %unsqueeze_24 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_7 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_7,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_8 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 73728, 'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ol/colgrzmbbrh4mert2stfjdupylxl3i5ubzsfgapf45zzzewdx7ky.py
# Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
# Graph fragment:
#   %arg16_1 : Tensor "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convolution_3 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf10
triton_poi_fused_convolution_6 = async_compile.triton('triton_poi_fused_convolution_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 36864, 'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/a7/ca752j5o35ag7feqb2lq5a7gekcw43g6uxfexifwoyophpz7ihs4.py
# Topologically Sorted Source Nodes: [batch_norm_3, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, add_8, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   out_5 => relu_3
# Graph fragment:
#   %convolution_3 : Tensor "f32[1, 32, 16, 16][8192, 1, 512, 32]cuda:0" = PlaceHolder[target=convolution_3]
#   %arg17_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %unsqueeze_24 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_7 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_7,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_8 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
#   return %relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
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


# kernel path: /tmp/torchinductor_emre/uf/cufvqgu6vh33lmpomy2nzpguuigx6v3mg26cvkdmwowcdc3r455n.py
# Topologically Sorted Source Nodes: [out_6, input_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_11, add_12, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   out_6 => add_10, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   out_7 => add_13
#   out_8 => relu_4
# Graph fragment:
#   %convolution_4 : Tensor "f32[1, 32, 16, 16][8192, 1, 512, 32]cuda:0" = PlaceHolder[target=convolution_4]
#   %arg22_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %convolution_5 : Tensor "f32[1, 32, 16, 16][8192, 1, 512, 32]cuda:0" = PlaceHolder[target=convolution_5]
#   %arg27_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %arg29_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %add_13 : Tensor "f32[1, 32, 16, 16][8192, 1, 512, 32]cuda:0" = PlaceHolder[target=add_13]
#   %unsqueeze_32 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_9 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_10 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg27_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_5 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %add_11 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_11,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg29_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg30_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_12 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %add_13 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %add_12), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   return %add_13,%relu_4
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 132096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/cv/ccv6n67cehy55jnap7fvk45at2447ayitxbrbo6mlcilkyjxryw4.py
# Topologically Sorted Source Nodes: [out_8, conv2d_6], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_6 => convolution_6
#   out_8 => relu_4
# Graph fragment:
#   %arg31_1 : Tensor "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = PlaceHolder[target=arg31_1]
#   %relu_4 : Tensor "f32[1, 32, 16, 16][8192, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg31_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf18
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_poi_fused_convolution_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 147456, 'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/or/cork3y7w4o536ugss2qdhktt3uxfowwowfn5cov2rydhirs7lphm.py
# Topologically Sorted Source Nodes: [batch_norm_6, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_6 => add_14, add_15, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   out_9 => relu_5
# Graph fragment:
#   %convolution_6 : Tensor "f32[1, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=convolution_6]
#   %arg32_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %arg33_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg33_1]
#   %arg34_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %unsqueeze_48 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_14 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg33_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg34_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_15 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_5 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   return %relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50176}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
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


# kernel path: /tmp/torchinductor_emre/w7/cw7axxr3g7qrrq4i3fvth347s3fjp33bixxvfgi4pbnt77frzpat.py
# Topologically Sorted Source Nodes: [batch_norm_6, out_9, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_6 => add_14, add_15, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   conv2d_7 => convolution_7
#   out_9 => relu_5
# Graph fragment:
#   %arg36_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg36_1]
#   %unsqueeze_48 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_14 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg33_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg34_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_15 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_5 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf21
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 294912, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ra/crazqqcgg6zq55k23pw4zvnq2vi5ygorifnole3km4xd3wdhccqz.py
# Topologically Sorted Source Nodes: [out_10, input_4, out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   input_4 => add_18, add_19, mul_24, mul_25, mul_26, reciprocal_8, sqrt_8, sub_8, unsqueeze_64, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, unsqueeze_71
#   out_10 => add_16, add_17, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   out_11 => add_20
#   out_12 => relu_6
#   out_13 => mean
# Graph fragment:
#   %convolution_7 : Tensor "f32[1, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=convolution_7]
#   %arg37_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %convolution_8 : Tensor "f32[1, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=convolution_8]
#   %arg42_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %arg43_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %arg44_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %add_20 : Tensor "f32[1, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=add_20]
#   %buf25 : Tensor "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0" = PlaceHolder[target=buf25]
#   %unsqueeze_56 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_16 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_17 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %sub_8 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %add_18 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg43_1, 1e-05), kwargs = {})
#   %sqrt_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_18,), kwargs = {})
#   %reciprocal_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_8,), kwargs = {})
#   %mul_24 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_24, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %mul_25 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg44_1, -1), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, -1), kwargs = {})
#   %mul_26 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg45_1, -1), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, -1), kwargs = {})
#   %add_19 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_20 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %add_19), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_20,), kwargs = {})
#   %mean : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_6, [-1, -2], True), kwargs = {})
#   return %add_20,%buf25,%mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_12 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 35328, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + 64*r0_1), xmask, other=0.0)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = 64.0
    tmp37 = (tmp35 / tmp36)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp37, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (1, 3, 32, 32), (3072, 1024, 32, 1))
        assert_size_stride(arg2_1, (16, ), (1, ))
        assert_size_stride(arg3_1, (16, ), (1, ))
        assert_size_stride(arg4_1, (16, ), (1, ))
        assert_size_stride(arg5_1, (16, ), (1, ))
        assert_size_stride(arg6_1, (16, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg7_1, (16, ), (1, ))
        assert_size_stride(arg8_1, (16, ), (1, ))
        assert_size_stride(arg9_1, (16, ), (1, ))
        assert_size_stride(arg10_1, (16, ), (1, ))
        assert_size_stride(arg11_1, (16, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg12_1, (16, ), (1, ))
        assert_size_stride(arg13_1, (16, ), (1, ))
        assert_size_stride(arg14_1, (16, ), (1, ))
        assert_size_stride(arg15_1, (16, ), (1, ))
        assert_size_stride(arg16_1, (32, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg17_1, (32, ), (1, ))
        assert_size_stride(arg18_1, (32, ), (1, ))
        assert_size_stride(arg19_1, (32, ), (1, ))
        assert_size_stride(arg20_1, (32, ), (1, ))
        assert_size_stride(arg21_1, (32, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg22_1, (32, ), (1, ))
        assert_size_stride(arg23_1, (32, ), (1, ))
        assert_size_stride(arg24_1, (32, ), (1, ))
        assert_size_stride(arg25_1, (32, ), (1, ))
        assert_size_stride(arg26_1, (32, 16, 1, 1), (16, 1, 1, 1))
        assert_size_stride(arg27_1, (32, ), (1, ))
        assert_size_stride(arg28_1, (32, ), (1, ))
        assert_size_stride(arg29_1, (32, ), (1, ))
        assert_size_stride(arg30_1, (32, ), (1, ))
        assert_size_stride(arg31_1, (64, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg32_1, (64, ), (1, ))
        assert_size_stride(arg33_1, (64, ), (1, ))
        assert_size_stride(arg34_1, (64, ), (1, ))
        assert_size_stride(arg35_1, (64, ), (1, ))
        assert_size_stride(arg36_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg37_1, (64, ), (1, ))
        assert_size_stride(arg38_1, (64, ), (1, ))
        assert_size_stride(arg39_1, (64, ), (1, ))
        assert_size_stride(arg40_1, (64, ), (1, ))
        assert_size_stride(arg41_1, (64, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(arg42_1, (64, ), (1, ))
        assert_size_stride(arg43_1, (64, ), (1, ))
        assert_size_stride(arg44_1, (64, ), (1, ))
        assert_size_stride(arg45_1, (64, ), (1, ))
        assert_size_stride(arg46_1, (10, 64), (64, 1))
        assert_size_stride(arg47_1, (10, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg0_1, buf1, 48, 9, stream=stream0)
            del arg0_1
            buf0 = empty_strided_cuda((1, 3, 32, 32), (3072, 1, 96, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg1_1, buf0, 3, 1024, stream=stream0)
            del arg1_1
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 16, 32, 32), (16384, 1, 512, 16), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [batch_norm, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 16384, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf7 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm_1, out_1, conv2d_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3:5
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg11_1, buf7, 256, 9, stream=stream0)
            del arg11_1
            buf4 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg6_1, buf4, 256, 9, stream=stream0)
            del arg6_1
            # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:7
            buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf5, (1, 16, 32, 32), (16384, 1, 512, 16), 'torch.ops.aten.convolution.default')
            del buf4
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_2:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 16384, stream=stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            # Topologically Sorted Source Nodes: [batch_norm_1, out_1, conv2d_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:9
            buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf8, (1, 16, 32, 32), (16384, 1, 512, 16), 'torch.ops.aten.convolution.default')
            del buf6
            del buf7
            buf9 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [out_2, out_3, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf9, arg12_1, arg13_1, arg14_1, arg15_1, buf3, 16384, stream=stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            del buf3
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:11
            buf15 = extern_kernels.convolution(buf9, arg26_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf15, (1, 32, 16, 16), (8192, 1, 512, 32), 'torch.ops.aten.convolution.default')
            del arg26_1
            buf13 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm_3, out_5, conv2d_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5:12
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg21_1, buf13, 1024, 9, stream=stream0)
            del arg21_1
            buf10 = empty_strided_cuda((32, 16, 3, 3), (144, 1, 48, 16), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_6:13
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_6.run(arg16_1, buf10, 512, 9, stream=stream0)
            del arg16_1
            # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:14
            buf11 = extern_kernels.convolution(buf9, buf10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf11, (1, 32, 16, 16), (8192, 1, 512, 32), 'torch.ops.aten.convolution.default')
            del buf10
            del buf9
            buf12 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_3, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_7:15
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf12, arg17_1, arg18_1, arg19_1, arg20_1, 8192, stream=stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            # Topologically Sorted Source Nodes: [batch_norm_3, out_5, conv2d_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:16
            buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf14, (1, 32, 16, 16), (8192, 1, 512, 32), 'torch.ops.aten.convolution.default')
            del buf12
            del buf13
            buf16 = buf14; del buf14  # reuse
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [out_6, input_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8:17
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf17, arg22_1, arg23_1, arg24_1, arg25_1, buf15, arg27_1, arg28_1, arg29_1, arg30_1, 8192, stream=stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            del buf15
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:18
            buf23 = extern_kernels.convolution(buf17, arg41_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf23, (1, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            del arg41_1
            buf18 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
            # Topologically Sorted Source Nodes: [out_8, conv2d_6], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_9:19
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_9.run(arg31_1, buf18, 2048, 9, stream=stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [out_8, conv2d_6], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:20
            buf19 = extern_kernels.convolution(buf17, buf18, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf19, (1, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            del buf17
            del buf18
            buf20 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_6, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_10:21
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf20, arg32_1, arg33_1, arg34_1, arg35_1, 4096, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            buf21 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm_6, out_9, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11:22
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(arg36_1, buf21, 4096, 9, stream=stream0)
            del arg36_1
            # Topologically Sorted Source Nodes: [batch_norm_6, out_9, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:23
            buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf22, (1, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            del buf20
            del buf21
            buf24 = buf22; del buf22  # reuse
            buf25 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
            buf26 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [out_10, input_4, out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_12:24
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_12.run(buf24, buf26, arg37_1, arg38_1, arg39_1, arg40_1, buf23, arg42_1, arg43_1, arg44_1, arg45_1, 64, 64, stream=stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            del buf23
            del buf24
            buf27 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out_12, out_13, out_14, out_15], Original ATen: [aten.relu, aten.mean, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:25
            extern_kernels.addmm(arg47_1, reinterpret_tensor(buf26, (1, 64), (0, 1), 0), reinterpret_tensor(arg46_1, (64, 10), (1, 64), 0), alpha=1, beta=1, out=buf27)
            del arg46_1
            del arg47_1
            del buf26
        return (buf27, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((10, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
