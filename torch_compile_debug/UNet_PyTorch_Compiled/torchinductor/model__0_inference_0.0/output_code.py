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


# kernel path: /tmp/torchinductor_emre/bb/cbbqm33gdp6fihdoavl2yldip7oauj4b4t664dmg3o2335grvl5e.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[1, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 589824, 'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 65536*y0), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/6v/c6vh6v7rhpvc3zoklwxrmnryeiftvfeswi6jktds3zsighympeif.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[8, 3, 3, 3][27, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_emre/tp/ctp2kw2xrwtcqq2cdwfv5agx2krfv6x6lhfgrsd4h2jm3647xmif.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %buf2 : Tensor "f32[1, 8, 256, 256][524288, 1, 2048, 8]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291488}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/x5/cx5r3665d5e7udbwoafsuprsw4sit6xivdu2pcvojsjxecabmtaq.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[8, 8, 3, 3][72, 9, 3, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf4
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_poi_fused_convolution_relu_3', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4608, 'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 72*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/h5/ch5gkz63zu4jthnzsbxlvkb676zbilfi6njrby4z3xtnnbttba57.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
# Graph fragment:
#   %buf5 : Tensor "f32[1, 8, 256, 256][524288, 1, 2048, 8]cuda:0" = PlaceHolder[target=buf5]
#   %arg4_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_convolution_relu_4 = async_compile.triton('triton_poi_fused_convolution_relu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 65536}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2097184, 'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8*x1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x1 + 65536*y0), tmp4, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/bz/cbzgujiodzy74tfto4ln5aylaxtvzfpbrhgfznc2fvk6xockdu6o.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   x => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu_1 : Tensor "f32[1, 8, 256, 256][1572864, 65536, 256, 1]cuda:0" = PlaceHolder[target=relu_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048576, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = (xindex % 128)
    x2 = xindex // 128
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1 + 512*x2 + 65536*y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 512*x2 + 65536*y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + 2*x1 + 512*x2 + 65536*y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (257 + 2*x1 + 512*x2 + 65536*y0), ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 8*x3), tmp6, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/uh/cuhxthnulfy7ce37gyh7glomu247nx3rvox2h7upw6l7qhjswe26.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   x => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %arg5_1 : Tensor "f32[16, 8, 3, 3][72, 9, 3, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf8
triton_poi_fused_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 9216, 'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 72*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ee/ceesoyvhrwbuwovtmfvvkkg5julmn3ed4ayqdt4oso2xfzasjyz5.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   x => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %buf9 : Tensor "f32[1, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = PlaceHolder[target=buf9]
#   %arg6_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   return %relu_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3145792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/w7/cw7mulkdxhzu4bx6wwt4grbgbicso2z2vpo5lkr4ocid5b3gq7o3.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   x => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %arg7_1 : Tensor "f32[16, 16, 3, 3][144, 9, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf11
triton_poi_fused_convolution_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 18432, 'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/i5/ci52rtkuujuvhd3ogk5qvsk5avnshe4x4wkg3dpchqfxioh5hril.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   x => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %buf12 : Tensor "f32[1, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = PlaceHolder[target=buf12]
#   %arg8_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   return %relu_3
triton_poi_fused_convolution_max_pool2d_with_indices_relu_9 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048640, 'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_9(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp4, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/dj/cdjjrndiei6t2hbfyfajdvbaor4z6wsjkvj6vuhm2jqhsvyn7ts2.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %relu_3 : Tensor "f32[1, 16, 128, 128][786432, 16384, 128, 1]cuda:0" = PlaceHolder[target=relu_3]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 524288, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = (xindex % 64)
    x2 = xindex // 64
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1 + 256*x2 + 16384*y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 256*x2 + 16384*y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x1 + 256*x2 + 16384*y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x1 + 256*x2 + 16384*y0), ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 16*x3), tmp6, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ie/ciegg67zo63ik67mfu2vz5wdrq4emj3oml4ccjnvnaanhumkvjck.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %arg9_1 : Tensor "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf15
triton_poi_fused_convolution_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 36864, 'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/24/c24ad4igk3vnswpkecasbbvf4eejzh7qgzgmvoblqxta4rcew4ks.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %buf16 : Tensor "f32[1, 32, 64, 64][131072, 1, 2048, 32]cuda:0" = PlaceHolder[target=buf16]
#   %arg10_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   return %relu_4
triton_poi_fused_convolution_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1572992}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/az/cazkikg2asnw64lwmetkh6bgwiqh5jzxlkkqi6fmbgoir4wjtery.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %arg11_1 : Tensor "f32[32, 32, 3, 3][288, 9, 3, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf18
triton_poi_fused_convolution_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 73728, 'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/7q/c7qihzwpqbetdjd7kkru4d4jblbieaxgsk6ynwbukkrwpg6k36bl.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %buf19 : Tensor "f32[1, 32, 64, 64][131072, 1, 2048, 32]cuda:0" = PlaceHolder[target=buf19]
#   %arg12_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   return %relu_5
triton_poi_fused_convolution_max_pool2d_with_indices_relu_14 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 524416, 'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x1 + 4096*y0), tmp4, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/6o/c6oucyxg3h3vlsusgwqf5cmcsapb26lvvhkuz3kb5dizz74sjqym.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %relu_5 : Tensor "f32[1, 32, 64, 64][393216, 4096, 64, 1]cuda:0" = PlaceHolder[target=relu_5]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_4
triton_poi_fused_convolution_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 262144, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 32)
    x2 = xindex // 32
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1 + 128*x2 + 4096*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 128*x2 + 4096*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x1 + 128*x2 + 4096*y0), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x1 + 128*x2 + 4096*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 32*x3), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/zq/czq3vj75eezatoz2iggqduomkxoqjpg5rj4vkkzde326gau63kya.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %arg13_1 : Tensor "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf22
triton_poi_fused_convolution_max_pool2d_with_indices_relu_16 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 147456, 'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/gf/cgfveo7gvwe2ddqr43qn5xerssbvteg3ndlzdvp3bow4xr5tvuwh.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %buf23 : Tensor "f32[1, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=buf23]
#   %arg14_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   return %relu_6
triton_poi_fused_convolution_max_pool2d_with_indices_relu_17 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ds/cdstmdthtwyvx6arei7uf6p62mjvsnjzpg3qxglgmsmhx5glfqxb.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %arg15_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg15_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf25
triton_poi_fused_convolution_max_pool2d_with_indices_relu_18 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_18', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 294912, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/zc/czclwlty47sevous4znn4sqkp2s4rtgvkcxmgtjkn7ih6yny7u3n.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_16 => relu_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %buf26 : Tensor "f32[1, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=buf26]
#   %arg16_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   return %relu_7
triton_poi_fused_convolution_max_pool2d_with_indices_relu_19 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 262400, 'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_19(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x1 + 1024*y0), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/wm/cwmssx5attz2xvc36veajkyswomuvxcnmqxt7m23f7jzyqrv73vx.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_16 => relu_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
#   x_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %relu_7 : Tensor "f32[1, 64, 32, 32][196608, 1024, 32, 1]cuda:0" = PlaceHolder[target=relu_7]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_6
triton_poi_fused_convolution_max_pool2d_with_indices_relu_20 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 131072, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 16)
    x2 = xindex // 16
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1 + 64*x2 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 64*x2 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x1 + 64*x2 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x1 + 64*x2 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 64*x3), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/v5/cv5fqx5l2b2ood7vw4kpqjtz2yohfvvystwrhicofq2sy5k6cbbz.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_16 => relu_7
#   input_17 => convolution_8
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
#   x_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %arg17_1 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf29
triton_poi_fused_convolution_max_pool2d_with_indices_relu_21 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 589824, 'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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


# kernel path: /tmp/torchinductor_emre/7r/c7rqtymwj23oyohzpsff5t3qrnqukk3k2du5ys65loiv5meno6nr.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_16 => relu_7
#   input_17 => convolution_8
#   input_18 => relu_8
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
#   x_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %buf30 : Tensor "f32[1, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=buf30]
#   %arg18_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   return %relu_8
triton_poi_fused_convolution_max_pool2d_with_indices_relu_22 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 393728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/nl/cnlyfwuqmcs24zncevduztiuwo5ofa23jdbtvmmejzyzivwv3o7j.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_16 => relu_7
#   input_17 => convolution_8
#   input_18 => relu_8
#   input_19 => convolution_9
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
#   x_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %arg19_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf32
triton_poi_fused_convolution_max_pool2d_with_indices_relu_23 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/6d/c6d424v6ybfjrewla7qhrpxw7f73mx3sq7yhpfvedovcw3ujklop.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18, input_19, input_20, x_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => relu_4
#   input_11 => convolution_5
#   input_12 => relu_5
#   input_13 => convolution_6
#   input_14 => relu_6
#   input_15 => convolution_7
#   input_16 => relu_7
#   input_17 => convolution_8
#   input_18 => relu_8
#   input_19 => convolution_9
#   input_2 => relu
#   input_20 => relu_9
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => convolution_2
#   input_6 => relu_2
#   input_7 => convolution_3
#   input_8 => relu_3
#   input_9 => convolution_4
#   x => _low_memory_max_pool_with_offsets
#   x_1 => _low_memory_max_pool_with_offsets_1
#   x_2 => _low_memory_max_pool_with_offsets_2
#   x_3 => _low_memory_max_pool_with_offsets_3
#   x_4 => _unsafe_index, add, add_1, add_2, add_3, convert_element_type, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota, iota_1, mul, mul_1, mul_2, mul_3, unsqueeze
# Graph fragment:
#   %buf33 : Tensor "f32[1, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=buf33]
#   %arg20_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %convolution : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : Tensor "f32[1, 128, 16, 16][32768, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %iota : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0), kwargs = {})
#   %convert_element_type : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %add_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %convert_element_type_1 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.int64), kwargs = {})
#   %unsqueeze : Tensor "i64[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, -1), kwargs = {})
#   %iota_1 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_2 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_2 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %add_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.0), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.5), kwargs = {})
#   %convert_element_type_3 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %unsqueeze, %convert_element_type_3]), kwargs = {})
#   return %_unsafe_index
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_max_pool2d_with_indices_mul_relu_unsqueeze_24 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_max_pool2d_with_indices_mul_relu_unsqueeze_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_max_pool2d_with_indices_mul_relu_unsqueeze_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_max_pool2d_with_indices_mul_relu_unsqueeze_24(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + 128*tmp8 + 2048*tmp4), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/47/c472muqtsc62nd5ldmwxjxquafjnbsrzkqt5neiz3u5rmymejmqh.py
# Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_21 => convolution_10
# Graph fragment:
#   %cat : Tensor "f32[1, 192, 32, 32][196608, 1024, 32, 1]cuda:0" = PlaceHolder[target=cat]
#   %convolution_10 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf36
triton_poi_fused_convolution_25 = async_compile.triton('triton_poi_fused_convolution_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1572864, 'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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
    tl.store(out_ptr0 + (y0 + 192*x1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/lk/clkvzozs7tgzc7fyvdyicdjof4afz4ccsvsum2eeyl2hftuwgyjv.py
# Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_21 => convolution_10
# Graph fragment:
#   %arg21_1 : Tensor "f32[64, 192, 3, 3][1728, 9, 3, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %convolution_10 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf37
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 884736, 'x': 442368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/mj/cmjqgmnpt6iqxks6fryz45miznsfcmz4xhb5fktvqlx3jmvs6jcy.py
# Topologically Sorted Source Nodes: [input_21, input_22, input_23, input_24, x_6], Original ATen: [aten.convolution, aten.relu, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_21 => convolution_10
#   input_22 => relu_10
#   input_23 => convolution_11
#   input_24 => relu_11
#   x_6 => _unsafe_index_1, add_4, add_5, add_6, add_7, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, iota_2, iota_3, mul_4, mul_5, mul_6, mul_7, unsqueeze_1
# Graph fragment:
#   %buf41 : Tensor "f32[1, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=buf41]
#   %arg24_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %convolution_10 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : Tensor "f32[1, 64, 32, 32][65536, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %iota_2 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_4 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_4 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %add_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %mul_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.5), kwargs = {})
#   %convert_element_type_5 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.int64), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_5, -1), kwargs = {})
#   %iota_3 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_6 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_6 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, 0), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float32), kwargs = {})
#   %add_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 0.0), kwargs = {})
#   %mul_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.5), kwargs = {})
#   %convert_element_type_7 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[1, 64, 64, 64][262144, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_11, [None, None, %unsqueeze_1, %convert_element_type_7]), kwargs = {})
#   return %_unsafe_index_1
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_27 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + 64*tmp8 + 2048*tmp4), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/76/c76l6axsgyasx5ssjturszgs7b2ucihqowssjkvdueow57ujglgx.py
# Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_25 => convolution_12
# Graph fragment:
#   %cat_1 : Tensor "f32[1, 96, 64, 64][393216, 4096, 64, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %convolution_12 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf44
triton_poi_fused_convolution_28 = async_compile.triton('triton_poi_fused_convolution_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 3145728, 'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 4096*y0), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 96*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/tn/ctnxusscxrad7iufb6jwb6znvquyqqxegoulw3yf3homvrrhzwpf.py
# Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_25 => convolution_12
# Graph fragment:
#   %arg25_1 : Tensor "f32[32, 96, 3, 3][864, 9, 3, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %convolution_12 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf45
triton_poi_fused_convolution_29 = async_compile.triton('triton_poi_fused_convolution_29', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 221184, 'x': 110592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ij/cijlkm4txkqnfbiw2syjbwjwtzc752j5xajab5ig32kwwinc2n5z.py
# Topologically Sorted Source Nodes: [input_25, input_26, input_27, input_28, x_8], Original ATen: [aten.convolution, aten.relu, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_25 => convolution_12
#   input_26 => relu_12
#   input_27 => convolution_13
#   input_28 => relu_13
#   x_8 => _unsafe_index_2, add_10, add_11, add_8, add_9, convert_element_type_10, convert_element_type_11, convert_element_type_8, convert_element_type_9, iota_4, iota_5, mul_10, mul_11, mul_8, mul_9, unsqueeze_2
# Graph fragment:
#   %buf49 : Tensor "f32[1, 32, 64, 64][131072, 1, 2048, 32]cuda:0" = PlaceHolder[target=buf49]
#   %arg28_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %convolution_12 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg27_1, %arg28_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : Tensor "f32[1, 32, 64, 64][131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %iota_4 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_8 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_8 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 0), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %add_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_8, 0.0), kwargs = {})
#   %mul_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.5), kwargs = {})
#   %convert_element_type_9 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.int64), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_9, -1), kwargs = {})
#   %iota_5 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_10 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_5, 1), kwargs = {})
#   %add_10 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, 0), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %add_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_10, 0.0), kwargs = {})
#   %mul_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 0.5), kwargs = {})
#   %convert_element_type_11 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.int64), kwargs = {})
#   %_unsafe_index_2 : Tensor "f32[1, 32, 128, 128][524288, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_13, [None, None, %unsqueeze_2, %convert_element_type_11]), kwargs = {})
#   return %_unsafe_index_2
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_30 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 128) % 128)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x4 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + 32*tmp8 + 2048*tmp4), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/v7/cv7z3kjea47lcvo5fkh3keitmbyvcjad4ofhikv3wyscp7nj5fs3.py
# Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_29 => convolution_14
# Graph fragment:
#   %cat_2 : Tensor "f32[1, 48, 128, 128][786432, 16384, 128, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %convolution_14 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg29_1, %arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf52
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 6291456, 'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 16384*y0), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/og/cogvahpmdwmoucypm67yvoljpnltjvhk2ntol6b65tyognne2udu.py
# Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_29 => convolution_14
# Graph fragment:
#   %arg29_1 : Tensor "f32[16, 48, 3, 3][432, 9, 3, 1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convolution_14 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg29_1, %arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf53
triton_poi_fused_convolution_32 = async_compile.triton('triton_poi_fused_convolution_32', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 55296, 'x': 27648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 432*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/qx/cqxq4wixghh2pzjinnmiqyjhaa6hceuyvcy4u5kemtb7bo62njcc.py
# Topologically Sorted Source Nodes: [input_29, input_30, input_31, input_32, x_10], Original ATen: [aten.convolution, aten.relu, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_29 => convolution_14
#   input_30 => relu_14
#   input_31 => convolution_15
#   input_32 => relu_15
#   x_10 => _unsafe_index_3, add_12, add_13, add_14, add_15, convert_element_type_12, convert_element_type_13, convert_element_type_14, convert_element_type_15, iota_6, iota_7, mul_12, mul_13, mul_14, mul_15, unsqueeze_3
# Graph fragment:
#   %buf57 : Tensor "f32[1, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = PlaceHolder[target=buf57]
#   %arg32_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %convolution_14 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg29_1, %arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg31_1, %arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : Tensor "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %iota_6 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_12 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_12 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, 0), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %add_13 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_12, 0.0), kwargs = {})
#   %mul_13 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, 0.5), kwargs = {})
#   %convert_element_type_13 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.int64), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_13, -1), kwargs = {})
#   %iota_7 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_14 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_7, 1), kwargs = {})
#   %add_14 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, 0), kwargs = {})
#   %convert_element_type_14 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_14, torch.float32), kwargs = {})
#   %add_15 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_14, 0.0), kwargs = {})
#   %mul_15 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, 0.5), kwargs = {})
#   %convert_element_type_15 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.int64), kwargs = {})
#   %_unsafe_index_3 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_15, [None, None, %unsqueeze_3, %convert_element_type_15]), kwargs = {})
#   return %_unsafe_index_3
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_33 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_33(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x2 = xindex // 65536
    x4 = xindex
    tmp10 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x2 + 16*tmp8 + 2048*tmp4), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/rc/crcbikrmxwmu2l5jbfx2wqsfqnd2auu7vq7z3fq35o2pmogpybgx.py
# Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_33 => convolution_16
# Graph fragment:
#   %cat_3 : Tensor "f32[1, 24, 256, 256][1572864, 65536, 256, 1]cuda:0" = PlaceHolder[target=cat_3]
#   %convolution_16 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf60
triton_poi_fused_convolution_34 = async_compile.triton('triton_poi_fused_convolution_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 12582912, 'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 65536*y0), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/jh/cjhnuy5nhkmvp4dlltypybxixcivopa5dfs56is3mnaraq6clvrd.py
# Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_33 => convolution_16
# Graph fragment:
#   %arg33_1 : Tensor "f32[8, 24, 3, 3][216, 9, 3, 1]cuda:0" = PlaceHolder[target=arg33_1]
#   %convolution_16 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf61
triton_poi_fused_convolution_35 = async_compile.triton('triton_poi_fused_convolution_35', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 13824, 'x': 6912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/5a/c5amdvjg3bnyofxh2e5dmdwilaortqbxlrrjskxq3mhzd5gxeqdh.py
# Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36, x_12, x_13], Original ATen: [aten.convolution, aten.relu, aten.sigmoid]
# Source node to ATen node mapping:
#   input_33 => convolution_16
#   input_34 => relu_16
#   input_35 => convolution_17
#   input_36 => relu_17
#   x_12 => convolution_18
#   x_13 => sigmoid
# Graph fragment:
#   %buf67 : Tensor "f32[1, 1, 256, 256][65536, 1, 256, 1]cuda:0" = PlaceHolder[target=buf67]
#   %arg38_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %convolution_16 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %arg35_1, %arg36_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : Tensor "f32[1, 8, 256, 256][524288, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg37_1, %arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_18,), kwargs = {})
#   return %sigmoid
triton_poi_fused_convolution_relu_sigmoid_36 = async_compile.triton('triton_poi_fused_convolution_relu_sigmoid_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_sigmoid_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_sigmoid_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (8, ), (1, ))
        assert_size_stride(arg2_1, (1, 3, 256, 256), (196608, 65536, 256, 1))
        assert_size_stride(arg3_1, (8, 8, 3, 3), (72, 9, 3, 1))
        assert_size_stride(arg4_1, (8, ), (1, ))
        assert_size_stride(arg5_1, (16, 8, 3, 3), (72, 9, 3, 1))
        assert_size_stride(arg6_1, (16, ), (1, ))
        assert_size_stride(arg7_1, (16, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg8_1, (16, ), (1, ))
        assert_size_stride(arg9_1, (32, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg10_1, (32, ), (1, ))
        assert_size_stride(arg11_1, (32, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg12_1, (32, ), (1, ))
        assert_size_stride(arg13_1, (64, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg14_1, (64, ), (1, ))
        assert_size_stride(arg15_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg16_1, (64, ), (1, ))
        assert_size_stride(arg17_1, (128, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg18_1, (128, ), (1, ))
        assert_size_stride(arg19_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg20_1, (128, ), (1, ))
        assert_size_stride(arg21_1, (64, 192, 3, 3), (1728, 9, 3, 1))
        assert_size_stride(arg22_1, (64, ), (1, ))
        assert_size_stride(arg23_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg24_1, (64, ), (1, ))
        assert_size_stride(arg25_1, (32, 96, 3, 3), (864, 9, 3, 1))
        assert_size_stride(arg26_1, (32, ), (1, ))
        assert_size_stride(arg27_1, (32, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg28_1, (32, ), (1, ))
        assert_size_stride(arg29_1, (16, 48, 3, 3), (432, 9, 3, 1))
        assert_size_stride(arg30_1, (16, ), (1, ))
        assert_size_stride(arg31_1, (16, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg32_1, (16, ), (1, ))
        assert_size_stride(arg33_1, (8, 24, 3, 3), (216, 9, 3, 1))
        assert_size_stride(arg34_1, (8, ), (1, ))
        assert_size_stride(arg35_1, (8, 8, 3, 3), (72, 9, 3, 1))
        assert_size_stride(arg36_1, (8, ), (1, ))
        assert_size_stride(arg37_1, (1, 8, 1, 1), (8, 1, 1, 1))
        assert_size_stride(arg38_1, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 3, 65536, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((8, 3, 3, 3), (27, 1, 9, 3), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 24, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 8, 256, 256), (524288, 1, 2048, 8), 'torch.ops.aten.convolution.default')
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf3, arg1_1, 524288, stream=stream0)
            del arg1_1
            buf4 = empty_strided_cuda((8, 8, 3, 3), (72, 1, 24, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_3:5
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_3.run(arg3_1, buf4, 64, 9, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:6
            buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf5, (1, 8, 256, 256), (524288, 1, 2048, 8), 'torch.ops.aten.convolution.default')
            del buf3
            del buf4
            buf59 = empty_strided_cuda((1, 24, 256, 256), (1572864, 65536, 256, 1), torch.float32)
            buf6 = reinterpret_tensor(buf59, (1, 8, 256, 256), (1572864, 65536, 256, 1), 1048576)  # alias
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_4:7
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_4.run(buf5, arg4_1, buf6, 8, 65536, stream=stream0)
            del arg4_1
            del buf5
            buf7 = empty_strided_cuda((1, 8, 128, 128), (131072, 1, 1024, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_5:8
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(buf6, buf7, 8, 16384, stream=stream0)
            buf8 = empty_strided_cuda((16, 8, 3, 3), (72, 1, 24, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_6:9
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(arg5_1, buf8, 128, 9, stream=stream0)
            del arg5_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:10
            buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf9, (1, 16, 128, 128), (262144, 1, 2048, 16), 'torch.ops.aten.convolution.default')
            del buf7
            del buf8
            buf10 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_7:11
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf10, arg6_1, 262144, stream=stream0)
            del arg6_1
            buf11 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_8:12
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(arg7_1, buf11, 256, 9, stream=stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:13
            buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf12, (1, 16, 128, 128), (262144, 1, 2048, 16), 'torch.ops.aten.convolution.default')
            del buf10
            buf51 = empty_strided_cuda((1, 48, 128, 128), (786432, 16384, 128, 1), torch.float32)
            buf13 = reinterpret_tensor(buf51, (1, 16, 128, 128), (786432, 16384, 128, 1), 524288)  # alias
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_9:14
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(buf12, arg8_1, buf13, 16, 16384, stream=stream0)
            del arg8_1
            del buf12
            buf14 = empty_strided_cuda((1, 16, 64, 64), (65536, 1, 1024, 16), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_10:15
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf13, buf14, 16, 4096, stream=stream0)
            buf15 = empty_strided_cuda((32, 16, 3, 3), (144, 1, 48, 16), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_11:16
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(arg9_1, buf15, 512, 9, stream=stream0)
            del arg9_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:17
            buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf16, (1, 32, 64, 64), (131072, 1, 2048, 32), 'torch.ops.aten.convolution.default')
            del buf14
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_12:18
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf17, arg10_1, 131072, stream=stream0)
            del arg10_1
            buf18 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_13:19
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_13.run(arg11_1, buf18, 1024, 9, stream=stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:20
            buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf19, (1, 32, 64, 64), (131072, 1, 2048, 32), 'torch.ops.aten.convolution.default')
            del buf17
            buf43 = empty_strided_cuda((1, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
            buf20 = reinterpret_tensor(buf43, (1, 32, 64, 64), (393216, 4096, 64, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_14:21
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf19, arg12_1, buf20, 32, 4096, stream=stream0)
            del arg12_1
            del buf19
            buf21 = empty_strided_cuda((1, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_15:22
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(buf20, buf21, 32, 1024, stream=stream0)
            buf22 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_16:23
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_16.run(arg13_1, buf22, 2048, 9, stream=stream0)
            del arg13_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:24
            buf23 = extern_kernels.convolution(buf21, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf23, (1, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution.default')
            del buf21
            del buf22
            buf24 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_17:25
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf24, arg14_1, 65536, stream=stream0)
            del arg14_1
            buf25 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_18:26
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg15_1, buf25, 4096, 9, stream=stream0)
            del arg15_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:27
            buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (1, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution.default')
            del buf24
            buf35 = reinterpret_tensor(buf0, (1, 192, 32, 32), (196608, 1024, 32, 1), 0); del buf0  # reuse
            buf27 = reinterpret_tensor(buf35, (1, 64, 32, 32), (196608, 1024, 32, 1), 131072)  # alias
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_19:28
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf26, arg16_1, buf27, 64, 1024, stream=stream0)
            del arg16_1
            del buf26
            buf28 = empty_strided_cuda((1, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_20:29
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_20.run(buf27, buf28, 64, 256, stream=stream0)
            buf29 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_21:30
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_21.run(arg17_1, buf29, 8192, 9, stream=stream0)
            del arg17_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:31
            buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf30, (1, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution.default')
            del buf28
            del buf29
            buf31 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_22:32
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_22.run(buf31, arg18_1, 32768, stream=stream0)
            del arg18_1
            buf32 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_23:33
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_23.run(arg19_1, buf32, 16384, 9, stream=stream0)
            del arg19_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:34
            buf33 = extern_kernels.convolution(buf31, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf33, (1, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution.default')
            del buf31
            del buf32
            buf34 = reinterpret_tensor(buf35, (1, 128, 32, 32), (196608, 1024, 32, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, x, input_5, input_6, input_7, input_8, x_1, input_9, input_10, input_11, input_12, x_2, input_13, input_14, input_15, input_16, x_3, input_17, input_18, input_19, input_20, x_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_max_pool2d_with_indices_mul_relu_unsqueeze_24:35
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_max_pool2d_with_indices_mul_relu_unsqueeze_24.run(buf33, arg20_1, buf34, 131072, stream=stream0)
            del arg20_1
            del buf33
            buf36 = empty_strided_cuda((1, 192, 32, 32), (196608, 1, 6144, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_25:36
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_25.run(buf35, buf36, 192, 1024, stream=stream0)
            del buf27
            del buf34
            del buf35
            buf37 = empty_strided_cuda((64, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_26:37
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(arg21_1, buf37, 12288, 9, stream=stream0)
            del arg21_1
            # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:38
            buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf38, (1, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution.default')
            del buf36
            del buf37
            buf39 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_17:39
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf39, arg22_1, 65536, stream=stream0)
            del arg22_1
            buf40 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_18:40
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg23_1, buf40, 4096, 9, stream=stream0)
            del arg23_1
            # Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:41
            buf41 = extern_kernels.convolution(buf39, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf41, (1, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution.default')
            del buf39
            del buf40
            buf42 = reinterpret_tensor(buf43, (1, 64, 64, 64), (393216, 4096, 64, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_21, input_22, input_23, input_24, x_6], Original ATen: [aten.convolution, aten.relu, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_27:42
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_27.run(buf41, arg24_1, buf42, 262144, stream=stream0)
            del arg24_1
            del buf41
            buf44 = empty_strided_cuda((1, 96, 64, 64), (393216, 1, 6144, 96), torch.float32)
            # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_28:43
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_28.run(buf43, buf44, 96, 4096, stream=stream0)
            del buf20
            del buf42
            del buf43
            buf45 = empty_strided_cuda((32, 96, 3, 3), (864, 1, 288, 96), torch.float32)
            # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_29:44
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_29.run(arg25_1, buf45, 3072, 9, stream=stream0)
            del arg25_1
            # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:45
            buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf46, (1, 32, 64, 64), (131072, 1, 2048, 32), 'torch.ops.aten.convolution.default')
            del buf44
            del buf45
            buf47 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_12:46
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf47, arg26_1, 131072, stream=stream0)
            del arg26_1
            buf48 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_13:47
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_13.run(arg27_1, buf48, 1024, 9, stream=stream0)
            del arg27_1
            # Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:48
            buf49 = extern_kernels.convolution(buf47, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf49, (1, 32, 64, 64), (131072, 1, 2048, 32), 'torch.ops.aten.convolution.default')
            del buf47
            del buf48
            buf50 = reinterpret_tensor(buf51, (1, 32, 128, 128), (786432, 16384, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_25, input_26, input_27, input_28, x_8], Original ATen: [aten.convolution, aten.relu, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_30:49
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_30.run(buf49, arg28_1, buf50, 524288, stream=stream0)
            del arg28_1
            del buf49
            buf52 = empty_strided_cuda((1, 48, 128, 128), (786432, 1, 6144, 48), torch.float32)
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_31:50
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_31.run(buf51, buf52, 48, 16384, stream=stream0)
            del buf13
            del buf50
            del buf51
            buf53 = empty_strided_cuda((16, 48, 3, 3), (432, 1, 144, 48), torch.float32)
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_32:51
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_32.run(arg29_1, buf53, 768, 9, stream=stream0)
            del arg29_1
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:52
            buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf54, (1, 16, 128, 128), (262144, 1, 2048, 16), 'torch.ops.aten.convolution.default')
            del buf52
            del buf53
            buf55 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_7:53
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf55, arg30_1, 262144, stream=stream0)
            del arg30_1
            buf56 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_29, input_30, input_31], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_8:54
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(arg31_1, buf56, 256, 9, stream=stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [input_29, input_30, input_31], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:55
            buf57 = extern_kernels.convolution(buf55, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf57, (1, 16, 128, 128), (262144, 1, 2048, 16), 'torch.ops.aten.convolution.default')
            del buf55
            del buf56
            buf58 = reinterpret_tensor(buf59, (1, 16, 256, 256), (1572864, 65536, 256, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_29, input_30, input_31, input_32, x_10], Original ATen: [aten.convolution, aten.relu, aten.arange, aten.add, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_33:56
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_mul_relu_unsqueeze_33.run(buf57, arg32_1, buf58, 1048576, stream=stream0)
            del arg32_1
            del buf57
            buf60 = empty_strided_cuda((1, 24, 256, 256), (1572864, 1, 6144, 24), torch.float32)
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_34:57
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_34.run(buf59, buf60, 24, 65536, stream=stream0)
            del buf58
            del buf59
            del buf6
            buf61 = empty_strided_cuda((8, 24, 3, 3), (216, 1, 72, 24), torch.float32)
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_35:58
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_35.run(arg33_1, buf61, 192, 9, stream=stream0)
            del arg33_1
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:59
            buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf62, (1, 8, 256, 256), (524288, 1, 2048, 8), 'torch.ops.aten.convolution.default')
            del buf60
            del buf61
            buf63 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:60
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf63, arg34_1, 524288, stream=stream0)
            del arg34_1
            buf64 = empty_strided_cuda((8, 8, 3, 3), (72, 1, 24, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_33, input_34, input_35], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_3:61
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_3.run(arg35_1, buf64, 64, 9, stream=stream0)
            del arg35_1
            # Topologically Sorted Source Nodes: [input_33, input_34, input_35], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:62
            buf65 = extern_kernels.convolution(buf63, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf65, (1, 8, 256, 256), (524288, 1, 2048, 8), 'torch.ops.aten.convolution.default')
            del buf63
            del buf64
            buf66 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:63
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf66, arg36_1, 524288, stream=stream0)
            del arg36_1
            # Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36, x_12], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:64
            buf67 = extern_kernels.convolution(buf66, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf67, (1, 1, 256, 256), (65536, 1, 256, 1), 'torch.ops.aten.convolution.default')
            del arg37_1
            del buf66
            buf68 = reinterpret_tensor(buf67, (1, 1, 256, 256), (65536, 65536, 256, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36, x_12, x_13], Original ATen: [aten.convolution, aten.relu, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_sigmoid_36:65
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_sigmoid_36.run(buf68, arg38_1, 65536, stream=stream0)
            del arg38_1
        return (buf68, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((16, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((8, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
