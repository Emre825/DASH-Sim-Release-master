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


# kernel path: /tmp/torchinductor_emre/dh/cdhupd5etjkzkiuks62adyw7dmoxa2up6z7zjompwwbphp6xs32a.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[1, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 451584, 'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 50176*y0), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/vy/cvyyavenxxmgmjkgd64nkz5lydd45oyrocm7cweuj3jdq2hpcnf4.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 3, 3, 3][27, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 13824, 'x': 6912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/4y/c4y3c4jerkwoqrip4lxbfou2tzi5ppdvn5i6xxkwvrnzyimip3ue.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %buf2 : Tensor "f32[1, 64, 224, 224][3211264, 1, 14336, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 38535424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_emre/t6/ct6rcn2l4jwzk4rrxmhqupjexriwslgcs4miqby44lwczpmxfojx.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf4
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_poi_fused_convolution_relu_3', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 294912, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/dj/cdjp5yjkchr7ayxkk2prwsrubgmvvmgsdqjstb2ivznwhdxtt2ap.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 1, 14336, 64]cuda:0" = PlaceHolder[target=relu_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 19267584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 112)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 128*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14400 + x0 + 128*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/zd/czdk5k2sizgdudermbmvmxg34piub7m2scyotmx2cse43z56bpvo.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
# Graph fragment:
#   %arg5_1 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf8
triton_poi_fused_convolution_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 589824, 'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/dh/cdhhm76hyrs4o5627y3hbffbykcuenv7ed7na3dupklcrzgvyzlf.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
# Graph fragment:
#   %buf9 : Tensor "f32[1, 128, 112, 112][1605632, 1, 14336, 128]cuda:0" = PlaceHolder[target=buf9]
#   %arg6_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   return %relu_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 19268096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_emre/vr/cvretfgu3hprkrzw42rilvyvqquafivdivzyoeiib2pdbtfuvbug.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
# Graph fragment:
#   %arg7_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf11
triton_poi_fused_convolution_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/56/c564hov2bvykw2yida5wd6726lzltih7ah3iuyy2noi5czw2vdcw.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 1, 14336, 128]cuda:0" = PlaceHolder[target=relu_3]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9633792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 56)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 256*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14464 + x0 + 256*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/2k/c2koncorcxp4jl7etglwrnrzx2ub4xzrzv7yvgyxgsspdifbitjb.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %arg9_1 : Tensor "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf15
triton_poi_fused_convolution_max_pool2d_with_indices_relu_9 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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


# kernel path: /tmp/torchinductor_emre/4y/c4y3etacz7rpooq5qavydvheuac4ww3qryr2nlgc74ydrcecdj24.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %buf16 : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=buf16]
#   %arg10_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   return %relu_4
triton_poi_fused_convolution_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9634816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ls/clsgoltl2v5ddkpkkid7zuhlmvyb7ozvd5s2j4hu4esxozt2hnrk.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %arg11_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf18
triton_poi_fused_convolution_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/hz/chzbar3py64sgms7lhwhcqhxwggyoxxujoumu7fslidkfqeyhqet.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=relu_6]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_4
triton_poi_fused_convolution_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 28)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 512*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14592 + x0 + 512*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/b4/cb4d5uf6yrgqcwtcw2aduxvov5p55jnrwfop4m6wefucs6elsq66.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_18 => convolution_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %arg15_1 : Tensor "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg15_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf25
triton_poi_fused_convolution_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/y4/cy4the5y3enz6bbt6a5wgwjzqt7e6spzgwfn6ox4yq637efizzrh.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %buf26 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=buf26]
#   %arg16_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   return %relu_7
triton_poi_fused_convolution_max_pool2d_with_indices_relu_14 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4818944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/wd/cwdgopviuvw7zxocn64a45uldmtsvrdawoz35i55sp2eayxtlagz.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %arg17_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf28
triton_poi_fused_convolution_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/im/cimwvztz4pc7vjq6pmrqunujeoby6kpw3ioibnqnfi4ppj2kayte.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => _low_memory_max_pool_with_offsets_3
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %relu_9 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=relu_9]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_6
triton_poi_fused_convolution_max_pool2d_with_indices_relu_16 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 14)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 28672*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 28672*x2), xmask)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 1024*x1 + 28672*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (14848 + x0 + 1024*x1 + 28672*x2), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/di/cdiu2mnayztxizulchti34nimj3bwxorpjq35udqhep4xoqearaf.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => _low_memory_max_pool_with_offsets_3
#   input_25 => convolution_10
#   input_26 => relu_10
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %buf36 : Tensor "f32[1, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = PlaceHolder[target=buf36]
#   %arg22_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   return %relu_10
triton_poi_fused_convolution_max_pool2d_with_indices_relu_17 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1206272}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/c6/cc62egpnumcjfopilvq7mutp6xkv5uqkilrxrsggbxkv6keyla7d.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, x], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => _low_memory_max_pool_with_offsets_3
#   input_25 => convolution_10
#   input_26 => relu_10
#   input_27 => convolution_11
#   input_28 => relu_11
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => _low_memory_max_pool_with_offsets_4
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   x => _adaptive_avg_pool2d
# Graph fragment:
#   %relu_12 : Tensor "f32[1, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = PlaceHolder[target=relu_12]
#   %getitem_8 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = PlaceHolder[target=getitem_8]
#   %convolution : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 224, 224][3211264, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 128, 112, 112][1605632, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %_adaptive_avg_pool2d : Tensor "f32[1, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%getitem_8, [7, 7]), kwargs = {})
#   return %getitem_8,%_adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_18 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 401408, 'x': 200704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_18(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 7)
    x2 = xindex // 7
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024*x1 + 14336*x2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (512 + y0 + 1024*x1 + 14336*x2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (7168 + y0 + 1024*x1 + 14336*x2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (7680 + y0 + 1024*x1 + 14336*x2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr1 + (x3 + 49*y0), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/gv/cgvqy2natiitbn4fhnvau2ipfbydnw3fbt43gx25jx2d3mnxorp7.py
# Topologically Sorted Source Nodes: [, input_33], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_1
#   input_33 => relu_13
# Graph fragment:
#   %arg28_1 : Tensor "f32[4096][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %mm_default_1 : Tensor "f32[1, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, %mm_default_1), kwargs = {})
#   %relu_13 : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   return %relu_13
triton_poi_fused_addmm_relu_19 = async_compile.triton('triton_poi_fused_addmm_relu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (1, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg3_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg4_1, (64, ), (1, ))
        assert_size_stride(arg5_1, (128, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg6_1, (128, ), (1, ))
        assert_size_stride(arg7_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg8_1, (128, ), (1, ))
        assert_size_stride(arg9_1, (256, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg10_1, (256, ), (1, ))
        assert_size_stride(arg11_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg12_1, (256, ), (1, ))
        assert_size_stride(arg13_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg14_1, (256, ), (1, ))
        assert_size_stride(arg15_1, (512, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg16_1, (512, ), (1, ))
        assert_size_stride(arg17_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg18_1, (512, ), (1, ))
        assert_size_stride(arg19_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg20_1, (512, ), (1, ))
        assert_size_stride(arg21_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg22_1, (512, ), (1, ))
        assert_size_stride(arg23_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg24_1, (512, ), (1, ))
        assert_size_stride(arg25_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg26_1, (512, ), (1, ))
        assert_size_stride(arg27_1, (4096, 25088), (25088, 1))
        assert_size_stride(arg28_1, (4096, ), (1, ))
        assert_size_stride(arg29_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg30_1, (4096, ), (1, ))
        assert_size_stride(arg31_1, (1000, 4096), (4096, 1))
        assert_size_stride(arg32_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 3, 50176, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 64, 224, 224), (3211264, 1, 14336, 64), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf3, arg1_1, 3211264, stream=stream0)
            del arg1_1
            buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_3:5
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_3.run(arg3_1, buf4, 4096, 9, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:6
            buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf5, (1, 64, 224, 224), (3211264, 1, 14336, 64), 'torch.ops.aten.convolution.default')
            del buf3
            del buf4
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:7
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf6, arg4_1, 3211264, stream=stream0)
            del arg4_1
            buf7 = empty_strided_cuda((1, 64, 112, 112), (802816, 1, 7168, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_4:8
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.run(buf6, buf7, 802816, stream=stream0)
            del buf6
            buf8 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_5:9
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(arg5_1, buf8, 8192, 9, stream=stream0)
            del arg5_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:10
            buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf9, (1, 128, 112, 112), (1605632, 1, 14336, 128), 'torch.ops.aten.convolution.default')
            del buf7
            del buf8
            buf10 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_6:11
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(buf10, arg6_1, 1605632, stream=stream0)
            del arg6_1
            buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_7:12
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(arg7_1, buf11, 16384, 9, stream=stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:13
            buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf12, (1, 128, 112, 112), (1605632, 1, 14336, 128), 'torch.ops.aten.convolution.default')
            del buf10
            del buf11
            buf13 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_6:14
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(buf13, arg8_1, 1605632, stream=stream0)
            del arg8_1
            buf14 = empty_strided_cuda((1, 128, 56, 56), (401408, 1, 7168, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_8:15
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf13, buf14, 401408, stream=stream0)
            del buf13
            buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_9:16
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(arg9_1, buf15, 32768, 9, stream=stream0)
            del arg9_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:17
            buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf16, (1, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
            del buf14
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_10:18
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf17, arg10_1, 802816, stream=stream0)
            del arg10_1
            buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_11:19
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(arg11_1, buf18, 65536, 9, stream=stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:20
            buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf19, (1, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
            del buf17
            buf20 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_10:21
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf20, arg12_1, 802816, stream=stream0)
            del arg12_1
            buf21 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_11:22
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(arg13_1, buf21, 65536, 9, stream=stream0)
            del arg13_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:23
            buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf22, (1, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
            del buf20
            del buf21
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_10:24
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf23, arg14_1, 802816, stream=stream0)
            del arg14_1
            buf24 = empty_strided_cuda((1, 256, 28, 28), (200704, 1, 7168, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_12:25
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf23, buf24, 200704, stream=stream0)
            del buf23
            buf25 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_13:26
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_13.run(arg15_1, buf25, 131072, 9, stream=stream0)
            del arg15_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:27
            buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (1, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
            del buf24
            del buf25
            buf27 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_14:28
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf27, arg16_1, 401408, stream=stream0)
            del arg16_1
            buf28 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_15:29
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg17_1, buf28, 262144, 9, stream=stream0)
            del arg17_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:30
            buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf29, (1, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
            del buf27
            buf30 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_14:31
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf30, arg18_1, 401408, stream=stream0)
            del arg18_1
            buf31 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_15:32
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg19_1, buf31, 262144, 9, stream=stream0)
            del arg19_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:33
            buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf32, (1, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
            del buf30
            buf33 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_14:34
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf33, arg20_1, 401408, stream=stream0)
            del arg20_1
            buf34 = empty_strided_cuda((1, 512, 14, 14), (100352, 1, 7168, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_16:35
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_16.run(buf33, buf34, 100352, stream=stream0)
            del buf33
            buf35 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_15:36
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg21_1, buf35, 262144, 9, stream=stream0)
            del arg21_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:37
            buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf36, (1, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution.default')
            del buf34
            buf37 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_17:38
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf37, arg22_1, 100352, stream=stream0)
            del arg22_1
            buf38 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_15:39
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg23_1, buf38, 262144, 9, stream=stream0)
            del arg23_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:40
            buf39 = extern_kernels.convolution(buf37, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf39, (1, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution.default')
            del buf37
            buf40 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_17:41
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf40, arg24_1, 100352, stream=stream0)
            del arg24_1
            buf41 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_15:42
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg25_1, buf41, 262144, 9, stream=stream0)
            del arg25_1
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] extern_kernels.convolution:43
            buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf42, (1, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution.default')
            del buf40
            del buf41
            buf43 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_17:44
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf43, arg26_1, 100352, stream=stream0)
            del arg26_1
            buf45 = empty_strided_cuda((1, 512, 7, 7), (25088, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, x], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            # [Provenance debug handles] triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_18:45
            stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_18.run(buf43, buf45, 512, 49, stream=stream0)
            del buf43
            buf46 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, x, x_1, input_32, ], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:46
            extern_kernels.mm(reinterpret_tensor(buf45, (1, 25088), (0, 1), 0), reinterpret_tensor(arg27_1, (25088, 4096), (1, 25088), 0), out=buf46)
            del arg27_1
            del buf45
            buf47 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [, input_33], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_19:47
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_19.run(buf47, arg28_1, 4096, stream=stream0)
            del arg28_1
            buf48 = empty_strided_cuda((1, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, input_33, input_35], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:48
            extern_kernels.mm(buf47, reinterpret_tensor(arg29_1, (4096, 4096), (1, 4096), 0), out=buf48)
            del arg29_1
            del buf47
            buf49 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [, input_36], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_19:49
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_19.run(buf49, arg30_1, 4096, stream=stream0)
            del arg30_1
            buf50 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, input_36, input_38], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.addmm:50
            extern_kernels.addmm(arg32_1, buf49, reinterpret_tensor(arg31_1, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf50)
            del arg31_1
            del arg32_1
            del buf49
        return (buf50, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((4096, 25088), (25088, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
