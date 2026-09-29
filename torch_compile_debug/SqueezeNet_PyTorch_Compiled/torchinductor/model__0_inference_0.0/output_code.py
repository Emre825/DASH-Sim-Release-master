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
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_emre/th/cthvxnjoip7qtvxabi6alaqvb6un54xuqiox4wafdmcm2zsxk7ex.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %buf2 : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9634048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_emre/jk/cjkafdwn47oitibf5mmykgcq5i3nj72qiuzy6cssizcwghaa3jzt.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = PlaceHolder[target=relu]
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8830976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 3584
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x4 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x2), tmp10, other=float("-inf"))
    tmp12 = 1 + 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x2), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 2 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (128 + x0 + 128*x1 + 14336*x2), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 1 + 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x2), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x2), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (7296 + x0 + 128*x1 + 14336*x2), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 2 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (14336 + x0 + 128*x1 + 14336*x2), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14400 + x0 + 128*x1 + 14336*x2), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (14464 + x0 + 128*x1 + 14336*x2), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/pw/cpwjgfj242aw2otphfrqinbmrah3xa4suy2owp7sgkkarpkz57wo.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => relu_1
# Graph fragment:
#   %buf5 : Tensor "f32[1, 16, 56, 56][50176, 1, 896, 16]cuda:0" = PlaceHolder[target=buf5]
#   %arg4_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution_1 : Tensor "f32[1, 16, 56, 56][50176, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 16, 56, 56][50176, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_convolution_relu_4 = async_compile.triton('triton_poi_fused_convolution_relu_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 602176}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/xc/cxcee5m25fqlmkbxg4i7xuxm5em235z4o4czjrpuagahlpgjx5zf.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_8 => convolution_3
# Graph fragment:
#   %arg7_1 : Tensor "f32[64, 16, 3, 3][144, 9, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convolution_3 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf8
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 73728, 'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/jl/cjlnv3yw4tsed6w7ebneiw7dr2qjz3dbjhk5a7iwg3kf4ukoxlpm.py
# Topologically Sorted Source Nodes: [input_6, input_7, input_8, input_9, x], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   x => cat
# Graph fragment:
#   %buf7 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf7]
#   %arg6_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf9 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf9]
#   %arg8_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convolution_2 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg5_1, %arg6_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %cat : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_2, %relu_3], 1), kwargs = {})
#   return %cat
triton_poi_fused_cat_convolution_relu_6 = async_compile.triton('triton_poi_fused_cat_convolution_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4817408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 128, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (64*x1 + ((-64) + x0)), tmp12, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-64) + x0), tmp12, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/jr/cjrfp2aytfmg5eqx47rtjjcb4exwn4jfqlet7mry6ykw2iu57afu.py
# Topologically Sorted Source Nodes: [input_12, input_13, input_14, input_15, x_1], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_12 => convolution_5
#   input_13 => relu_5
#   input_14 => convolution_6
#   input_15 => relu_6
#   x_1 => cat_1
# Graph fragment:
#   %buf13 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf13]
#   %arg12_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %buf15 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf15]
#   %arg14_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %convolution_5 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %cat_1 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_6], 1), kwargs = {})
#   return %cat_1
triton_poi_fused_cat_convolution_relu_7 = async_compile.triton('triton_poi_fused_cat_convolution_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1606144, 'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1, 1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 128, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (64*x1 + ((-64) + y0)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to((-64) + y0, [YBLOCK, XBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x1 + 3136*y0), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/a4/ca4pborpmjeos4ytet3wzl7iccz4g5k7t6v5tumvjxc7u3ltj4vg.py
# Topologically Sorted Source Nodes: [input_12, input_13, input_14, input_15, x_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.cat, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_12 => convolution_5
#   input_13 => relu_5
#   input_14 => convolution_6
#   input_15 => relu_6
#   x_1 => cat_1
#   x_2 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %cat_1 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %convolution_5 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : Tensor "f32[1, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %cat_1 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_6], 1), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   return %getitem_2
triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 802816, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex // 28
    x1 = (xindex % 28)
    y0 = yindex
    x3 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (2*x1 + 112*x2 + 3136*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 1 + 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (1 + 2*x1 + 112*x2 + 3136*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 2 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (2 + 2*x1 + 112*x2 + 3136*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 1 + 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (56 + 2*x1 + 112*x2 + 3136*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (57 + 2*x1 + 112*x2 + 3136*y0), tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (58 + 2*x1 + 112*x2 + 3136*y0), tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 2 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (112 + 2*x1 + 112*x2 + 3136*y0), tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (113 + 2*x1 + 112*x2 + 3136*y0), tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (114 + 2*x1 + 112*x2 + 3136*y0), tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (y0 + 128*x3), tmp51, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ls/clse4t7iegxmkkmkwddvmjbd4mzhmzabwgpog7mvkgxslojpsf4t.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_16 => convolution_7
#   input_17 => relu_7
# Graph fragment:
#   %buf18 : Tensor "f32[1, 32, 28, 28][25088, 1, 896, 32]cuda:0" = PlaceHolder[target=buf18]
#   %arg16_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convolution_7 : Tensor "f32[1, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg15_1, %arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : Tensor "f32[1, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   return %relu_7
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_poi_fused_convolution_relu_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 301184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ja/cjaurcinpzyjx2k22xrtwrsn4433i7sd36ffdkif444nzt7i4n5c.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_20 => convolution_9
# Graph fragment:
#   %arg19_1 : Tensor "f32[128, 32, 3, 3][288, 9, 3, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %convolution_9 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf21
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 294912, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ek/cekprrhzwazial3i5wzwd2qy77bhf2abngd4ra3zjt5uzdvnxigk.py
# Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_3], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_18 => convolution_8
#   input_19 => relu_8
#   input_20 => convolution_9
#   input_21 => relu_9
#   x_3 => cat_2
# Graph fragment:
#   %buf20 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf20]
#   %arg18_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %buf22 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf22]
#   %arg20_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %convolution_8 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %cat_2 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_8, %relu_9], 1), kwargs = {})
#   return %cat_2
triton_poi_fused_cat_convolution_relu_11 = async_compile.triton('triton_poi_fused_cat_convolution_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2409472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 256, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (128*x1 + ((-128) + x0)), tmp12, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-128) + x0), tmp12, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/4j/c4jhqhrbg6z6uamwyuw3fykxabjyouzbyvclpbds6mrylm7pb2ev.py
# Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, x_4], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_24 => convolution_11
#   input_25 => relu_11
#   input_26 => convolution_12
#   input_27 => relu_12
#   x_4 => cat_3
# Graph fragment:
#   %buf26 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf26]
#   %arg24_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %buf28 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf28]
#   %arg26_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %convolution_11 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %cat_3 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_11, %relu_12], 1), kwargs = {})
#   return %cat_3
triton_poi_fused_cat_convolution_relu_12 = async_compile.triton('triton_poi_fused_cat_convolution_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 803840, 'x': 1605632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1, 1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 256, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (128*x1 + ((-128) + y0)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to((-128) + y0, [YBLOCK, XBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x1 + 784*y0), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/qd/cqdl6dyr2y2l3by6omns5n2ymginhhotbtphqasugm6tkeqni5rl.py
# Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, x_4, x_5], Original ATen: [aten.convolution, aten.relu, aten.cat, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_24 => convolution_11
#   input_25 => relu_11
#   input_26 => convolution_12
#   input_27 => relu_12
#   x_4 => cat_3
#   x_5 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %cat_3 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]cuda:0" = PlaceHolder[target=cat_3]
#   %convolution_11 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : Tensor "f32[1, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %cat_3 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_11, %relu_12], 1), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   return %getitem_4
triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 401408, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex // 14
    x1 = (xindex % 14)
    y0 = yindex
    x3 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (2*x1 + 56*x2 + 784*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 1 + 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (1 + 2*x1 + 56*x2 + 784*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 2 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (2 + 2*x1 + 56*x2 + 784*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 1 + 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (28 + 2*x1 + 56*x2 + 784*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (29 + 2*x1 + 56*x2 + 784*y0), tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (30 + 2*x1 + 56*x2 + 784*y0), tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 2 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (56 + 2*x1 + 56*x2 + 784*y0), tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (57 + 2*x1 + 56*x2 + 784*y0), tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (58 + 2*x1 + 56*x2 + 784*y0), tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (y0 + 256*x3), tmp51, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/vl/cvl2flh67j447iebugijegi3vzg7xtcmao4qeponkcy2awtvsflx.py
# Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_28 => convolution_13
#   input_29 => relu_13
# Graph fragment:
#   %buf31 : Tensor "f32[1, 48, 14, 14][9408, 1, 672, 48]cuda:0" = PlaceHolder[target=buf31]
#   %arg28_1 : Tensor "f32[48][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %convolution_13 : Tensor "f32[1, 48, 14, 14][9408, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg27_1, %arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : Tensor "f32[1, 48, 14, 14][9408, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   return %relu_13
triton_poi_fused_convolution_relu_14 = async_compile.triton('triton_poi_fused_convolution_relu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 113088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/f7/cf753dikgsnlk4mlwocbquuujkjs4afghbnmxn2yz2mf3gizpmtt.py
# Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_32 => convolution_15
# Graph fragment:
#   %arg31_1 : Tensor "f32[192, 48, 3, 3][432, 9, 3, 1]cuda:0" = PlaceHolder[target=arg31_1]
#   %convolution_15 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg31_1, %arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf34
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 663552, 'x': 331776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 432*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/no/cnomcleljpafhl4isbjc7c6fk4j3ffwyz4nghfec3gq2q4f3yifc.py
# Topologically Sorted Source Nodes: [input_30, input_31, input_32, input_33, x_6], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_30 => convolution_14
#   input_31 => relu_14
#   input_32 => convolution_15
#   input_33 => relu_15
#   x_6 => cat_4
# Graph fragment:
#   %buf33 : Tensor "f32[1, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = PlaceHolder[target=buf33]
#   %arg30_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %buf35 : Tensor "f32[1, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = PlaceHolder[target=buf35]
#   %arg32_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %convolution_14 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg29_1, %arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg31_1, %arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %cat_4 : Tensor "f32[1, 384, 14, 14][75264, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_14, %relu_15], 1), kwargs = {})
#   return %cat_4
triton_poi_fused_cat_convolution_relu_16 = async_compile.triton('triton_poi_fused_cat_convolution_relu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 904704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 384, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (192*x1 + ((-192) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-192) + x0), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/t4/ct4oyf2izazhqpdoujbsqifco7lv67xhajr36qiha6jmmmbf5stk.py
# Topologically Sorted Source Nodes: [input_36, input_37, input_38, input_39, x_7, input_40, input_41], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_36 => convolution_17
#   input_37 => relu_17
#   input_38 => convolution_18
#   input_39 => relu_18
#   input_40 => convolution_19
#   input_41 => relu_19
#   x_7 => cat_5
# Graph fragment:
#   %buf43 : Tensor "f32[1, 64, 14, 14][12544, 1, 896, 64]cuda:0" = PlaceHolder[target=buf43]
#   %arg40_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %convolution_17 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %arg35_1, %arg36_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : Tensor "f32[1, 192, 14, 14][37632, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %cat_5 : Tensor "f32[1, 384, 14, 14][75264, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_17, %relu_18], 1), kwargs = {})
#   %convolution_19 : Tensor "f32[1, 64, 14, 14][12544, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %arg39_1, %arg40_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : Tensor "f32[1, 64, 14, 14][12544, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   return %relu_19
triton_poi_fused_cat_convolution_relu_17 = async_compile.triton('triton_poi_fused_cat_convolution_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 150784}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/6f/c6fenjsturfngif3txsahztilziqx5zsybxxa5i75fv4z5fam354.py
# Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_44 => convolution_21
# Graph fragment:
#   %arg43_1 : Tensor "f32[256, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg43_1]
#   %convolution_21 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg43_1, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf46
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/le/cleuw2yhmzlnja64n3by5yxdkkprlku7yj37whmw5w63kny4wgqe.py
# Topologically Sorted Source Nodes: [input_42, input_43, input_44, input_45, x_8], Original ATen: [aten.convolution, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   input_42 => convolution_20
#   input_43 => relu_20
#   input_44 => convolution_21
#   input_45 => relu_21
#   x_8 => cat_6
# Graph fragment:
#   %buf45 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf45]
#   %arg42_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %buf47 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf47]
#   %arg44_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %convolution_20 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg41_1, %arg42_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg43_1, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %cat_6 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_20, %relu_21], 1), kwargs = {})
#   return %cat_6
triton_poi_fused_cat_convolution_relu_19 = async_compile.triton('triton_poi_fused_cat_convolution_relu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1206272}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 512, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (256*x1 + ((-256) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-256) + x0), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/zo/czokwhd2hjnrtg6r6fhvj2f2dpsgz4snec7pi5xlpzmrefjs63lw.py
# Topologically Sorted Source Nodes: [input_48, input_49, input_50, input_51, x_9, input_53, input_54, input_55], Original ATen: [aten.convolution, aten.relu, aten.cat, aten.mean]
# Source node to ATen node mapping:
#   input_48 => convolution_23
#   input_49 => relu_23
#   input_50 => convolution_24
#   input_51 => relu_24
#   input_53 => convolution_25
#   input_54 => relu_25
#   input_55 => mean
#   x_9 => cat_7
# Graph fragment:
#   %buf55 : Tensor "f32[1, 1000, 14, 14][196000, 1, 14000, 1000]cuda:0" = PlaceHolder[target=buf55]
#   %arg52_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %buf56 : Tensor "f32[1, 1000, 1, 1][1000, 1, 1000, 1000]cuda:0" = PlaceHolder[target=buf56]
#   %convolution_23 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg47_1, %arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %convolution_24 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg49_1, %arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : Tensor "f32[1, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %cat_7 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_23, %relu_24], 1), kwargs = {})
#   %convolution_25 : Tensor "f32[1, 1000, 14, 14][196000, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_7, %arg51_1, %arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : Tensor "f32[1, 1000, 14, 14][196000, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %mean : Tensor "f32[1, 1000, 1, 1][1000, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_25, [-1, -2], True), kwargs = {})
#   return %buf56,%mean
triton_red_fused_cat_convolution_mean_relu_20 = async_compile.triton('triton_red_fused_cat_convolution_mean_relu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 796000, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_relu_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1000
    r0_numel = 196
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1000*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = 196.0
    tmp9 = (tmp6 / tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (1, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg3_1, (16, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg4_1, (16, ), (1, ))
        assert_size_stride(arg5_1, (64, 16, 1, 1), (16, 1, 1, 1))
        assert_size_stride(arg6_1, (64, ), (1, ))
        assert_size_stride(arg7_1, (64, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg8_1, (64, ), (1, ))
        assert_size_stride(arg9_1, (16, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg10_1, (16, ), (1, ))
        assert_size_stride(arg11_1, (64, 16, 1, 1), (16, 1, 1, 1))
        assert_size_stride(arg12_1, (64, ), (1, ))
        assert_size_stride(arg13_1, (64, 16, 3, 3), (144, 9, 3, 1))
        assert_size_stride(arg14_1, (64, ), (1, ))
        assert_size_stride(arg15_1, (32, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg16_1, (32, ), (1, ))
        assert_size_stride(arg17_1, (128, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(arg18_1, (128, ), (1, ))
        assert_size_stride(arg19_1, (128, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg20_1, (128, ), (1, ))
        assert_size_stride(arg21_1, (32, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg22_1, (32, ), (1, ))
        assert_size_stride(arg23_1, (128, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(arg24_1, (128, ), (1, ))
        assert_size_stride(arg25_1, (128, 32, 3, 3), (288, 9, 3, 1))
        assert_size_stride(arg26_1, (128, ), (1, ))
        assert_size_stride(arg27_1, (48, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg28_1, (48, ), (1, ))
        assert_size_stride(arg29_1, (192, 48, 1, 1), (48, 1, 1, 1))
        assert_size_stride(arg30_1, (192, ), (1, ))
        assert_size_stride(arg31_1, (192, 48, 3, 3), (432, 9, 3, 1))
        assert_size_stride(arg32_1, (192, ), (1, ))
        assert_size_stride(arg33_1, (48, 384, 1, 1), (384, 1, 1, 1))
        assert_size_stride(arg34_1, (48, ), (1, ))
        assert_size_stride(arg35_1, (192, 48, 1, 1), (48, 1, 1, 1))
        assert_size_stride(arg36_1, (192, ), (1, ))
        assert_size_stride(arg37_1, (192, 48, 3, 3), (432, 9, 3, 1))
        assert_size_stride(arg38_1, (192, ), (1, ))
        assert_size_stride(arg39_1, (64, 384, 1, 1), (384, 1, 1, 1))
        assert_size_stride(arg40_1, (64, ), (1, ))
        assert_size_stride(arg41_1, (256, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg42_1, (256, ), (1, ))
        assert_size_stride(arg43_1, (256, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg44_1, (256, ), (1, ))
        assert_size_stride(arg45_1, (64, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg46_1, (64, ), (1, ))
        assert_size_stride(arg47_1, (256, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg48_1, (256, ), (1, ))
        assert_size_stride(arg49_1, (256, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg50_1, (256, ), (1, ))
        assert_size_stride(arg51_1, (1000, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg52_1, (1000, ), (1, ))
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
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 64, 112, 112), (802816, 1, 7168, 64), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf3, arg1_1, 802816, stream=stream0)
            del arg1_1
            buf4 = empty_strided_cuda((1, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_3:5
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(buf3, buf4, 200704, stream=stream0)
            del buf3
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:6
            buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf5, (1, 16, 56, 56), (50176, 1, 896, 16), 'torch.ops.aten.convolution.default')
            del arg3_1
            del buf4
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_4:7
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_4.run(buf6, arg4_1, 50176, stream=stream0)
            del arg4_1
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:8
            buf7 = extern_kernels.convolution(buf6, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf7, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
            del arg5_1
            buf8 = empty_strided_cuda((64, 16, 3, 3), (144, 1, 48, 16), torch.float32)
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_5:9
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_5.run(arg7_1, buf8, 1024, 9, stream=stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:10
            buf9 = extern_kernels.convolution(buf6, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf9, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
            del buf6
            buf10 = empty_strided_cuda((1, 128, 56, 56), (401408, 1, 7168, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_6, input_7, input_8, input_9, x], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_6:11
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_6.run(buf7, arg6_1, buf9, arg8_1, buf10, 401408, stream=stream0)
            del arg6_1
            del arg8_1
            del buf7
            del buf9
            # Topologically Sorted Source Nodes: [input_6, input_7, input_8, input_9, x, input_10], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] extern_kernels.convolution:12
            buf11 = extern_kernels.convolution(buf10, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf11, (1, 16, 56, 56), (50176, 1, 896, 16), 'torch.ops.aten.convolution.default')
            del arg9_1
            buf12 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_6, input_7, input_8, input_9, x, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_4:13
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_4.run(buf12, arg10_1, 50176, stream=stream0)
            del arg10_1
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:14
            buf13 = extern_kernels.convolution(buf12, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf13, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
            del arg11_1
            buf14 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_5:15
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_5.run(arg13_1, buf14, 1024, 9, stream=stream0)
            del arg13_1
            # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:16
            buf15 = extern_kernels.convolution(buf12, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf15, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
            del buf14
            buf16 = reinterpret_tensor(buf10, (1, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_12, input_13, input_14, input_15, x_1], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_7:17
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_7.run(buf13, arg12_1, buf15, arg14_1, buf16, 128, 3136, stream=stream0)
            del arg12_1
            del arg14_1
            del buf13
            buf17 = empty_strided_cuda((1, 128, 28, 28), (100352, 1, 3584, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_12, input_13, input_14, input_15, x_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.cat, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_8:18
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_8.run(buf16, buf17, 128, 784, stream=stream0)
            del buf16
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:19
            buf18 = extern_kernels.convolution(buf17, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf18, (1, 32, 28, 28), (25088, 1, 896, 32), 'torch.ops.aten.convolution.default')
            del arg15_1
            del buf17
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_9:20
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_9.run(buf19, arg16_1, 25088, stream=stream0)
            del arg16_1
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:21
            buf20 = extern_kernels.convolution(buf19, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf20, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
            del arg17_1
            buf21 = empty_strided_cuda((128, 32, 3, 3), (288, 1, 96, 32), torch.float32)
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_10:22
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_10.run(arg19_1, buf21, 4096, 9, stream=stream0)
            del arg19_1
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:23
            buf22 = extern_kernels.convolution(buf19, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf22, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
            del buf19
            buf23 = reinterpret_tensor(buf15, (1, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_3], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_11:24
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_11.run(buf20, arg18_1, buf22, arg20_1, buf23, 200704, stream=stream0)
            del arg18_1
            del arg20_1
            del buf20
            del buf22
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_3, input_22], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] extern_kernels.convolution:25
            buf24 = extern_kernels.convolution(buf23, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf24, (1, 32, 28, 28), (25088, 1, 896, 32), 'torch.ops.aten.convolution.default')
            del arg21_1
            buf25 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_3, input_22, input_23], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_9:26
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_9.run(buf25, arg22_1, 25088, stream=stream0)
            del arg22_1
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:27
            buf26 = extern_kernels.convolution(buf25, arg23_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
            del arg23_1
            buf27 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_10:28
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_10.run(arg25_1, buf27, 4096, 9, stream=stream0)
            del arg25_1
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:29
            buf28 = extern_kernels.convolution(buf25, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf28, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
            del buf25
            del buf27
            buf29 = reinterpret_tensor(buf23, (1, 256, 28, 28), (200704, 784, 28, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, x_4], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_12:30
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_12.run(buf26, arg24_1, buf28, arg26_1, buf29, 256, 784, stream=stream0)
            del arg24_1
            del arg26_1
            del buf26
            buf30 = reinterpret_tensor(buf12, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, x_4, x_5], Original ATen: [aten.convolution, aten.relu, aten.cat, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_13:31
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_13.run(buf29, buf30, 256, 196, stream=stream0)
            del buf29
            # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:32
            buf31 = extern_kernels.convolution(buf30, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf31, (1, 48, 14, 14), (9408, 1, 672, 48), 'torch.ops.aten.convolution.default')
            del arg27_1
            del buf30
            buf32 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_14:33
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_14.run(buf32, arg28_1, 9408, stream=stream0)
            del arg28_1
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:34
            buf33 = extern_kernels.convolution(buf32, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf33, (1, 192, 14, 14), (37632, 1, 2688, 192), 'torch.ops.aten.convolution.default')
            del arg29_1
            buf34 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
            # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_15:35
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_15.run(arg31_1, buf34, 9216, 9, stream=stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:36
            buf35 = extern_kernels.convolution(buf32, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf35, (1, 192, 14, 14), (37632, 1, 2688, 192), 'torch.ops.aten.convolution.default')
            del buf32
            buf36 = empty_strided_cuda((1, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_30, input_31, input_32, input_33, x_6], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_16:37
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_16.run(buf33, arg30_1, buf35, arg32_1, buf36, 75264, stream=stream0)
            del arg30_1
            del arg32_1
            del buf33
            del buf35
            # Topologically Sorted Source Nodes: [input_30, input_31, input_32, input_33, x_6, input_34], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] extern_kernels.convolution:38
            buf37 = extern_kernels.convolution(buf36, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf37, (1, 48, 14, 14), (9408, 1, 672, 48), 'torch.ops.aten.convolution.default')
            del arg33_1
            buf38 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [input_30, input_31, input_32, input_33, x_6, input_34, input_35], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_14:39
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_14.run(buf38, arg34_1, 9408, stream=stream0)
            del arg34_1
            # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:40
            buf39 = extern_kernels.convolution(buf38, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf39, (1, 192, 14, 14), (37632, 1, 2688, 192), 'torch.ops.aten.convolution.default')
            del arg35_1
            buf40 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_15:41
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_15.run(arg37_1, buf40, 9216, 9, stream=stream0)
            del arg37_1
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:42
            buf41 = extern_kernels.convolution(buf38, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf41, (1, 192, 14, 14), (37632, 1, 2688, 192), 'torch.ops.aten.convolution.default')
            del buf38
            del buf40
            buf42 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [input_36, input_37, input_38, input_39, x_7], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_16:43
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_16.run(buf39, arg36_1, buf41, arg38_1, buf42, 75264, stream=stream0)
            del arg36_1
            del arg38_1
            del buf39
            del buf41
            # Topologically Sorted Source Nodes: [input_36, input_37, input_38, input_39, x_7, input_40], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] extern_kernels.convolution:44
            buf43 = extern_kernels.convolution(buf42, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf43, (1, 64, 14, 14), (12544, 1, 896, 64), 'torch.ops.aten.convolution.default')
            del arg39_1
            del buf42
            buf44 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [input_36, input_37, input_38, input_39, x_7, input_40, input_41], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_17:45
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_17.run(buf44, arg40_1, 12544, stream=stream0)
            del arg40_1
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:46
            buf45 = extern_kernels.convolution(buf44, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf45, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
            del arg41_1
            buf46 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_18:47
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_18.run(arg43_1, buf46, 16384, 9, stream=stream0)
            del arg43_1
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:48
            buf47 = extern_kernels.convolution(buf44, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf47, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
            del buf44
            buf48 = reinterpret_tensor(buf28, (1, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [input_42, input_43, input_44, input_45, x_8], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_19:49
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_19.run(buf45, arg42_1, buf47, arg44_1, buf48, 100352, stream=stream0)
            del arg42_1
            del arg44_1
            del buf45
            del buf47
            # Topologically Sorted Source Nodes: [input_42, input_43, input_44, input_45, x_8, input_46], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] extern_kernels.convolution:50
            buf49 = extern_kernels.convolution(buf48, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf49, (1, 64, 14, 14), (12544, 1, 896, 64), 'torch.ops.aten.convolution.default')
            del arg45_1
            buf50 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [input_42, input_43, input_44, input_45, x_8, input_46, input_47], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_17:51
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_17.run(buf50, arg46_1, 12544, stream=stream0)
            del arg46_1
            # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:52
            buf51 = extern_kernels.convolution(buf50, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf51, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
            del arg47_1
            buf52 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_18:53
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_18.run(arg49_1, buf52, 16384, 9, stream=stream0)
            del arg49_1
            # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:54
            buf53 = extern_kernels.convolution(buf50, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf53, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
            del buf50
            del buf52
            buf54 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [input_48, input_49, input_50, input_51, x_9], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] triton_poi_fused_cat_convolution_relu_19:55
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_convolution_relu_19.run(buf51, arg48_1, buf53, arg50_1, buf54, 100352, stream=stream0)
            del arg48_1
            del arg50_1
            del buf51
            del buf53
            # Topologically Sorted Source Nodes: [input_48, input_49, input_50, input_51, x_9, input_53], Original ATen: [aten.convolution, aten.relu, aten.cat]
            # [Provenance debug handles] extern_kernels.convolution:56
            buf55 = extern_kernels.convolution(buf54, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf55, (1, 1000, 14, 14), (196000, 1, 14000, 1000), 'torch.ops.aten.convolution.default')
            del arg51_1
            del buf54
            buf56 = empty_strided_cuda((1, 1000, 1, 1), (1000, 1, 1000, 1000), torch.float32)
            buf57 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [input_48, input_49, input_50, input_51, x_9, input_53, input_54, input_55], Original ATen: [aten.convolution, aten.relu, aten.cat, aten.mean]
            # [Provenance debug handles] triton_red_fused_cat_convolution_mean_relu_20:57
            stream0 = get_raw_stream(0)
            triton_red_fused_cat_convolution_mean_relu_20.run(buf57, buf55, arg52_1, 1000, 196, stream=stream0)
            del arg52_1
            del buf55
        return (reinterpret_tensor(buf57, (1, 1000), (1000, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
