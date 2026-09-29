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


# kernel path: /tmp/torchinductor_emre/ib/cibwxcmqvawntn25gzkkikw2dquunid2ujpt3ig6cjaddpdsv4az.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[64, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[64, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 77070336, 'x': 38535168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/gc/cgcsjl5ncmovx73wrgmpi6xwumhir3pjycburehiwwfe4ed4j2ns.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 3, 11, 11][363, 121, 11, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[64, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 185856, 'x': 92928}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 121
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
    tmp0 = tl.load(in_ptr0 + (x2 + 121*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 363*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/lg/clg3gthjaisksfoo62mo6pufix7v2wy6bojtclmoenl4jeoukuto.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %buf2 : Tensor "f32[64, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[64, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[64, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 148685056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12390400
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


# kernel path: /tmp/torchinductor_emre/t5/ct5wsyn2gpral4paeczsb7jcwychk6byrdn7e2qlar4t6wtsrkvr.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[64, 64, 55, 55][193600, 1, 3520, 64]cuda:0" = PlaceHolder[target=relu]
#   %convolution : Tensor "f32[64, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[64, 64, 55, 55][193600, 3025, 55, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 131383296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2985984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 27)
    x2 = ((xindex // 1728) % 27)
    x3 = xindex // 46656
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp5 = tl.load(in_ptr0 + (3520 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp7 = tl.load(in_ptr0 + (3584 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp9 = tl.load(in_ptr0 + (3648 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp11 = tl.load(in_ptr0 + (7040 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp13 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp15 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 7040*x2 + 193600*x3), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/52/c52322t5ednynrpsys43kqutq2a6s2yikd6jex6cvdojocin7kzu.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_4 => convolution_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[192, 64, 5, 5][1600, 25, 5, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution_1 : Tensor "f32[64, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf5
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2457600, 'x': 1228800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 1600*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/b2/cb24omrrtn4d5tbho7mxpqdtzpqmzjqbejdy2dghripurwhgbkvt.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => relu_1
# Graph fragment:
#   %buf6 : Tensor "f32[64, 192, 27, 27][139968, 1, 5184, 192]cuda:0" = PlaceHolder[target=buf6]
#   %arg4_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution_1 : Tensor "f32[64, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[64, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_convolution_relu_5 = async_compile.triton('triton_poi_fused_convolution_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 107496192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8957952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ac/cacesv4vqmbvlb6pmdffuzhlwvfxrswgikz2nxoeyqe2kktqzhj2.py
# Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => relu_1
#   input_6 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %relu_1 : Tensor "f32[64, 192, 27, 27][139968, 1, 5184, 192]cuda:0" = PlaceHolder[target=relu_1]
#   %convolution_1 : Tensor "f32[64, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[64, 192, 27, 27][139968, 729, 27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 91373568}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2076672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 13)
    x2 = ((xindex // 2496) % 13)
    x3 = xindex // 32448
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp5 = tl.load(in_ptr0 + (5184 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp7 = tl.load(in_ptr0 + (5376 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp9 = tl.load(in_ptr0 + (5568 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp11 = tl.load(in_ptr0 + (10368 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp13 = tl.load(in_ptr0 + (10560 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp15 = tl.load(in_ptr0 + (10752 + x0 + 384*x1 + 10368*x2 + 139968*x3), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/xm/cxm74bmadiyf44sj46qtvb723njqdilhr6ivmi3bygti2y7m4ask.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_7 => convolution_2
# Graph fragment:
#   %arg5_1 : Tensor "f32[384, 192, 3, 3][1728, 9, 3, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf9
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 5308416, 'x': 2654208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/6w/c6wd43hm2kopvlxf4h5popyhkmub2dksl5irgjhs65nxqczol2bq.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_8 => relu_2
# Graph fragment:
#   %buf10 : Tensor "f32[64, 384, 13, 13][64896, 1, 4992, 384]cuda:0" = PlaceHolder[target=buf10]
#   %arg6_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   return %relu_2
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_poi_fused_convolution_relu_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 49841664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4153344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/mo/cmos5qq6wfnmsxkuu4qfqooobvfa6cuwp3bgrj4qs2eap7uz3g6u.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_8 => relu_2
#   input_9 => convolution_3
# Graph fragment:
#   %arg7_1 : Tensor "f32[256, 384, 3, 3][3456, 9, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convolution_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf12
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_poi_fused_convolution_relu_9', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 7077888, 'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 3456*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/wl/cwlquqdzl7mmjbylhh5cck7wq7px5wy4utfv2ckr5hqr6y74n75a.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_10 => relu_3
#   input_7 => convolution_2
#   input_8 => relu_2
#   input_9 => convolution_3
# Graph fragment:
#   %buf13 : Tensor "f32[64, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = PlaceHolder[target=buf13]
#   %arg8_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convolution_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   return %relu_3
triton_poi_fused_convolution_relu_10 = async_compile.triton('triton_poi_fused_convolution_relu_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33227776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2768896
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


# kernel path: /tmp/torchinductor_emre/q4/cq44fuadgmiqm2pgwx2zt6rtkrdxgpacqp2x3fi5aql6aoiujwam.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_10 => relu_3
#   input_11 => convolution_4
#   input_7 => convolution_2
#   input_8 => relu_2
#   input_9 => convolution_3
# Graph fragment:
#   %arg9_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convolution_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %convolution_4 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf15
triton_poi_fused_convolution_relu_11 = async_compile.triton('triton_poi_fused_convolution_relu_11', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_emre/a4/ca4ahocfxnlyasc42rgucmgmq6b4gbjrg7ag4v65s5hj34hge65l.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11, input_12, input_13, x], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   input_10 => relu_3
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => _low_memory_max_pool_with_offsets_2
#   input_7 => convolution_2
#   input_8 => relu_2
#   input_9 => convolution_3
#   x => _adaptive_avg_pool2d
# Graph fragment:
#   %relu_4 : Tensor "f32[64, 256, 13, 13][43264, 1, 3328, 256]cuda:0" = PlaceHolder[target=relu_4]
#   %getitem_4 : Tensor "f32[64, 256, 6, 6][9216, 1, 1536, 256]cuda:0" = PlaceHolder[target=getitem_4]
#   %convolution_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[64, 384, 13, 13][64896, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %convolution_4 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : Tensor "f32[64, 256, 13, 13][43264, 169, 13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %_adaptive_avg_pool2d : Tensor "f32[64, 256, 6, 6][9216, 36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%getitem_4, [6, 6]), kwargs = {})
#   return %getitem_4,%_adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 21233664, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_12(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 6)
    x3 = xindex // 6
    y0 = (yindex % 256)
    y1 = yindex // 256
    x4 = xindex
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (256 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3328 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (3584 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3840 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6656 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (6912 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (7168 + y0 + 512*x2 + 6656*x3 + 43264*y1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tl.store(out_ptr1 + (x4 + 36*y5), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/ia/ciayaawgvfpguv3p35pu34zm5jbczn3e7vxjcnkb2wkr2zoxmn4a.py
# Topologically Sorted Source Nodes: [, input_16], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_1
#   input_16 => relu_5
# Graph fragment:
#   %arg12_1 : Tensor "f32[4096][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %mm_default_1 : Tensor "f32[64, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[64, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default_1), kwargs = {})
#   %relu_5 : Tensor "f32[64, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   return %relu_5
triton_poi_fused_addmm_relu_13 = async_compile.triton('triton_poi_fused_addmm_relu_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3162112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 3, 11, 11), (363, 121, 11, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (64, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg3_1, (192, 64, 5, 5), (1600, 25, 5, 1))
        assert_size_stride(arg4_1, (192, ), (1, ))
        assert_size_stride(arg5_1, (384, 192, 3, 3), (1728, 9, 3, 1))
        assert_size_stride(arg6_1, (384, ), (1, ))
        assert_size_stride(arg7_1, (256, 384, 3, 3), (3456, 9, 3, 1))
        assert_size_stride(arg8_1, (256, ), (1, ))
        assert_size_stride(arg9_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg10_1, (256, ), (1, ))
        assert_size_stride(arg11_1, (4096, 9216), (9216, 1))
        assert_size_stride(arg12_1, (4096, ), (1, ))
        assert_size_stride(arg13_1, (4096, 4096), (4096, 1))
        assert_size_stride(arg14_1, (4096, ), (1, ))
        assert_size_stride(arg15_1, (1000, 4096), (4096, 1))
        assert_size_stride(arg16_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((64, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 192, 50176, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 3, 11, 11), (363, 1, 33, 3), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 121, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (64, 64, 55, 55), (193600, 1, 3520, 64), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_2.run(buf3, arg1_1, 12390400, stream=stream0)
            del arg1_1
            buf4 = empty_strided_cuda((64, 64, 27, 27), (46656, 1, 1728, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_3:5
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(buf3, buf4, 2985984, stream=stream0)
            del buf3
            buf5 = empty_strided_cuda((192, 64, 5, 5), (1600, 1, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:6
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(arg3_1, buf5, 12288, 25, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:7
            buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf6, (64, 192, 27, 27), (139968, 1, 5184, 192), 'torch.ops.aten.convolution.default')
            del buf4
            del buf5
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_5:8
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_5.run(buf7, arg4_1, 8957952, stream=stream0)
            del arg4_1
            buf8 = empty_strided_cuda((64, 192, 13, 13), (32448, 1, 2496, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_convolution_max_pool2d_with_indices_relu_6:9
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(buf7, buf8, 2076672, stream=stream0)
            del buf7
            buf9 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_7:10
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_7.run(arg5_1, buf9, 73728, 9, stream=stream0)
            del arg5_1
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:11
            buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf10, (64, 384, 13, 13), (64896, 1, 4992, 384), 'torch.ops.aten.convolution.default')
            del buf8
            del buf9
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_8:12
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_8.run(buf11, arg6_1, 4153344, stream=stream0)
            del arg6_1
            buf12 = empty_strided_cuda((256, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_9:13
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_9.run(arg7_1, buf12, 98304, 9, stream=stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:14
            buf13 = extern_kernels.convolution(buf11, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf13, (64, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf11
            del buf12
            buf14 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_10:15
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_10.run(buf14, arg8_1, 2768896, stream=stream0)
            del arg8_1
            buf15 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_11:16
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_11.run(arg9_1, buf15, 65536, 9, stream=stream0)
            del arg9_1
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] extern_kernels.convolution:17
            buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf16, (64, 256, 13, 13), (43264, 1, 3328, 256), 'torch.ops.aten.convolution.default')
            del buf14
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu]
            # [Provenance debug handles] triton_poi_fused_convolution_relu_10:18
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_10.run(buf17, arg10_1, 2768896, stream=stream0)
            del arg10_1
            buf19 = reinterpret_tensor(buf15, (64, 256, 6, 6), (9216, 36, 6, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11, input_12, input_13, x], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten._adaptive_avg_pool2d]
            # [Provenance debug handles] triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_12:19
            stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_12.run(buf17, buf19, 16384, 36, stream=stream0)
            del buf17
            buf20 = empty_strided_cuda((64, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, input_15, ], Original ATen: [aten._adaptive_avg_pool2d, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(reinterpret_tensor(buf19, (64, 9216), (9216, 1), 0), reinterpret_tensor(arg11_1, (9216, 4096), (1, 9216), 0), out=buf20)
            del arg11_1
            del buf19
            buf21 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [, input_16], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_13:21
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_13.run(buf21, arg12_1, 262144, stream=stream0)
            del arg12_1
            buf22 = empty_strided_cuda((64, 4096), (4096, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, input_16, input_18], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(buf21, reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf22)
            del arg13_1
            del buf21
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [, input_19], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_13:23
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_13.run(buf23, arg14_1, 262144, stream=stream0)
            del arg14_1
            buf24 = empty_strided_cuda((64, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, input_19, input_20], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.addmm:24
            extern_kernels.addmm(arg16_1, buf23, reinterpret_tensor(arg15_1, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf24)
            del arg15_1
            del arg16_1
            del buf23
        return (buf24, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 11, 11), (363, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((192, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((4096, 9216), (9216, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
