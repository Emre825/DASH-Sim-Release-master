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


# kernel path: /tmp/torchinductor_emre/i4/ci4gdkjl5sk4vegynuzfxcq7wzhhqucbotu62g53oxxxqfrl2v5o.py
# Topologically Sorted Source Nodes: [, x], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_8
#   x => relu
# Graph fragment:
#   %arg1_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_default_8 : Tensor "f32[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_default_8]
#   %add_tensor_8 : Tensor "f32[1, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mm_default_8), kwargs = {})
#   %relu : Tensor "f32[1, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_8,), kwargs = {})
#   return %relu
triton_poi_fused_addmm_relu_0 = async_compile.triton('triton_poi_fused_addmm_relu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_emre/wr/cwrwyzwyofag3qpp67g3s7gtudyqklsew5xmizrwr2fpq2a55jqz.py
# Topologically Sorted Source Nodes: [, x_4], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_4
#   x_4 => relu_4
# Graph fragment:
#   %arg10_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %mm_default_4 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_4]
#   %add_tensor_4 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg10_1, %mm_default_4), kwargs = {})
#   %relu_4 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_4,), kwargs = {})
#   return %relu_4
triton_poi_fused_addmm_relu_1 = async_compile.triton('triton_poi_fused_addmm_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=14, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'CBAE4A4240C74796195FD74D4D83ACD1B2BA7C4CDDF4C840D2664C75DC2B8481', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg1_1, (2048, ), (1, ))
        assert_size_stride(arg2_1, (1, 2048), (2048, 1))
        assert_size_stride(arg3_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg4_1, (2048, ), (1, ))
        assert_size_stride(arg5_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg6_1, (2048, ), (1, ))
        assert_size_stride(arg7_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg8_1, (2048, ), (1, ))
        assert_size_stride(arg9_1, (256, 2048), (2048, 1))
        assert_size_stride(arg10_1, (256, ), (1, ))
        assert_size_stride(arg11_1, (2048, 256), (256, 1))
        assert_size_stride(arg12_1, (2048, ), (1, ))
        assert_size_stride(arg13_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg14_1, (2048, ), (1, ))
        assert_size_stride(arg15_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg16_1, (2048, ), (1, ))
        assert_size_stride(arg17_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg18_1, (2048, ), (1, ))
        assert_size_stride(arg19_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg20_1, (2048, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (2048, 2048), (1, 2048), 0), out=buf0)
            del arg0_1
            del arg2_1
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [, x], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf1, arg1_1, 2048, stream=stream0)
            del arg1_1
            buf2 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x, linear_1], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, (2048, 2048), (1, 2048), 0), out=buf2)
            del arg3_1
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [, x_1], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf3, arg4_1, 2048, stream=stream0)
            del arg4_1
            buf4 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_1, linear_2], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(buf3, reinterpret_tensor(arg5_1, (2048, 2048), (1, 2048), 0), out=buf4)
            del arg5_1
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [, x_2], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:6
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf5, arg6_1, 2048, stream=stream0)
            del arg6_1
            buf6 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_2, linear_3], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(buf5, reinterpret_tensor(arg7_1, (2048, 2048), (1, 2048), 0), out=buf6)
            del arg7_1
            del buf5
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [, x_3], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:8
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf7, arg8_1, 2048, stream=stream0)
            del arg8_1
            buf8 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_3, linear_4], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(buf7, reinterpret_tensor(arg9_1, (2048, 256), (1, 2048), 0), out=buf8)
            del arg9_1
            buf9 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [, x_4], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_1:10
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_1.run(buf9, arg10_1, 256, stream=stream0)
            del arg10_1
            buf10 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [, x_4, linear_5], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:11
            extern_kernels.mm(buf9, reinterpret_tensor(arg11_1, (256, 2048), (1, 256), 0), out=buf10)
            del arg11_1
            del buf9
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [, x_5], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:12
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf11, arg12_1, 2048, stream=stream0)
            del arg12_1
            buf12 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_5, linear_6], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:13
            extern_kernels.mm(buf11, reinterpret_tensor(arg13_1, (2048, 2048), (1, 2048), 0), out=buf12)
            del arg13_1
            del buf11
            buf13 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [, x_6], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:14
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf13, arg14_1, 2048, stream=stream0)
            del arg14_1
            buf14 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_6, linear_7], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:15
            extern_kernels.mm(buf13, reinterpret_tensor(arg15_1, (2048, 2048), (1, 2048), 0), out=buf14)
            del arg15_1
            del buf13
            buf15 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [, x_7], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:16
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf15, arg16_1, 2048, stream=stream0)
            del arg16_1
            buf16 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_7, linear_8], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:17
            extern_kernels.mm(buf15, reinterpret_tensor(arg17_1, (2048, 2048), (1, 2048), 0), out=buf16)
            del arg17_1
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [, x_8], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_0:18
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf17, arg18_1, 2048, stream=stream0)
            del arg18_1
            buf18 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_8, x_9], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.addmm:19
            extern_kernels.addmm(arg20_1, buf17, reinterpret_tensor(arg19_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf18)
            del arg19_1
            del arg20_1
            del buf17
        return (buf18, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
