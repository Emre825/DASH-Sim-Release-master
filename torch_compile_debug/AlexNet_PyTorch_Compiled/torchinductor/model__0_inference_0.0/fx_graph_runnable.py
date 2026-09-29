
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_emre'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cu128
# torch cuda version: 12.8
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Aug_20_01:58:59_PM_PDT_2025 
# Cuda compilation tools, release 13.0, V13.0.88 
# Build cuda_13.0.r13.0/compiler.36424714_0 

# GPU Hardware Info: 
# NVIDIA GeForce GTX 1650 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1):
        convolution = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu = None
        getitem = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_1 = torch.ops.aten.convolution.default(getitem, arg3_1, arg4_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  getitem = arg3_1 = arg4_1 = None
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem_2 = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_2 = torch.ops.aten.convolution.default(getitem_2, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg5_1 = arg6_1 = None
        relu_2 = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3 = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg9_1 = arg10_1 = None
        relu_4 = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_4 = None
        getitem_4 = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        _adaptive_avg_pool2d = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6]);  getitem_4 = None
        view = torch.ops.aten.view.default(_adaptive_avg_pool2d, [64, 9216]);  _adaptive_avg_pool2d = None
        permute = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm = torch.ops.aten.addmm.default(arg12_1, view, permute);  arg12_1 = view = permute = None
        relu_5 = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg14_1, relu_5, permute_1);  arg14_1 = relu_5 = permute_1 = None
        relu_6 = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg16_1, relu_6, permute_2);  arg16_1 = relu_6 = permute_2 = None
        return (addmm_2,)
        
def load_args(reader):
    buf0 = reader.storage(None, 92928, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 11, 11), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 38535168, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64, 3, 224, 224), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf3, (192, 64, 5, 5), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf4, (192,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2654208, device=device(type='cuda', index=0))
    reader.tensor(buf5, (384, 192, 3, 3), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf7, (256, 384, 3, 3), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf8, (256,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (256, 256, 3, 3), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 150994944, device=device(type='cuda', index=0))
    reader.tensor(buf11, (4096, 9216), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf12, (4096,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf13, (4096, 4096), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4096,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16384000, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1000, 4096), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1000,), is_leaf=True)  # arg16_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)