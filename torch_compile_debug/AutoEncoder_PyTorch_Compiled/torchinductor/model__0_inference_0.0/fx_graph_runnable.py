
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg4_1, relu, permute_1);  arg4_1 = relu = permute_1 = None
        relu_1 = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg6_1, relu_1, permute_2);  arg6_1 = relu_1 = permute_2 = None
        relu_2 = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None
        permute_3 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg8_1, relu_2, permute_3);  arg8_1 = relu_2 = permute_3 = None
        relu_3 = torch.ops.aten.relu.default(addmm_3);  addmm_3 = None
        permute_4 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg10_1, relu_3, permute_4);  arg10_1 = relu_3 = permute_4 = None
        relu_4 = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        permute_5 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg12_1, relu_4, permute_5);  arg12_1 = relu_4 = permute_5 = None
        relu_5 = torch.ops.aten.relu.default(addmm_5);  addmm_5 = None
        permute_6 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg14_1, relu_5, permute_6);  arg14_1 = relu_5 = permute_6 = None
        relu_6 = torch.ops.aten.relu.default(addmm_6);  addmm_6 = None
        permute_7 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg16_1, relu_6, permute_7);  arg16_1 = relu_6 = permute_7 = None
        relu_7 = torch.ops.aten.relu.default(addmm_7);  addmm_7 = None
        permute_8 = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg18_1, relu_7, permute_8);  arg18_1 = relu_7 = permute_8 = None
        relu_8 = torch.ops.aten.relu.default(addmm_8);  addmm_8 = None
        permute_9 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg20_1, relu_8, permute_9);  arg20_1 = relu_8 = permute_9 = None
        return (addmm_9,)
        
def load_args(reader):
    buf0 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2048, 2048), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2048,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 2048), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf3, (2048, 2048), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2048,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2048, 2048), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2048,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2048, 2048), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2048,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf9, (256, 2048), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf11, (2048, 256), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf12, (2048,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf13, (2048, 2048), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf14, (2048,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf15, (2048, 2048), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf16, (2048,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (2048, 2048), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf18, (2048,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2048, 2048), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf20, (2048,), is_leaf=True)  # arg20_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)