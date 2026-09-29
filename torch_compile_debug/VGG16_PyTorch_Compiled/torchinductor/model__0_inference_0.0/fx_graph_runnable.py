
import os
os.environ['TORCH_LOGS'] = '+output_code,+graph_code'
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1):
        convolution = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1 = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_2 = torch.ops.aten.convolution.default(getitem, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg5_1 = arg6_1 = None
        relu_2 = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3 = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_3 = None
        getitem_2 = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem_2, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg9_1 = arg10_1 = None
        relu_4 = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = arg12_1 = None
        relu_5 = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg13_1 = arg14_1 = None
        relu_6 = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_6 = None
        getitem_4 = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        convolution_7 = torch.ops.aten.convolution.default(getitem_4, arg15_1, arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg15_1 = arg16_1 = None
        relu_7 = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_7, arg17_1, arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg17_1 = arg18_1 = None
        relu_8 = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_8, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg19_1 = arg20_1 = None
        relu_9 = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_9 = None
        getitem_6 = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None
        convolution_10 = torch.ops.aten.convolution.default(getitem_6, arg21_1, arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg21_1 = arg22_1 = None
        relu_10 = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg23_1 = arg24_1 = None
        relu_11 = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_11, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg25_1 = arg26_1 = None
        relu_12 = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_4 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_12 = None
        getitem_8 = _low_memory_max_pool_with_offsets_4[0];  _low_memory_max_pool_with_offsets_4 = None
        _adaptive_avg_pool2d = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7]);  getitem_8 = None
        view = torch.ops.aten.view.default(_adaptive_avg_pool2d, [1, 25088]);  _adaptive_avg_pool2d = None
        permute = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm = torch.ops.aten.addmm.default(arg28_1, view, permute);  arg28_1 = view = permute = None
        relu_13 = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg30_1, relu_13, permute_1);  arg30_1 = relu_13 = permute_1 = None
        relu_14 = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg32_1, relu_14, permute_2);  arg32_1 = relu_14 = permute_2 = None
        return (addmm_2,)
        
def load_args(reader):
    buf0 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 3, 224, 224), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64, 64, 3, 3), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128, 64, 3, 3), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128, 128, 3, 3), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf9, (256, 128, 3, 3), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256, 256, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256, 256, 3, 3), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512, 256, 3, 3), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512, 512, 3, 3), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512, 512, 3, 3), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512, 512, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512, 512, 3, 3), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512, 512, 3, 3), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 411041792, device=device(type='cuda', index=0))
    reader.tensor(buf27, (4096, 25088), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf28, (4096,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf29, (4096, 4096), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf30, (4096,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 16384000, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1000, 4096), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1000,), is_leaf=True)  # arg32_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)