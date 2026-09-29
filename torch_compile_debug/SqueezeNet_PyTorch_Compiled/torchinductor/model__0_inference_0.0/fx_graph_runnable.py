
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1):
        convolution = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu = None
        getitem = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_1 = torch.ops.aten.convolution.default(getitem, arg3_1, arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg3_1 = arg4_1 = None
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, arg5_1, arg6_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg5_1 = arg6_1 = None
        relu_2 = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_1, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg7_1 = arg8_1 = None
        relu_3 = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        cat = torch.ops.aten.cat.default([relu_2, relu_3], 1);  relu_2 = relu_3 = None
        convolution_4 = torch.ops.aten.convolution.default(cat, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg9_1 = arg10_1 = None
        relu_4 = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = arg12_1 = None
        relu_5 = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_4, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg13_1 = arg14_1 = None
        relu_6 = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        cat_1 = torch.ops.aten.cat.default([relu_5, relu_6], 1);  relu_5 = relu_6 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_1 = None
        getitem_2 = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_7 = torch.ops.aten.convolution.default(getitem_2, arg15_1, arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg15_1 = arg16_1 = None
        relu_7 = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_7, arg17_1, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg17_1 = arg18_1 = None
        relu_8 = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg19_1 = arg20_1 = None
        relu_9 = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        cat_2 = torch.ops.aten.cat.default([relu_8, relu_9], 1);  relu_8 = relu_9 = None
        convolution_10 = torch.ops.aten.convolution.default(cat_2, arg21_1, arg22_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg21_1 = arg22_1 = None
        relu_10 = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg23_1 = arg24_1 = None
        relu_11 = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_10, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg25_1 = arg26_1 = None
        relu_12 = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        cat_3 = torch.ops.aten.cat.default([relu_11, relu_12], 1);  relu_11 = relu_12 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_3 = None
        getitem_4 = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        convolution_13 = torch.ops.aten.convolution.default(getitem_4, arg27_1, arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_4 = arg27_1 = arg28_1 = None
        relu_13 = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_13, arg29_1, arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg29_1 = arg30_1 = None
        relu_14 = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_13, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg31_1 = arg32_1 = None
        relu_15 = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
        cat_4 = torch.ops.aten.cat.default([relu_14, relu_15], 1);  relu_14 = relu_15 = None
        convolution_16 = torch.ops.aten.convolution.default(cat_4, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg33_1 = arg34_1 = None
        relu_16 = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_16, arg35_1, arg36_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg35_1 = arg36_1 = None
        relu_17 = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_16, arg37_1, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg37_1 = arg38_1 = None
        relu_18 = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
        cat_5 = torch.ops.aten.cat.default([relu_17, relu_18], 1);  relu_17 = relu_18 = None
        convolution_19 = torch.ops.aten.convolution.default(cat_5, arg39_1, arg40_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg39_1 = arg40_1 = None
        relu_19 = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_19, arg41_1, arg42_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = arg42_1 = None
        relu_20 = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_19, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = arg43_1 = arg44_1 = None
        relu_21 = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
        cat_6 = torch.ops.aten.cat.default([relu_20, relu_21], 1);  relu_20 = relu_21 = None
        convolution_22 = torch.ops.aten.convolution.default(cat_6, arg45_1, arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg45_1 = arg46_1 = None
        relu_22 = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_22, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg47_1 = arg48_1 = None
        relu_23 = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_22, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg49_1 = arg50_1 = None
        relu_24 = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
        cat_7 = torch.ops.aten.cat.default([relu_23, relu_24], 1);  relu_23 = relu_24 = None
        convolution_25 = torch.ops.aten.convolution.default(cat_7, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg51_1 = arg52_1 = None
        relu_25 = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean = torch.ops.aten.mean.dim(relu_25, [-1, -2], True);  relu_25 = None
        view = torch.ops.aten.view.default(mean, [1, 1000]);  mean = None
        return (view,)
        
def load_args(reader):
    buf0 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 3, 224, 224), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16, 64, 1, 1), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64, 16, 1, 1), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64, 16, 3, 3), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf9, (16, 128, 1, 1), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf10, (16,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 16, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64, 16, 3, 3), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf15, (32, 128, 1, 1), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf16, (32,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128, 32, 1, 1), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 32, 3, 3), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf21, (32, 256, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf22, (32,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128, 32, 1, 1), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128, 32, 3, 3), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf27, (48, 256, 1, 1), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf28, (48,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf29, (192, 48, 1, 1), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf30, (192,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf31, (192, 48, 3, 3), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf32, (192,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf33, (48, 384, 1, 1), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf34, (48,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf35, (192, 48, 1, 1), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf36, (192,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf37, (192, 48, 3, 3), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf38, (192,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64, 384, 1, 1), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 64, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 64, 3, 3), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64, 512, 1, 1), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256, 64, 1, 1), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256, 64, 3, 3), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (256,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1000, 512, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1000,), is_leaf=True)  # arg52_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)