
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1):
        convolution = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1 = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_2 = torch.ops.aten.convolution.default(getitem, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg5_1 = arg6_1 = None
        relu_2 = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3 = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2 = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem_2, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg9_1 = arg10_1 = None
        relu_4 = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = arg12_1 = None
        relu_5 = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4 = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        convolution_6 = torch.ops.aten.convolution.default(getitem_4, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg13_1 = arg14_1 = None
        relu_6 = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_6, arg15_1, arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_6 = arg15_1 = arg16_1 = None
        relu_7 = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6 = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None
        convolution_8 = torch.ops.aten.convolution.default(getitem_6, arg17_1, arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg17_1 = arg18_1 = None
        relu_8 = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_8, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg19_1 = arg20_1 = None
        relu_9 = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        add_1 = torch.ops.aten.add.Tensor(convert_element_type, 0.0);  convert_element_type = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1);  convert_element_type_1 = None
        iota_1 = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2 = torch.ops.aten.mul.Tensor(iota_1, 1);  iota_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        add_3 = torch.ops.aten.add.Tensor(convert_element_type_2, 0.0);  convert_element_type_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, unsqueeze, convert_element_type_3]);  relu_9 = unsqueeze = convert_element_type_3 = None
        cat = torch.ops.aten.cat.default([_unsafe_index, relu_7], 1);  _unsafe_index = relu_7 = None
        convolution_10 = torch.ops.aten.convolution.default(cat, arg21_1, arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat = arg21_1 = arg22_1 = None
        relu_10 = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg23_1 = arg24_1 = None
        relu_11 = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        iota_2 = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_4 = torch.ops.aten.mul.Tensor(iota_2, 1);  iota_2 = None
        add_4 = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        add_5 = torch.ops.aten.add.Tensor(convert_element_type_4, 0.0);  convert_element_type_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1);  convert_element_type_5 = None
        iota_3 = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_6 = torch.ops.aten.mul.Tensor(iota_3, 1);  iota_3 = None
        add_6 = torch.ops.aten.add.Tensor(mul_6, 0);  mul_6 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        add_7 = torch.ops.aten.add.Tensor(convert_element_type_6, 0.0);  convert_element_type_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_7, torch.int64);  mul_7 = None
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, unsqueeze_1, convert_element_type_7]);  relu_11 = unsqueeze_1 = convert_element_type_7 = None
        cat_1 = torch.ops.aten.cat.default([_unsafe_index_1, relu_5], 1);  _unsafe_index_1 = relu_5 = None
        convolution_12 = torch.ops.aten.convolution.default(cat_1, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_1 = arg25_1 = arg26_1 = None
        relu_12 = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_12, arg27_1, arg28_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_12 = arg27_1 = arg28_1 = None
        relu_13 = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
        iota_4 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_8 = torch.ops.aten.mul.Tensor(iota_4, 1);  iota_4 = None
        add_8 = torch.ops.aten.add.Tensor(mul_8, 0);  mul_8 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        add_9 = torch.ops.aten.add.Tensor(convert_element_type_8, 0.0);  convert_element_type_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_9, torch.int64);  mul_9 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1);  convert_element_type_9 = None
        iota_5 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_10 = torch.ops.aten.mul.Tensor(iota_5, 1);  iota_5 = None
        add_10 = torch.ops.aten.add.Tensor(mul_10, 0);  mul_10 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        add_11 = torch.ops.aten.add.Tensor(convert_element_type_10, 0.0);  convert_element_type_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(mul_11, torch.int64);  mul_11 = None
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, unsqueeze_2, convert_element_type_11]);  relu_13 = unsqueeze_2 = convert_element_type_11 = None
        cat_2 = torch.ops.aten.cat.default([_unsafe_index_2, relu_3], 1);  _unsafe_index_2 = relu_3 = None
        convolution_14 = torch.ops.aten.convolution.default(cat_2, arg29_1, arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_2 = arg29_1 = arg30_1 = None
        relu_14 = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_14, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_14 = arg31_1 = arg32_1 = None
        relu_15 = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
        iota_6 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_12 = torch.ops.aten.mul.Tensor(iota_6, 1);  iota_6 = None
        add_12 = torch.ops.aten.add.Tensor(mul_12, 0);  mul_12 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        add_13 = torch.ops.aten.add.Tensor(convert_element_type_12, 0.0);  convert_element_type_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(mul_13, torch.int64);  mul_13 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(convert_element_type_13, -1);  convert_element_type_13 = None
        iota_7 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_14 = torch.ops.aten.mul.Tensor(iota_7, 1);  iota_7 = None
        add_14 = torch.ops.aten.add.Tensor(mul_14, 0);  mul_14 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        add_15 = torch.ops.aten.add.Tensor(convert_element_type_14, 0.0);  convert_element_type_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_15, torch.int64);  mul_15 = None
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, unsqueeze_3, convert_element_type_15]);  relu_15 = unsqueeze_3 = convert_element_type_15 = None
        cat_3 = torch.ops.aten.cat.default([_unsafe_index_3, relu_1], 1);  _unsafe_index_3 = relu_1 = None
        convolution_16 = torch.ops.aten.convolution.default(cat_3, arg33_1, arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_3 = arg33_1 = arg34_1 = None
        relu_16 = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_16, arg35_1, arg36_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg35_1 = arg36_1 = None
        relu_17 = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_17, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg37_1 = arg38_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        return (sigmoid,)
        
def load_args(reader):
    buf0 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 3, 256, 256), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf3, (8, 8, 3, 3), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf4, (8,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16, 8, 3, 3), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf6, (16,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf7, (16, 16, 3, 3), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf8, (16,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf9, (32, 16, 3, 3), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf10, (32,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf11, (32, 32, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64, 32, 3, 3), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64, 64, 3, 3), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128, 64, 3, 3), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 128, 3, 3), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 192, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64, 64, 3, 3), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf25, (32, 96, 3, 3), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf26, (32,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32, 32, 3, 3), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf29, (16, 48, 3, 3), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf30, (16,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf31, (16, 16, 3, 3), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf32, (16,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf33, (8, 24, 3, 3), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf34, (8,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf35, (8, 8, 3, 3), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf36, (8,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1, 8, 1, 1), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1,), is_leaf=True)  # arg38_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)