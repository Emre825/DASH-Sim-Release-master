
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1):
        convolution = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        sub = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        relu = torch.ops.aten.relu.default(add_1);  add_1 = None
        convolution_1 = torch.ops.aten.convolution.default(relu, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg6_1 = None
        add_2 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_1 = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3 = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        relu_1 = torch.ops.aten.relu.default(add_3);  add_3 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg11_1 = None
        add_4 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_2 = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5 = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        add_6 = torch.ops.aten.add.Tensor(add_5, relu);  add_5 = relu = None
        relu_2 = torch.ops.aten.relu.default(add_6);  add_6 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg16_1 = None
        add_7 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_8 = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        relu_3 = torch.ops.aten.relu.default(add_8);  add_8 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg21_1 = None
        add_9 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_10 = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_2, arg26_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg26_1 = None
        add_11 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_11);  add_11 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_12 = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        add_13 = torch.ops.aten.add.Tensor(add_10, add_12);  add_10 = add_12 = None
        relu_4 = torch.ops.aten.relu.default(add_13);  add_13 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_4, arg31_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg31_1 = None
        add_14 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18 = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        sub_6 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_15 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        relu_5 = torch.ops.aten.relu.default(add_15);  add_15 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_5, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg36_1 = None
        add_16 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_7 = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21 = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        sub_7 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_17 = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_4, arg41_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg41_1 = None
        add_18 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_19 = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        add_20 = torch.ops.aten.add.Tensor(add_17, add_19);  add_17 = add_19 = None
        relu_6 = torch.ops.aten.relu.default(add_20);  add_20 = None
        mean = torch.ops.aten.mean.dim(relu_6, [-1, -2], True);  relu_6 = None
        view = torch.ops.aten.view.default(mean, [1, -1]);  mean = None
        permute = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm = torch.ops.aten.addmm.default(arg47_1, view, permute);  arg47_1 = view = permute = None
        return (addmm,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 3, 32, 32), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf2, (16,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf6, (16, 16, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf7, (16,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf8, (16,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf9, (16,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf10, (16,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 16, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf16, (32, 16, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf17, (32,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf18, (32,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf19, (32,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf20, (32,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf21, (32, 32, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf22, (32,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf23, (32,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf24, (32,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf25, (32,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (32, 16, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf29, (32,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf30, (32,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 32, 3, 3), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf33, (64,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf34, (64,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf36, (64, 64, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64, 32, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf46, (10, 64), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf47, (10,), is_leaf=True)  # arg47_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)