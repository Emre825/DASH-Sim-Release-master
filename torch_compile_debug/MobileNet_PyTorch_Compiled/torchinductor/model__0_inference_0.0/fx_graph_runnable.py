
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1):
        convolution = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
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
        convolution_1 = torch.ops.aten.convolution.default(relu, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  relu = arg6_1 = None
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
        convolution_2 = torch.ops.aten.convolution.default(relu_1, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg11_1 = None
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
        relu_2 = torch.ops.aten.relu.default(add_5);  add_5 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  relu_2 = arg16_1 = None
        add_6 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_6);  add_6 = None
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
        add_7 = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        relu_3 = torch.ops.aten.relu.default(add_7);  add_7 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_3 = arg21_1 = None
        add_8 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_8);  add_8 = None
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
        add_9 = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        relu_4 = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  relu_4 = arg26_1 = None
        add_10 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_10);  add_10 = None
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
        add_11 = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        relu_5 = torch.ops.aten.relu.default(add_11);  add_11 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg31_1 = None
        add_12 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_12);  add_12 = None
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
        add_13 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        relu_6 = torch.ops.aten.relu.default(add_13);  add_13 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_6, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32);  relu_6 = arg36_1 = None
        add_14 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_7 = torch.ops.aten.sqrt.default(add_14);  add_14 = None
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
        add_15 = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        relu_7 = torch.ops.aten.relu.default(add_15);  add_15 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_7, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg41_1 = None
        add_16 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_16);  add_16 = None
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
        add_17 = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        relu_8 = torch.ops.aten.relu.default(add_17);  add_17 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_8, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  relu_8 = arg46_1 = None
        add_18 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_9 = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27 = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        sub_9 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_19 = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        relu_9 = torch.ops.aten.relu.default(add_19);  add_19 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_9, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg51_1 = None
        add_20 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_10 = torch.ops.aten.sqrt.default(add_20);  add_20 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        sub_10 = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_21 = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
        relu_10 = torch.ops.aten.relu.default(add_21);  add_21 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_10, arg56_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_10 = arg56_1 = None
        add_22 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_11 = torch.ops.aten.sqrt.default(add_22);  add_22 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        sub_11 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_23 = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
        relu_11 = torch.ops.aten.relu.default(add_23);  add_23 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_11, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg61_1 = None
        add_24 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_12 = torch.ops.aten.sqrt.default(add_24);  add_24 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_25 = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
        relu_12 = torch.ops.aten.relu.default(add_25);  add_25 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_12, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_12 = arg66_1 = None
        add_26 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_13 = torch.ops.aten.sqrt.default(add_26);  add_26 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_39 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        sub_13 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_27 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
        relu_13 = torch.ops.aten.relu.default(add_27);  add_27 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_13, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg71_1 = None
        add_28 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_14 = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_42 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        sub_14 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_29 = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
        relu_14 = torch.ops.aten.relu.default(add_29);  add_29 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_14, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_14 = arg76_1 = None
        add_30 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_15 = torch.ops.aten.sqrt.default(add_30);  add_30 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_45 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        sub_15 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_31 = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
        relu_15 = torch.ops.aten.relu.default(add_31);  add_31 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_15, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg81_1 = None
        add_32 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_16 = torch.ops.aten.sqrt.default(add_32);  add_32 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_48 = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_33 = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
        relu_16 = torch.ops.aten.relu.default(add_33);  add_33 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_16, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_16 = arg86_1 = None
        add_34 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_17 = torch.ops.aten.sqrt.default(add_34);  add_34 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_51 = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        sub_17 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_35 = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
        relu_17 = torch.ops.aten.relu.default(add_35);  add_35 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_17, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg91_1 = None
        add_36 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_18 = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_54 = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        sub_18 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_37 = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
        relu_18 = torch.ops.aten.relu.default(add_37);  add_37 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_18, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_18 = arg96_1 = None
        add_38 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_19 = torch.ops.aten.sqrt.default(add_38);  add_38 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_57 = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        sub_19 = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_39 = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
        relu_19 = torch.ops.aten.relu.default(add_39);  add_39 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_19, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg101_1 = None
        add_40 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_20 = torch.ops.aten.sqrt.default(add_40);  add_40 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_60 = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        sub_20 = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_41 = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
        relu_20 = torch.ops.aten.relu.default(add_41);  add_41 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_20, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_20 = arg106_1 = None
        add_42 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_21 = torch.ops.aten.sqrt.default(add_42);  add_42 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_63 = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        sub_21 = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_43 = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
        relu_21 = torch.ops.aten.relu.default(add_43);  add_43 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_21, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg111_1 = None
        add_44 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_22 = torch.ops.aten.sqrt.default(add_44);  add_44 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_66 = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        sub_22 = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_68 = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_45 = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
        relu_22 = torch.ops.aten.relu.default(add_45);  add_45 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_22, arg116_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128);  relu_22 = arg116_1 = None
        add_46 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_46);  add_46 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_69 = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        sub_23 = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_47 = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
        relu_23 = torch.ops.aten.relu.default(add_47);  add_47 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_23, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg121_1 = None
        add_48 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_48);  add_48 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_72 = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        sub_24 = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_49 = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
        relu_24 = torch.ops.aten.relu.default(add_49);  add_49 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_24, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  relu_24 = arg126_1 = None
        add_50 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_25 = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_75 = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        sub_25 = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_51 = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
        relu_25 = torch.ops.aten.relu.default(add_51);  add_51 = None
        convolution_26 = torch.ops.aten.convolution.default(relu_25, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_25 = arg131_1 = None
        add_52 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_26 = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_78 = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        sub_26 = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_53 = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
        relu_26 = torch.ops.aten.relu.default(add_53);  add_53 = None
        mean = torch.ops.aten.mean.dim(relu_26, [-1, -2], True);  relu_26 = None
        view = torch.ops.aten.view.default(mean, [1, 256]);  mean = None
        permute = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm = torch.ops.aten.addmm.default(arg137_1, view, permute);  arg137_1 = view = permute = None
        return (addmm,)
        
def load_args(reader):
    buf0 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 3, 96, 96), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf2, (8,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf3, (8,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf4, (8,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf5, (8,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf6, (8, 1, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf7, (8,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf8, (8,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf9, (8,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf10, (8,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 8, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf16, (16, 1, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf17, (16,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf18, (16,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf19, (16,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf20, (16,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf21, (32, 16, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf22, (32,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf23, (32,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf24, (32,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf25, (32,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf26, (32, 1, 3, 3), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf29, (32,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf30, (32,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf31, (32, 32, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf32, (32,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf33, (32,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf34, (32,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf35, (32,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf36, (32, 1, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf37, (32,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf38, (32,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf39, (32,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf40, (32,), is_leaf=True)  # arg40_1
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
    buf46 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64, 1, 3, 3), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf51, (64, 64, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf52, (64,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf54, (64,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf55, (64,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64, 1, 3, 3), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf57, (64,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf58, (64,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf59, (64,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf60, (64,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 64, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf66, (128, 1, 3, 3), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf71, (128, 128, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128, 1, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128, 128, 1, 1), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128, 1, 3, 3), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf87, (128,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf88, (128,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128, 128, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf95, (128,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128, 1, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf99, (128,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf100, (128,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf101, (128, 128, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf103, (128,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128, 1, 3, 3), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf109, (128,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (128, 128, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf112, (128,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128, 1, 3, 3), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf117, (128,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf118, (128,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf119, (128,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 128, 1, 1), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256, 1, 3, 3), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf128, (256,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf131, (256, 256, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf132, (256,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (256,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf136, (2, 256), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf137, (2,), is_leaf=True)  # arg137_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)