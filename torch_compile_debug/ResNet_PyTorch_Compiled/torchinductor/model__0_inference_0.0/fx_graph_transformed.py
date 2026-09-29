class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[1, 3, 32, 32]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[16, 16, 3, 3]", arg7_1: "f32[16]", arg8_1: "f32[16]", arg9_1: "f32[16]", arg10_1: "f32[16]", arg11_1: "f32[16, 16, 3, 3]", arg12_1: "f32[16]", arg13_1: "f32[16]", arg14_1: "f32[16]", arg15_1: "f32[16]", arg16_1: "f32[32, 16, 3, 3]", arg17_1: "f32[32]", arg18_1: "f32[32]", arg19_1: "f32[32]", arg20_1: "f32[32]", arg21_1: "f32[32, 32, 3, 3]", arg22_1: "f32[32]", arg23_1: "f32[32]", arg24_1: "f32[32]", arg25_1: "f32[32]", arg26_1: "f32[32, 16, 1, 1]", arg27_1: "f32[32]", arg28_1: "f32[32]", arg29_1: "f32[32]", arg30_1: "f32[32]", arg31_1: "f32[64, 32, 3, 3]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[64, 64, 3, 3]", arg37_1: "f32[64]", arg38_1: "f32[64]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64, 32, 1, 1]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64]", arg46_1: "f32[10, 64]", arg47_1: "f32[10]"):
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:55 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convolution: "f32[1, 16, 32, 32]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[1, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        add: "f32[16]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[1, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[1, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[1, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        relu: "f32[1, 16, 32, 32]" = torch.ops.aten.relu.default(add_1);  add_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:22 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convolution_1: "f32[1, 16, 32, 32]" = torch.ops.aten.convolution.default(relu, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg6_1 = None
        unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[1, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        add_2: "f32[16]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_1: "f32[16]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[1, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[1, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[1, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        relu_1: "f32[1, 16, 32, 32]" = torch.ops.aten.relu.default(add_3);  add_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:23 in forward, code: out = self.bn2(self.conv2(out))
        convolution_2: "f32[1, 16, 32, 32]" = torch.ops.aten.convolution.default(relu_1, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg11_1 = None
        unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[1, 16, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        add_4: "f32[16]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[1, 16, 32, 32]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[1, 16, 32, 32]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5: "f32[1, 16, 32, 32]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:24 in forward, code: out += self.shortcut(x) # returns only x if stride = 1 and in_channels = out_channels, returns BN2D(Conv2D(x)) otherwise
        add_6: "f32[1, 16, 32, 32]" = torch.ops.aten.add.Tensor(add_5, relu);  add_5 = relu = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:25 in forward, code: out = F.relu(out)
        relu_2: "f32[1, 16, 32, 32]" = torch.ops.aten.relu.default(add_6);  add_6 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:22 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convolution_3: "f32[1, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_2, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg16_1 = None
        unsqueeze_24: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_25: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[1, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        add_7: "f32[32]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_3: "f32[32]" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_3: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[1, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_29: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[1, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_31: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_8: "f32[1, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        relu_3: "f32[1, 32, 16, 16]" = torch.ops.aten.relu.default(add_8);  add_8 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:23 in forward, code: out = self.bn2(self.conv2(out))
        convolution_4: "f32[1, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_3, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg21_1 = None
        unsqueeze_32: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_33: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[1, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        add_9: "f32[32]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_4: "f32[32]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        reciprocal_4: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[1, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_37: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[1, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_39: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_10: "f32[1, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:24 in forward, code: out += self.shortcut(x) # returns only x if stride = 1 and in_channels = out_channels, returns BN2D(Conv2D(x)) otherwise
        convolution_5: "f32[1, 32, 16, 16]" = torch.ops.aten.convolution.default(relu_2, arg26_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg26_1 = None
        unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[1, 32, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        add_11: "f32[32]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_5: "f32[32]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
        reciprocal_5: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[1, 32, 16, 16]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[1, 32, 16, 16]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_12: "f32[1, 32, 16, 16]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        add_13: "f32[1, 32, 16, 16]" = torch.ops.aten.add.Tensor(add_10, add_12);  add_10 = add_12 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:25 in forward, code: out = F.relu(out)
        relu_4: "f32[1, 32, 16, 16]" = torch.ops.aten.relu.default(add_13);  add_13 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:22 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convolution_6: "f32[1, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_4, arg31_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg31_1 = None
        unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[1, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        add_14: "f32[64]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[1, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[1, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_15: "f32[1, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        relu_5: "f32[1, 64, 8, 8]" = torch.ops.aten.relu.default(add_15);  add_15 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:23 in forward, code: out = self.bn2(self.conv2(out))
        convolution_7: "f32[1, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_5, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg36_1 = None
        unsqueeze_56: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_57: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[1, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        add_16: "f32[64]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_7: "f32[64]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_7: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[1, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_61: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[1, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_63: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_17: "f32[1, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:24 in forward, code: out += self.shortcut(x) # returns only x if stride = 1 and in_channels = out_channels, returns BN2D(Conv2D(x)) otherwise
        convolution_8: "f32[1, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_4, arg41_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg41_1 = None
        unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[1, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        add_18: "f32[64]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_8: "f32[64]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_8: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[1, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[1, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_19: "f32[1, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        add_20: "f32[1, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_17, add_19);  add_17 = add_19 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:25 in forward, code: out = F.relu(out)
        relu_6: "f32[1, 64, 8, 8]" = torch.ops.aten.relu.default(add_20);  add_20 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:59 in forward, code: out = self.avg_pool(out)
        mean: "f32[1, 64, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True);  relu_6 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:60 in forward, code: out = out.view(out.size(0), -1)
        view: "f32[1, 64]" = torch.ops.aten.reshape.default(mean, [1, -1]);  mean = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:61 in forward, code: out = self.fc(out)
        permute: "f32[64, 10]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm: "f32[1, 10]" = torch.ops.aten.addmm.default(arg47_1, view, permute);  arg47_1 = view = permute = None
        return (addmm,)
        