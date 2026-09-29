class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 3, 3]", arg1_1: "f32[8]", arg2_1: "f32[1, 3, 256, 256]", arg3_1: "f32[8, 8, 3, 3]", arg4_1: "f32[8]", arg5_1: "f32[16, 8, 3, 3]", arg6_1: "f32[16]", arg7_1: "f32[16, 16, 3, 3]", arg8_1: "f32[16]", arg9_1: "f32[32, 16, 3, 3]", arg10_1: "f32[32]", arg11_1: "f32[32, 32, 3, 3]", arg12_1: "f32[32]", arg13_1: "f32[64, 32, 3, 3]", arg14_1: "f32[64]", arg15_1: "f32[64, 64, 3, 3]", arg16_1: "f32[64]", arg17_1: "f32[128, 64, 3, 3]", arg18_1: "f32[128]", arg19_1: "f32[128, 128, 3, 3]", arg20_1: "f32[128]", arg21_1: "f32[64, 192, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64, 64, 3, 3]", arg24_1: "f32[64]", arg25_1: "f32[32, 96, 3, 3]", arg26_1: "f32[32]", arg27_1: "f32[32, 32, 3, 3]", arg28_1: "f32[32]", arg29_1: "f32[16, 48, 3, 3]", arg30_1: "f32[16]", arg31_1: "f32[16, 16, 3, 3]", arg32_1: "f32[16]", arg33_1: "f32[8, 24, 3, 3]", arg34_1: "f32[8]", arg35_1: "f32[8, 8, 3, 3]", arg36_1: "f32[8]", arg37_1: "f32[1, 8, 1, 1]", arg38_1: "f32[1]"):
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:40 in forward, code: x = self.down1a(x)
        convolution: "f32[1, 8, 256, 256]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "f32[1, 8, 256, 256]" = torch.ops.aten.relu.default(convolution);  convolution = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:41 in forward, code: conv1 = self.down1b(x)
        convolution_1: "f32[1, 8, 256, 256]" = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
        relu_1: "f32[1, 8, 256, 256]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:42 in forward, code: x = self.maxpool(conv1)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[1, 8, 128, 128]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:43 in forward, code: x = self.down2a(x)
        convolution_2: "f32[1, 16, 128, 128]" = torch.ops.aten.convolution.default(getitem, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg5_1 = arg6_1 = None
        relu_2: "f32[1, 16, 128, 128]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:44 in forward, code: conv2 = self.down2b(x)
        convolution_3: "f32[1, 16, 128, 128]" = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3: "f32[1, 16, 128, 128]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:45 in forward, code: x = self.maxpool(conv2)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "f32[1, 16, 64, 64]" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:46 in forward, code: x = self.down3a(x)
        convolution_4: "f32[1, 32, 64, 64]" = torch.ops.aten.convolution.default(getitem_2, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg9_1 = arg10_1 = None
        relu_4: "f32[1, 32, 64, 64]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:47 in forward, code: conv3 = self.down3b(x)
        convolution_5: "f32[1, 32, 64, 64]" = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = arg12_1 = None
        relu_5: "f32[1, 32, 64, 64]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:48 in forward, code: x = self.maxpool(conv3)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "f32[1, 32, 32, 32]" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:49 in forward, code: x = self.down4a(x)
        convolution_6: "f32[1, 64, 32, 32]" = torch.ops.aten.convolution.default(getitem_4, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg13_1 = arg14_1 = None
        relu_6: "f32[1, 64, 32, 32]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:50 in forward, code: conv4 = self.down4b(x)
        convolution_7: "f32[1, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_6, arg15_1, arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_6 = arg15_1 = arg16_1 = None
        relu_7: "f32[1, 64, 32, 32]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:51 in forward, code: x = self.maxpool(conv4)
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6: "f32[1, 64, 16, 16]" = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:52 in forward, code: x = self.upreg1(x)
        convolution_8: "f32[1, 128, 16, 16]" = torch.ops.aten.convolution.default(getitem_6, arg17_1, arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg17_1 = arg18_1 = None
        relu_8: "f32[1, 128, 16, 16]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:53 in forward, code: x = self.upreg2(x)
        convolution_9: "f32[1, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_8, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg19_1 = arg20_1 = None
        relu_9: "f32[1, 128, 16, 16]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:54 in forward, code: x = self.upsample(x)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i64[32]" = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add: "i64[32]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        add_1: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type, 0.0);  convert_element_type = None
        mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        convert_element_type_1: "i64[32]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        unsqueeze: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1);  convert_element_type_1 = None
        iota_1: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2: "i64[32]" = torch.ops.aten.mul.Tensor(iota_1, 1);  iota_1 = None
        add_2: "i64[32]" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None
        convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        add_3: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_2, 0.0);  convert_element_type_2 = None
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        convert_element_type_3: "i64[32]" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None
        _unsafe_index: "f32[1, 128, 32, 32]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, unsqueeze, convert_element_type_3]);  relu_9 = unsqueeze = convert_element_type_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:55 in forward, code: x = torch.cat((x, conv4), 1)
        cat: "f32[1, 192, 32, 32]" = torch.ops.aten.cat.default([_unsafe_index, relu_7], 1);  _unsafe_index = relu_7 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:56 in forward, code: x = self.up1a(x)
        convolution_10: "f32[1, 64, 32, 32]" = torch.ops.aten.convolution.default(cat, arg21_1, arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat = arg21_1 = arg22_1 = None
        relu_10: "f32[1, 64, 32, 32]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:57 in forward, code: x = self.up1b(x)
        convolution_11: "f32[1, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg23_1 = arg24_1 = None
        relu_11: "f32[1, 64, 32, 32]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:58 in forward, code: x = self.upsample(x)
        iota_2: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_4: "i64[64]" = torch.ops.aten.mul.Tensor(iota_2, 1);  iota_2 = None
        add_4: "i64[64]" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None
        convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        add_5: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_4, 0.0);  convert_element_type_4 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
        convert_element_type_5: "i64[64]" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None
        unsqueeze_1: "i64[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1);  convert_element_type_5 = None
        iota_3: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_6: "i64[64]" = torch.ops.aten.mul.Tensor(iota_3, 1);  iota_3 = None
        add_6: "i64[64]" = torch.ops.aten.add.Tensor(mul_6, 0);  mul_6 = None
        convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_6, 0.0);  convert_element_type_6 = None
        mul_7: "f32[64]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        convert_element_type_7: "i64[64]" = torch.ops.prims.convert_element_type.default(mul_7, torch.int64);  mul_7 = None
        _unsafe_index_1: "f32[1, 64, 64, 64]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, unsqueeze_1, convert_element_type_7]);  relu_11 = unsqueeze_1 = convert_element_type_7 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:59 in forward, code: x = torch.cat((x, conv3), 1)
        cat_1: "f32[1, 96, 64, 64]" = torch.ops.aten.cat.default([_unsafe_index_1, relu_5], 1);  _unsafe_index_1 = relu_5 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:60 in forward, code: x = self.up2a(x)
        convolution_12: "f32[1, 32, 64, 64]" = torch.ops.aten.convolution.default(cat_1, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_1 = arg25_1 = arg26_1 = None
        relu_12: "f32[1, 32, 64, 64]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:61 in forward, code: x = self.up2b(x)
        convolution_13: "f32[1, 32, 64, 64]" = torch.ops.aten.convolution.default(relu_12, arg27_1, arg28_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_12 = arg27_1 = arg28_1 = None
        relu_13: "f32[1, 32, 64, 64]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:62 in forward, code: x = self.upsample(x)
        iota_4: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_8: "i64[128]" = torch.ops.aten.mul.Tensor(iota_4, 1);  iota_4 = None
        add_8: "i64[128]" = torch.ops.aten.add.Tensor(mul_8, 0);  mul_8 = None
        convert_element_type_8: "f32[128]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        add_9: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_8, 0.0);  convert_element_type_8 = None
        mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        convert_element_type_9: "i64[128]" = torch.ops.prims.convert_element_type.default(mul_9, torch.int64);  mul_9 = None
        unsqueeze_2: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1);  convert_element_type_9 = None
        iota_5: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_10: "i64[128]" = torch.ops.aten.mul.Tensor(iota_5, 1);  iota_5 = None
        add_10: "i64[128]" = torch.ops.aten.add.Tensor(mul_10, 0);  mul_10 = None
        convert_element_type_10: "f32[128]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        add_11: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_10, 0.0);  convert_element_type_10 = None
        mul_11: "f32[128]" = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
        convert_element_type_11: "i64[128]" = torch.ops.prims.convert_element_type.default(mul_11, torch.int64);  mul_11 = None
        _unsafe_index_2: "f32[1, 32, 128, 128]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, unsqueeze_2, convert_element_type_11]);  relu_13 = unsqueeze_2 = convert_element_type_11 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:63 in forward, code: x = torch.cat((x, conv2), 1)
        cat_2: "f32[1, 48, 128, 128]" = torch.ops.aten.cat.default([_unsafe_index_2, relu_3], 1);  _unsafe_index_2 = relu_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:64 in forward, code: x = self.up3a(x)
        convolution_14: "f32[1, 16, 128, 128]" = torch.ops.aten.convolution.default(cat_2, arg29_1, arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_2 = arg29_1 = arg30_1 = None
        relu_14: "f32[1, 16, 128, 128]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:65 in forward, code: x = self.up3b(x)
        convolution_15: "f32[1, 16, 128, 128]" = torch.ops.aten.convolution.default(relu_14, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_14 = arg31_1 = arg32_1 = None
        relu_15: "f32[1, 16, 128, 128]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:66 in forward, code: x = self.upsample(x)
        iota_6: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_12: "i64[256]" = torch.ops.aten.mul.Tensor(iota_6, 1);  iota_6 = None
        add_12: "i64[256]" = torch.ops.aten.add.Tensor(mul_12, 0);  mul_12 = None
        convert_element_type_12: "f32[256]" = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        add_13: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_12, 0.0);  convert_element_type_12 = None
        mul_13: "f32[256]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        convert_element_type_13: "i64[256]" = torch.ops.prims.convert_element_type.default(mul_13, torch.int64);  mul_13 = None
        unsqueeze_3: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_13, -1);  convert_element_type_13 = None
        iota_7: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_14: "i64[256]" = torch.ops.aten.mul.Tensor(iota_7, 1);  iota_7 = None
        add_14: "i64[256]" = torch.ops.aten.add.Tensor(mul_14, 0);  mul_14 = None
        convert_element_type_14: "f32[256]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        add_15: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_14, 0.0);  convert_element_type_14 = None
        mul_15: "f32[256]" = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        convert_element_type_15: "i64[256]" = torch.ops.prims.convert_element_type.default(mul_15, torch.int64);  mul_15 = None
        _unsafe_index_3: "f32[1, 16, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, unsqueeze_3, convert_element_type_15]);  relu_15 = unsqueeze_3 = convert_element_type_15 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:67 in forward, code: x = torch.cat((x, conv1), 1)
        cat_3: "f32[1, 24, 256, 256]" = torch.ops.aten.cat.default([_unsafe_index_3, relu_1], 1);  _unsafe_index_3 = relu_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:68 in forward, code: x = self.up4a(x)
        convolution_16: "f32[1, 8, 256, 256]" = torch.ops.aten.convolution.default(cat_3, arg33_1, arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_3 = arg33_1 = arg34_1 = None
        relu_16: "f32[1, 8, 256, 256]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:69 in forward, code: x = self.up4b(x)
        convolution_17: "f32[1, 8, 256, 256]" = torch.ops.aten.convolution.default(relu_16, arg35_1, arg36_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg35_1 = arg36_1 = None
        relu_17: "f32[1, 8, 256, 256]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:70 in forward, code: x = self.upreg3(x)
        convolution_18: "f32[1, 1, 256, 256]" = torch.ops.aten.convolution.default(relu_17, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg37_1 = arg38_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:71 in forward, code: x = self.sig(x)
        sigmoid: "f32[1, 1, 256, 256]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        return (sigmoid,)
        