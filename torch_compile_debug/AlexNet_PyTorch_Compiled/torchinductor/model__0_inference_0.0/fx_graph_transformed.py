class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 11, 11]", arg1_1: "f32[64]", arg2_1: "f32[64, 3, 224, 224]", arg3_1: "f32[192, 64, 5, 5]", arg4_1: "f32[192]", arg5_1: "f32[384, 192, 3, 3]", arg6_1: "f32[384]", arg7_1: "f32[256, 384, 3, 3]", arg8_1: "f32[256]", arg9_1: "f32[256, 256, 3, 3]", arg10_1: "f32[256]", arg11_1: "f32[4096, 9216]", arg12_1: "f32[4096]", arg13_1: "f32[4096, 4096]", arg14_1: "f32[4096]", arg15_1: "f32[1000, 4096]", arg16_1: "f32[1000]"):
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:39 in forward, code: x = self.features(x)
        convolution: "f32[64, 64, 55, 55]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "f32[64, 64, 55, 55]" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu = None
        getitem: "f32[64, 64, 27, 27]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_1: "f32[64, 192, 27, 27]" = torch.ops.aten.convolution.default(getitem, arg3_1, arg4_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  getitem = arg3_1 = arg4_1 = None
        relu_1: "f32[64, 192, 27, 27]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem_2: "f32[64, 192, 13, 13]" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_2: "f32[64, 384, 13, 13]" = torch.ops.aten.convolution.default(getitem_2, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg5_1 = arg6_1 = None
        relu_2: "f32[64, 384, 13, 13]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[64, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3: "f32[64, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        convolution_4: "f32[64, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_3, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg9_1 = arg10_1 = None
        relu_4: "f32[64, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_4 = None
        getitem_4: "f32[64, 256, 6, 6]" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:40 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "f32[64, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6]);  getitem_4 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:41 in forward, code: x = torch.flatten(x, 1)
        view: "f32[64, 9216]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [64, 9216]);  _adaptive_avg_pool2d = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:42 in forward, code: x = self.classifier(x)
        permute: "f32[9216, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_default_1: "f32[64, 4096]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        add_tensor_1: "f32[64, 4096]" = torch.ops.aten.add.Tensor(arg12_1, mm_default_1);  arg12_1 = mm_default_1 = None
        relu_5: "f32[64, 4096]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_default: "f32[64, 4096]" = torch.ops.aten.mm.default(relu_5, permute_1);  relu_5 = permute_1 = None
        add_tensor: "f32[64, 4096]" = torch.ops.aten.add.Tensor(arg14_1, mm_default);  arg14_1 = mm_default = None
        relu_6: "f32[64, 4096]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2: "f32[64, 1000]" = torch.ops.aten.addmm.default(arg16_1, relu_6, permute_2);  arg16_1 = relu_6 = permute_2 = None
        return (addmm_2,)
        