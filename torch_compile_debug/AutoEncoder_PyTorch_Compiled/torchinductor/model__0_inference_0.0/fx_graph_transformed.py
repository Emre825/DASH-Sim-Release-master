class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2048, 2048]", arg1_1: "f32[2048]", arg2_1: "f32[1, 2048]", arg3_1: "f32[2048, 2048]", arg4_1: "f32[2048]", arg5_1: "f32[2048, 2048]", arg6_1: "f32[2048]", arg7_1: "f32[2048, 2048]", arg8_1: "f32[2048]", arg9_1: "f32[256, 2048]", arg10_1: "f32[256]", arg11_1: "f32[2048, 256]", arg12_1: "f32[2048]", arg13_1: "f32[2048, 2048]", arg14_1: "f32[2048]", arg15_1: "f32[2048, 2048]", arg16_1: "f32[2048]", arg17_1: "f32[2048, 2048]", arg18_1: "f32[2048]", arg19_1: "f32[2048, 2048]", arg20_1: "f32[2048]"):
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:32 in forward, code: x = F.relu(self.fc1(x))
        permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm_default_8: "f32[1, 2048]" = torch.ops.aten.mm.default(arg2_1, permute);  arg2_1 = permute = None
        add_tensor_8: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg1_1, mm_default_8);  arg1_1 = mm_default_8 = None
        relu: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_8);  add_tensor_8 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:33 in forward, code: x = F.relu(self.fc2(x))
        permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm_default_7: "f32[1, 2048]" = torch.ops.aten.mm.default(relu, permute_1);  relu = permute_1 = None
        add_tensor_7: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg4_1, mm_default_7);  arg4_1 = mm_default_7 = None
        relu_1: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:34 in forward, code: x = F.relu(self.fc3(x))
        permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_default_6: "f32[1, 2048]" = torch.ops.aten.mm.default(relu_1, permute_2);  relu_1 = permute_2 = None
        add_tensor_6: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg6_1, mm_default_6);  arg6_1 = mm_default_6 = None
        relu_2: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_6);  add_tensor_6 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:35 in forward, code: x = F.relu(self.fc4(x))
        permute_3: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_default_5: "f32[1, 2048]" = torch.ops.aten.mm.default(relu_2, permute_3);  relu_2 = permute_3 = None
        add_tensor_5: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg8_1, mm_default_5);  arg8_1 = mm_default_5 = None
        relu_3: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:36 in forward, code: x = F.relu(self.fc5(x))
        permute_4: "f32[2048, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_default_4: "f32[1, 256]" = torch.ops.aten.mm.default(relu_3, permute_4);  relu_3 = permute_4 = None
        add_tensor_4: "f32[1, 256]" = torch.ops.aten.add.Tensor(arg10_1, mm_default_4);  arg10_1 = mm_default_4 = None
        relu_4: "f32[1, 256]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:37 in forward, code: x = F.relu(self.fc6(x))
        permute_5: "f32[256, 2048]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_default_3: "f32[1, 2048]" = torch.ops.aten.mm.default(relu_4, permute_5);  relu_4 = permute_5 = None
        add_tensor_3: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg12_1, mm_default_3);  arg12_1 = mm_default_3 = None
        relu_5: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:38 in forward, code: x = F.relu(self.fc7(x))
        permute_6: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_default_2: "f32[1, 2048]" = torch.ops.aten.mm.default(relu_5, permute_6);  relu_5 = permute_6 = None
        add_tensor_2: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg14_1, mm_default_2);  arg14_1 = mm_default_2 = None
        relu_6: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:39 in forward, code: x = F.relu(self.fc8(x))
        permute_7: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_default_1: "f32[1, 2048]" = torch.ops.aten.mm.default(relu_6, permute_7);  relu_6 = permute_7 = None
        add_tensor_1: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg16_1, mm_default_1);  arg16_1 = mm_default_1 = None
        relu_7: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:40 in forward, code: x = F.relu(self.fc9(x))
        permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_default: "f32[1, 2048]" = torch.ops.aten.mm.default(relu_7, permute_8);  relu_7 = permute_8 = None
        add_tensor: "f32[1, 2048]" = torch.ops.aten.add.Tensor(arg18_1, mm_default);  arg18_1 = mm_default = None
        relu_8: "f32[1, 2048]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:41 in forward, code: x = self.output_layer(x)
        permute_9: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_9: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg20_1, relu_8, permute_9);  arg20_1 = relu_8 = permute_9 = None
        return (addmm_9,)
        