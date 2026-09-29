class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2048, 2048]", arg1_1: "f32[2048]", arg2_1: "f32[1, 2048]", arg3_1: "f32[2048, 2048]", arg4_1: "f32[2048]", arg5_1: "f32[2048, 2048]", arg6_1: "f32[2048]", arg7_1: "f32[2048, 2048]", arg8_1: "f32[2048]", arg9_1: "f32[256, 2048]", arg10_1: "f32[256]", arg11_1: "f32[2048, 256]", arg12_1: "f32[2048]", arg13_1: "f32[2048, 2048]", arg14_1: "f32[2048]", arg15_1: "f32[2048, 2048]", arg16_1: "f32[2048]", arg17_1: "f32[2048, 2048]", arg18_1: "f32[2048]", arg19_1: "f32[2048, 2048]", arg20_1: "f32[2048]"):
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:32 in forward, code: x = F.relu(self.fc1(x))
        permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        relu: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm);  addmm = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:33 in forward, code: x = F.relu(self.fc2(x))
        permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg4_1, relu, permute_1);  arg4_1 = relu = permute_1 = None
        relu_1: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:34 in forward, code: x = F.relu(self.fc3(x))
        permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg6_1, relu_1, permute_2);  arg6_1 = relu_1 = permute_2 = None
        relu_2: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:35 in forward, code: x = F.relu(self.fc4(x))
        permute_3: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_3: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg8_1, relu_2, permute_3);  arg8_1 = relu_2 = permute_3 = None
        relu_3: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_3);  addmm_3 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:36 in forward, code: x = F.relu(self.fc5(x))
        permute_4: "f32[2048, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_4: "f32[1, 256]" = torch.ops.aten.addmm.default(arg10_1, relu_3, permute_4);  arg10_1 = relu_3 = permute_4 = None
        relu_4: "f32[1, 256]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:37 in forward, code: x = F.relu(self.fc6(x))
        permute_5: "f32[256, 2048]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_5: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg12_1, relu_4, permute_5);  arg12_1 = relu_4 = permute_5 = None
        relu_5: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_5);  addmm_5 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:38 in forward, code: x = F.relu(self.fc7(x))
        permute_6: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_6: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg14_1, relu_5, permute_6);  arg14_1 = relu_5 = permute_6 = None
        relu_6: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_6);  addmm_6 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:39 in forward, code: x = F.relu(self.fc8(x))
        permute_7: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_7: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg16_1, relu_6, permute_7);  arg16_1 = relu_6 = permute_7 = None
        relu_7: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_7);  addmm_7 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:40 in forward, code: x = F.relu(self.fc9(x))
        permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_8: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg18_1, relu_7, permute_8);  arg18_1 = relu_7 = permute_8 = None
        relu_8: "f32[1, 2048]" = torch.ops.aten.relu.default(addmm_8);  addmm_8 = None
        
        # File: /home/emre/Desktop/DASH-Sim-Release-master/PyTorch_compile.py:41 in forward, code: x = self.output_layer(x)
        permute_9: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_9: "f32[1, 2048]" = torch.ops.aten.addmm.default(arg20_1, relu_8, permute_9);  arg20_1 = relu_8 = permute_9 = None
        return (addmm_9,)
        