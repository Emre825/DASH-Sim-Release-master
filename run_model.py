from obj import Model
import torch
import torch.nn as nn

model = Model().cuda()
x = torch.randn(1, 3, 224, 224).cuda()  # Change resolution if needed [Batch Size, Channel, Height, Width]
model.eval()

def log_shapes(name):
    def hook(module, input, output):
        print(f"{name} | In: {[i.shape for i in input]} | Out: {output.shape}")
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Upsample)):
        layer.register_forward_hook(log_shapes(name))

with torch.no_grad():
    model(x)


