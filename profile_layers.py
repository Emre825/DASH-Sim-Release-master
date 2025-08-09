import torch
import torch.nn as nn
import json
from obj import Model

# Dictionary to store profiling results
profile_results = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Model().to(device)
model.eval()

# Helper function to compute tensor volume in bits
def get_volume_bits(tensor):
    return tensor.numel() * tensor.element_size() * 8

# Create pre and post hooks for GPU timing and volume computations
def create_hooks(name, module):
    def pre_hook(module, input):
        
        # init per-module call counter (once)
        if not hasattr(module, "_call_idx"):
            module._call_idx = -1
        module._call_idx += 1

        # Create and record start event
        module._start_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        module._start_event.record()
    def post_hook(module, input, output):
        # Record end event and compute elapsed time
        torch.cuda.synchronize()
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = module._start_event.elapsed_time(end_event)  # time in ms
        
        # Compute input volume in bits
        total_input_bits = 0
        for inp in input:
            if isinstance(inp, torch.Tensor):
                total_input_bits += get_volume_bits(inp)
            elif isinstance(inp, (list, tuple)):
                for t in inp:
                    if isinstance(t, torch.Tensor):
                        total_input_bits += get_volume_bits(t)
        
        # Compute output volume in bits
        total_output_bits = 0
        if isinstance(output, torch.Tensor):
            total_output_bits += get_volume_bits(output)
        elif isinstance(output, (list, tuple)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    total_output_bits += get_volume_bits(t)
        
        key = f"{name}[{module._call_idx}]"
        
        profile_results[key] = {
            "layer_type": type(module).__name__,
            "execution_time_ms": elapsed_ms,
            "input_volume_bits": total_input_bits,
            "output_volume_bits": total_output_bits
        }
    return pre_hook, post_hook

# Attach hooks to modules of interest
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.UpsamplingNearest2d, nn.Sigmoid)):
        pre, post = create_hooks(name, module)
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

# Monkey patch torch.cat to profile cat operations.
_original_cat = torch.cat
def profiled_cat(tensors, dim=0):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    output = _original_cat(tensors, dim)
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    total_input_bits = sum(get_volume_bits(t) for t in tensors if isinstance(t, torch.Tensor))
    total_output_bits = get_volume_bits(output)
    # Use a unique key for each call
    key = f"torch.cat_{id(start)}"
    profile_results[key] = {
        "layer_type": "cat",
        "execution_time_ms": elapsed_ms,
        "input_volume_bits": total_input_bits,
        "output_volume_bits": total_output_bits
    }
    return output
torch.cat = profiled_cat

# Run the model with a dummy input and record profiling info
x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    _ = model(x)

# Print the profiling results as JSON
print(json.dumps(profile_results, indent=4))