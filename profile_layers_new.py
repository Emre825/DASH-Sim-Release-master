import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from new_obj import BiFPNNet

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Helpers ---
def get_volume_bits(tensor):
    if not isinstance(tensor, torch.Tensor):
        return 0
    return tensor.numel() * tensor.element_size() * 8

def sum_vol_bits(x):
    total = 0
    if isinstance(x, torch.Tensor):
        total += get_volume_bits(x)
    elif isinstance(x, (list, tuple)):
        for t in x:
            total += sum_vol_bits(t)
    elif isinstance(x, dict):
        for t in x.values():
            total += sum_vol_bits(t)
    return total

profile_results = {}
_global_op_counter = {"functional": 0}

def record_op(name, layer_type, start_event, end_event, inputs, outputs):
    # compute elapsed (ms) using CUDA events when available, else use time delta stored in start_event
    if isinstance(start_event, torch.cuda.Event):
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (end_event - start_event) * 1000.0
    input_bits = sum_vol_bits(inputs)
    output_bits = sum_vol_bits(outputs)
    key = f"{name}"
    profile_results[key] = {
        "layer_type": layer_type,
        "execution_time_ms": float(elapsed_ms),
        "input_volume_bits": int(input_bits),
        "output_volume_bits": int(output_bits)
    }

# --- Model and module hooks ---
model = BiFPNNet().to(device)
model.eval()

def create_hooks(name, module):
    # maintain call index per module to disambiguate repeated calls
    if not hasattr(module, "_call_idx"):
        module._call_idx = -1

    def pre_hook(mod, inp):
        mod._call_idx += 1
        if torch.cuda.is_available():
            mod._start_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            mod._start_event.record()
        else:
            mod._start_event = time.time()
    def post_hook(mod, inp, out):
        if torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            end.record()
        else:
            end = time.time()
        opname = f"{name}[{mod._call_idx}]"
        record_op(opname, type(mod).__name__, mod._start_event, end, inp, out)
    return pre_hook, post_hook

# Attach hooks for module-based layers
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Sigmoid, nn.MaxPool2d)):
        pre, post = create_hooks(name, module)
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

# --- Monkey-patch functional ops and torch.* helpers ---
_orig_interpolate = F.interpolate
_orig_adaptive_avg_pool2d = F.adaptive_avg_pool2d
_orig_cat = torch.cat
_orig_torch_add = torch.add
_orig_torch_mul = torch.mul

def _wrap_functional(orig_fn, opname_base):
    def wrapper(*args, **kwargs):
        _global_op_counter["functional"] += 1
        idx = _global_op_counter["functional"]
        name = f"{opname_base}_{idx}"
        # start
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
        else:
            start = time.time()
        out = orig_fn(*args, **kwargs)
        # end
        if torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            end.record()
        else:
            end = time.time()
        inputs = args
        record_op(name, opname_base, start, end, inputs, out)
        return out
    return wrapper

F.interpolate = _wrap_functional(_orig_interpolate, "interpolate")
F.adaptive_avg_pool2d = _wrap_functional(_orig_adaptive_avg_pool2d, "adaptive_avg_pool2d")

def profiled_cat(tensors, dim=0, *args, **kwargs):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"cat_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_cat(tensors, dim, *args, **kwargs)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "cat", start, end, tensors, out)
    return out
torch.cat = profiled_cat

def profiled_torch_add(input, other, *args, **kwargs):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"add_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_torch_add(input, other, *args, **kwargs)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "add", start, end, (input, other), out)
    return out
torch.add = profiled_torch_add

def profiled_torch_mul(input, other, *args, **kwargs):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"mul_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_torch_mul(input, other, *args, **kwargs)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "mul", start, end, (input, other), out)
    return out
torch.mul = profiled_torch_mul

# --- Monkey-patch Tensor dunder methods to catch operator uses (a + b, a * b, etc.) ---
# keep originals
_orig_tensor_add = torch.Tensor.__add__
_orig_tensor_radd = getattr(torch.Tensor, "__radd__", None)
_orig_tensor_iadd = getattr(torch.Tensor, "__iadd__", None)
_orig_tensor_mul = torch.Tensor.__mul__
_orig_tensor_rmul = getattr(torch.Tensor, "__rmul__", None)
_orig_tensor_imul = getattr(torch.Tensor, "__imul__", None)

def _tensor_add(self, other):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"add_op_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_tensor_add(self, other)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "add", start, end, (self, other), out)
    return out

def _tensor_radd(self, other):
    # right-add funnels here when other.__add__ returned NotImplemented
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"radd_op_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_tensor_radd(self, other) if _orig_tensor_radd is not None else _orig_tensor_add(self, other)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "add", start, end, (other, self), out)
    return out

def _tensor_iadd(self, other):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"iadd_op_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_tensor_iadd(self, other) if _orig_tensor_iadd is not None else _orig_tensor_add(self, other)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "add", start, end, (self, other), out)
    return out

def _tensor_mul(self, other):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"mul_op_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_tensor_mul(self, other)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "mul", start, end, (self, other), out)
    return out

def _tensor_rmul(self, other):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"rmul_op_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_tensor_rmul(self, other) if _orig_tensor_rmul is not None else _orig_tensor_mul(self, other)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "mul", start, end, (other, self), out)
    return out

def _tensor_imul(self, other):
    _global_op_counter["functional"] += 1
    idx = _global_op_counter["functional"]
    name = f"imul_op_{idx}"
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start.record()
    else:
        start = time.time()
    out = _orig_tensor_imul(self, other) if _orig_tensor_imul is not None else _orig_tensor_mul(self, other)
    if torch.cuda.is_available():
        end = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); end.record()
    else:
        end = time.time()
    record_op(name, "mul", start, end, (self, other), out)
    return out

# Install dunder patches
torch.Tensor.__add__ = _tensor_add
if _orig_tensor_radd is not None:
    torch.Tensor.__radd__ = _tensor_radd
if _orig_tensor_iadd is not None:
    torch.Tensor.__iadd__ = _tensor_iadd
torch.Tensor.__mul__ = _tensor_mul
if _orig_tensor_rmul is not None:
    torch.Tensor.__rmul__ = _tensor_rmul
if _orig_tensor_imul is not None:
    torch.Tensor.__imul__ = _tensor_imul

# --- Run one forward pass with dummy input ---
x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    _ = model(x)

# --- Write results to a file ---
out_path = "profile_results.txt"
with open(out_path, "w") as f:
    json.dump(profile_results, f, indent=2)