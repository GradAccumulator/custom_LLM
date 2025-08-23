import torch


def device_dtype_check(arg1=None, arg2=None):
    dtype = None
    device = None
    if isinstance(arg1, str):
        device = arg1
    elif isinstance(arg1, torch.dtype):
        dtype = arg1
    if isinstance(arg2, str):
        device = arg2
    elif isinstance(arg2, torch.dtype):
        dtype = arg2
    return device, dtype