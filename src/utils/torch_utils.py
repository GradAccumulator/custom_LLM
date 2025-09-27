import torch

def restore_param(tensor:torch.Tensor, *args, **kwargs):
    return tensor.to(*args, **kwargs).requires_grad_()

def 보이면_바로_교체하기(arg1=None, arg2=None):
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

def device_dtype_check(*args, default_device=None, default_dtype=None, **kwargs) -> tuple[torch.device, torch.dtype]:
    device = default_device
    dtype = default_dtype
    
    if kwargs is not None:
        device = kwargs.get("device", device)
        dtype = kwargs.get("dtype", dtype)
        if dtype is not None:
            if dtype=='float32' or dtype == 'float':
                dtype = torch.float32
            elif dtype=='float16':
                dtype = torch.float16
            elif not isinstance(dtype, torch.dtype):
                raise TypeError(f"Invalid dtype {dtype}")
        
        if device is not None:
            try:
                device = torch.device(device)
            except Exception as e:
                raise TypeError(f'Invalid device {device}') from e
    if args is not None:
        for arg in args:
            if isinstance(arg, str):
                if arg == 'float16':
                    dtype = torch.float16
                elif arg == 'float32' or arg == 'float':
                    dtype = torch.float32
                else:
                    try:
                        device = torch.device(arg)
                    except Exception as e:
                        raise TypeError(f'Invalid device {device}') from e
            elif isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
    
    return device,dtype