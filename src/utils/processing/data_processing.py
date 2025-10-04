import math
import random
import torch
import gc
from typing import overload
import threading
import queue


def check_and_raise(condition:bool,text:str):
    if condition:
        raise ValueError(text)

@overload
def batchify(input:list|torch.Tensor, label:list|torch.Tensor=None, batch_size:int=1000)->tuple[list,list]:...
@overload
def batchify(input:list|torch.Tensor,*, batch_size:int=1000)->list:...
def batchify(input:list|torch.Tensor, label:list|torch.Tensor=None, batch_size:int=1000):
    if label is not None:
        out_x = []
        out_label = []
        for i in range(math.ceil(len(input)/batch_size)):
            out_x.append(input[i*batch_size:(i+1)*batch_size])
            out_label.append(label[i*batch_size:(i+1)*batch_size])
        return (out_x, out_label)
    else:
        out = []
        for i in range(math.ceil(len(input)/batch_size)):
            out.append(input[i*batch_size:(i+1)*batch_size])
        return out


def flatten(x:list[list]) -> list:
    '''(B, L, seq_len) -> (B*L, seq_len)'''
    return [item for sublist in x for item in sublist]

swap = lambda x, y: (y, x)
def shuffle(x:list):
    '''x의 순서를 랜덤하게 섞음'''
    for k in range(2):
        for i in range(len(x)-1, 0, -1):
            j = random.randint(0, i)
            x[i], x[j] = swap(x[i], x[j])
    return x
def shuffle_dataset(x:list, y:list,/):
    '''x와 y의 순서를 랜덤하게 섞음'''
    if len(x)!=len(y):
        raise IndexError("x와 y의 길이가 다름")
    for k in range(2):
        for i in range(len(x)-1, 0, -1):
            j = random.randint(0, i)
            x[i], x[j] = swap(x[i], x[j])
            y[i], y[j] = swap(y[i], y[j])
    return (x, y)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def delete_cudatensors(*passing_tensors_id:int) -> int:
    '''함수에 값 전달할 때는 꼭 id(텐서)로 전달해야함.\n
    반환값은 삭제한 텐서 개수'''
    i = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.device.type == 'cuda' and \
            id(obj) not in passing_tensors_id:
            del obj
            i+=1
    clear_memory()
    return i

import torch
from typing import Dict
def get_param_norms(params: dict) -> Dict[str, float]:
    """
    모든 파라미터에 대해 L2 norm 계산.
    qkv_weights -> L{layer}/H{head}/{Q,K,V}
    scale_qk    -> L{layer}/H{head}
    나머지      -> 레이어별 또는 전체 norm
    """
    norms = {}
    qkv = params['qkv_weights']
    num_layers = qkv.shape[0]
    num_heads = params['scale_qk'].shape[1]

    for name, tensor in params.items():
        if not torch.is_tensor(tensor):
            continue
        with torch.no_grad():
            # QKV (레이어 + 헤드 단위)
            if name == 'qkv_weights':
                d_model = tensor.shape[1]
                d_head = d_model // num_heads
                head_dim_total = d_head * num_heads
                head_q, head_k, head_v = torch.split(
                    tensor.view(num_layers, d_model, num_heads, d_head * 3),
                    d_head, dim=3
                )  # 각각 shape: (L, d_model, H, d_head)
                for l in range(num_layers):
                    for h in range(num_heads):
                        norms[f"{name}/L{l}/H{h}/Q"] = head_q[l, :, h].norm(2).item()
                        norms[f"{name}/L{l}/H{h}/K"] = head_k[l, :, h].norm(2).item()
                        norms[f"{name}/L{l}/H{h}/V"] = head_v[l, :, h].norm(2).item()

            # scale_qk (레이어 + 헤드 단위)
            elif name == 'scale_qk':
                for l in range(tensor.shape[0]):
                    for h in range(tensor.shape[1]):
                        norms[f"{name}/L{l}/H{h}"] = tensor[l, h].norm(2).item()

            # 레이어별 파라미터
            elif tensor.ndim >= 2 and tensor.shape[0] == num_layers:
                for l in range(num_layers):
                    norms[f"{name}/L{l}"] = tensor[l].norm(2).item()

            # 나머지 (전체)
            else:
                norms[name] = tensor.norm(2).item()

    return norms


def get_grad_norms(params: dict) -> Dict[str, float]:
    norms = {}
    qkv = params['qkv_weights']
    num_layers = qkv.shape[0]
    num_heads = params['scale_qk'].shape[1]

    for name, tensor in params.items():
        if not torch.is_tensor(tensor) or tensor.grad is None:
            continue
        with torch.no_grad():
            if name == 'qkv_weights':
                d_model = tensor.shape[1]
                d_head = d_model // num_heads
                head_q, head_k, head_v = torch.split(
                    tensor.grad.view(num_layers, d_model, num_heads, d_head * 3),
                    d_head, dim=3
                )
                for l in range(num_layers):
                    for h in range(num_heads):
                        norms[f"{name}/L{l}/H{h}/Q"] = head_q[l, :, h].norm(2).item()
                        norms[f"{name}/L{l}/H{h}/K"] = head_k[l, :, h].norm(2).item()
                        norms[f"{name}/L{l}/H{h}/V"] = head_v[l, :, h].norm(2).item()

            elif name == 'scale_qk':
                for l in range(tensor.shape[0]):
                    for h in range(tensor.shape[1]):
                        norms[f"{name}/L{l}/H{h}"] = tensor.grad[l, h].norm(2).item()

            elif tensor.ndim >= 2 and tensor.shape[0] == num_layers:
                for l in range(num_layers):
                    norms[f"{name}/L{l}"] = tensor.grad[l].norm(2).item()

            else:
                norms[name] = tensor.grad.norm(2).item()

    return norms


class Saver:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Saver, cls).__new__(cls)
        return cls.instance


    def __init__(self):
        self.save_queue = queue.Queue()
    

    def _worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            torch.save(item[0], item[1])
            self.save_queue.task_done()
            del item

    def save(self, obj, path):
        self.save_queue.put((obj, path))
    
    def close(self):
        self.save_queue.put(None)
        self.save_queue.join()

    def __del__(self):
        self.close()

saver = Saver()
