import sentencepiece as spm
import os
import ctypes
import numpy as np
import traceback
#my packages
import src.utils.dev_utils as dev_utils
from PATHES import PATHES

process_bottleneck = ctypes.CDLL(dev_utils.join(PATHES.SRC_C_MODULES, 'dlls', "seq_windows.dll"))

#(int** encoded, int max_length, int size, int* lengths, int seq_len, float stride_rate, int pad_id)
process_bottleneck.make_seq_windows_with_padding.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)), 
    ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int), 
    ctypes.c_int, 
    ctypes.c_float,
    ctypes.c_int
]
class results(ctypes.Structure):
    _fields_ = [
        ('arr_x', ctypes.POINTER(ctypes.c_int16)), 
        ('arr_label', ctypes.POINTER(ctypes.c_int16)),
        ('act_len', ctypes.c_int),
]
process_bottleneck.make_seq_windows_with_padding.restype = results
process_bottleneck.free_results.argtypes = [
    ctypes.POINTER(ctypes.c_int16)
]
process_bottleneck.free_results.restype = None

class koTokenizer:
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        self.num_cpus = os.cpu_count()

    @property
    def eos_id(self):
        return self.tokenizer.eos_id()
    @property
    def bos_id(self):
        return self.tokenizer.bos_id()
    @property
    def pad_id(self):
        return self.tokenizer.pad_id()
    @property
    def unk_id(self):
        return self.tokenizer.unk_id()
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()
    def __len__(self):
        return self.tokenizer.vocab_size()
    
    def get_id(self, token:str) -> int:
        return self.tokenizer.Encode(token, out_type=int, add_bos=False, add_eos=False)[0]

    def add_padding(self, encoded:list[list[int]], seq_len:int):
        out = []
        for x in encoded:
            if len(x)>=seq_len:
                out.append(x[:seq_len])
            else:
                out.append(x + ([self.pad_id] * (seq_len-len(x))))
        return out
    
    def encode(self, x:str|list[str], bos=True, eos=True, seq_len=None, is_dataset=False, stride_rate:float=0.75, max_length=1000000) -> list[int]|tuple[list[int],list[int]] | list[list[int]]|tuple[list[list[int]],list[list[int]]]:
        '''
        입력은 (L), 출력은 (L, seq_len)\t 
        #seq_len==None이면 각자 문장 크기\n
        만약 입력이 (L, batch)라면 batch_encode로 하기\n
        아니면 직접 for문 돌리든가\n
        만약 is_dataset == True면\n
        출력은 (L,seq_len)크기의 str list 2개
        --> (input, label)
        '''

        if type(x) == str:
            x = [x]

        if not is_dataset:
            encoded = self.tokenizer.Encode(x, out_type=int, add_bos=bos, add_eos=eos, num_threads=self.num_cpus)
            if seq_len == None:
                return encoded
            out = self.add_padding(encoded, seq_len)
            return out
        else:
            #with dev_utils.timer("인코딩"):
            encoded = self.tokenizer.Encode(x, out_type=int, add_bos=False, add_eos=False, num_threads=self.num_cpus)
            del x

            if seq_len == None:
                encoded_x = encoded
                encoded_label = [e[1:] for e in encoded]
                return (encoded_x, encoded_label)
            else:
                try:
                    #with dev_utils.parent_timer("깊은복사(python->C)"):
                        #with dev_utils.timer("int* lengths 만들기"):
                    c_lengths = \
                        (ctypes.c_int * len(encoded))(
                            *[len(e) for e in encoded]
                        )
                        #with dev_utils.timer("encoded -> int** 변환"):
                    np_array = [np.array(e, dtype=np.int16) for e in encoded]
                    ptr_list= [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)) for arr in np_array]
                    c_array = (ctypes.POINTER(ctypes.c_int16) * len(ptr_list))(*ptr_list)
                    if type(max_length) != int:
                            max_length = int(max_length)
                    #with dev_utils.timer("C함수", 3):
                        #(int16** encoded, int max_length, int size, int* lengths, int seq_len, float stride_rate, int pad_id)
                    res = process_bottleneck.make_seq_windows_with_padding(c_array, max_length, len(encoded), c_lengths, seq_len, stride_rate, self.pad_id)
                    del c_array
                    del c_lengths

                    #with dev_utils.parent_timer("깊은복사(C->python)", 3):
                        #with dev_utils.timer("입력값 깊은복사"):
                    out_x = np.ctypeslib.as_array(res.arr_x, shape=(max_length, seq_len))[:res.act_len].astype(np.int16)
                    process_bottleneck.free_results(res.arr_x, max_length)
                        #with dev_utils.timer("label값 깊은복사"):
                    out_label = np.ctypeslib.as_array(res.arr_label, shape=(max_length, seq_len))[:res.act_len].astype(np.int16)
                    process_bottleneck.free_results(res.arr_label, max_length)
                except Exception:
                    traceback.print_exc()
                    process_bottleneck.free_results(res.arr_x, max_length)
                    process_bottleneck.free_results(res.arr_label, max_length)
            return (out_x, out_label)
    
    def decode(self, input:list[list[int]]|list[int]) -> list[str]|str:
        return self.tokenizer.Decode(input, out_type=str,num_threads=self.num_cpus-2)
    
    def compute_frequency(self, x:list[list[int]]) -> dict[int,int]:
        '''
        입력은 (L, seq_len)크기의 list
        출력은 {토큰ID: 빈도수} 형태의 dict
        '''
        from collections import Counter
        tokens = np.array(x).flatten()
        return Counter(tokens)