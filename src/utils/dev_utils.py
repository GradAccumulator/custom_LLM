from time import time
from contextlib import contextmanager
from os.path import join
import torch

__tab_num = 0
@contextmanager
def timer(text:str, n=3, **kargs):
    global __tab_num
    start = time()
    yield
    print(f'{("   "*(__tab_num))}{("└"*(__tab_num!=0))}{text.format(**kargs)}: {time()-start:.{n}f}초')

@contextmanager
def parent_timer(text:str, n=3, **kargs):
    global __tab_num
    __tab_num += 1
    print(("----"*__tab_num) + text.format(**kargs))
    start = time()
    yield
    print(("----"*__tab_num)+f">{text.format(**kargs)}: {time()-start:.{n}f}초")
    __tab_num -= 1
def print_log(text:str, **kargs):
    print(("    "*__tab_num)[:-1]+("─"*(__tab_num!=0))+text.format(**kargs))

def get_valid_posnumber(log:str, return_type:int|float):
    while True:
        out = input(log)
        if out.isdecimal():
            return return_type(out)
        print("올바른 값을 입력하세요")

def ask_options(log:str, options:list[str]):
    '''options는 무조건 소문자로 해야함'''
    return input(log).strip().lower() in options

def write(s:str, base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    with writing(base_dir, *dir_and_file_names, encoding=encoding, binary=binary, plus=plus) as f:
        f.write(s)
def append(s:str, base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    with appending(base_dir, *dir_and_file_names, encoding=encoding, binary=binary, plus=plus) as f:
        f.write(s)

@contextmanager
def writing(base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    mode = 'w'
    if binary:
        mode += 'b'
    if plus:
        mode += '+'
    f = open(join(base_dir, *dir_and_file_names), mode=mode, encoding=encoding)
    yield f
    f.close()
@contextmanager
def reading(base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    mode = 'r'
    if binary:
        mode += 'b'
    if plus:
        mode += '+'
    f = open(join(base_dir, *dir_and_file_names), mode=mode, encoding=encoding)
    yield f
    f.close()
@contextmanager
def appending(base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    mode = 'a'
    if binary:
        mode += 'b'
    if plus:
        mode += '+'
    f = open(join(base_dir, *dir_and_file_names), mode=mode, encoding=encoding)
    yield f
    f.close()
