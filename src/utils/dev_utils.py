from time import time
from contextlib import contextmanager
from os.path import join
from typing import Callable, Any
import keyword


def is_valid_varname(s:str)->bool:
    return s.isidentifier() and not keyword.iskeyword(s)

class BoolWrapper:
    def __init__(self, value=False):
        self.flag = value
    
    def true(self):
        self.flag = True
    
    def false(self):
        self.flag = False

@contextmanager
def run_on_exit_if_true(fn:Callable[[], Any], init_value=False):
    '''
    init_value 값을 가진 BoolWrapper변수를 return함.  \n
    이후에 해당 리턴값이 True면 fn을 실행함             \n
    trigger를 true로 바꿀땐 .true()                 \n
    trigger를 false로 바꿀땐 .false()               \n
    >>> a = []
    >>> list = [1,2,3,4,5,6,7]
    >>> with run_on_exit_if_true(lambda:a.append(10)) as trigger:
            for value in list:
                if value==7:
                    trigger.true()
    >>> a
    [10]
    '''
    trigger = BoolWrapper(init_value)
    yield trigger
    if trigger.flag:
        fn()

__tab_num = 0
@contextmanager
def timer(text:str, n=3, **kargs):
    
    '''
    시간 측정을 위한 contextmanager
    >>> with timer("test"):
    ...     time.sleep(1)
    test: 1.000초
    >>> #if use parent_timer,
    >>> with parent_timer("test"):
    ...     with timer("test2"):
    ...         time.sleep(1)
    ...     with timer("test3):
    ...         time.sleep(1)
    ----test
        └test2: 1.000초
        └test3: 1.000초
    ---->test: 2.000초
    '''
    global __tab_num
    start = time()
    yield
    print(f'{("   "*(__tab_num))}{("└"*(__tab_num!=0))}{text.format(**kargs)}: {time()-start:.{n}f}초')

@contextmanager
def parent_timer(text:str, n=3, **kargs):
    
    '''
    parent_timer는 timer 여러개를 묶는 용도로 씀
    >>> with parent_timer("test"):
    ...     with timer("test2"):
    ...         time.sleep(1)
    ...     with timer("test3"):
    ...         time.sleep(1)
    ----test
        └test2: 1.000초
        └test3: 1.000초
    ---->test: 2.000초
    >>> with parent_timer("딥러닝 모델 생성 전 처리"):
    ...     with timer("데이터 불러오기"):
    ...         ...
    ...     with parent_timer("데이터 전처리"):
    ...         with timer("train x,label 분리"):
    ...             ...
    ...         with timer("데이터 셔플"):
    ...             ...
    ----딥러닝 모델 생성 전 세팅
        └데이터 불러오기: 1.000초
    --------데이터 전처리
            └train x,label 분리: 1.000초
            └데이터 셔플: 1.000초
    -------->데이터 전처리: 2.000초
    ---->딥러닝 모델 생성 전 세팅: 3.000초
    '''
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
        try:
            return return_type(out)
        except:
            print("올바른 값을 입력하세요")

def ask_options(log:str, options:list[str]):
    '''options는 무조건 소문자로 해야함'''
    return input(log).strip().lower() in options
def get_option(msg, options:list[str]):
    '''
    prints {msg} {options}: 
    '''
    while True:
        out = input(f"{msg} {options}: ").strip().lower()
        if out in options:
            return out
        print(f"{options} 안의 값을 입력하세요.")

def write(s:str, base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    with writing(base_dir, *dir_and_file_names, encoding=encoding, binary=binary, plus=plus) as f:
        f.write(s)
def append(s:str, base_dir, *dir_and_file_names, encoding='utf-8', binary=False, plus=False):
    with appending(base_dir, *dir_and_file_names, encoding=encoding, binary=binary, plus=plus) as f:
        f.write(s)
        

def alert():
    import pygame
    from time import sleep
    pygame.mixer.init()
    pygame.mixer.music.load("C:\\Users\\GIGABYTE\\Downloads\\99412B355CF6B93806.mp3")
    for _ in range(3):
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            sleep(0.1)

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
