from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from collections import deque
from os.path import join
from glob import glob
from os import remove


class TBoardManager:
    def __init__(self, logs_dir:str,*, window_size=20):
        self.dir = logs_dir
        self.del_prev_files()
        self.writer = SummaryWriter(self.dir)
        self.window_size = window_size
        self.logs_buffer = {}

    def del_prev_files(self):
        for f in glob(join(self.dir, 'events.*')):
            remove(f)

    def close(self):
        self.writer.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def add_scalars(self, *args:list[tuple[str,float,int, bool]]):
        '''args는 (tag, value, step, moving기록여부)순으로 전달해야함'''
        for (tag, value, step, moving) in args:
            self.add_scalar(tag, value, step, moving)

    def add_scalar(self, tag, value, step, moving=True):
        '''****나중에 보기'''
        self.writer.add_scalar(tag, value, step)

        if moving:
            if tag not in self.logs_buffer.keys():
                self.logs_buffer[tag] = deque(maxlen=self.window_size)
            self.logs_buffer[tag].append((step, value))
            buffer = self.logs_buffer[tag]
            if len(buffer) == 0:
                return

            values = [v for _, v in buffer]
            moving_avg = mean(values)
            latest_step = buffer[-1][0]

            self.writer.add_scalar(f"moving/{tag}", moving_avg, latest_step)


