from ..dev_utils import write, append
import threading
from statistics import mean
import os.path as path
import datetime

class Logger:
    def __init__(self, logs_dir):
        self.dir = logs_dir
        self.logs = []
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file = f'{now}.txt'

    def append(self, log:str):
        self.logs.append(log)
        
    def appends(self, *logs:str):
        self.logs.extend(logs)
        
    def write(self, log:str):
        threading.Thread(target=append, args=(log+'\n',self.dir, self.file)).start()

    def writes(self, *logs:str):
        threading.Thread(target=append, args=(("\n".join(logs))+'\n',self.dir, self.file)).start()
            
    def close(self):
        if self.logs is not None:
            append("\n".join(self.logs), self.dir, self.file)
