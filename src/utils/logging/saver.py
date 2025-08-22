from torch import save
import multiprocessing as mp

def is_pickleable(obj):
    import pickle
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        print(f"Cannot pickle {type(obj)}: {e}")
        return False

def save_worker(save_queue: mp.JoinableQueue):
    while True:
        item = save_queue.get()
        if item is None:
            break
        is_pickleable(item[0])
        save(item[0].state_dict(), item[1])
        save_queue.task_done()

class Saver:
    def __init__(self, num_workers=2):
        from os import cpu_count
        if num_workers > cpu_count():
            num_workers = cpu_count()-1
        if num_workers < 1:
            num_workers = 1
        self.num_workers = num_workers
        self.save_queue = mp.JoinableQueue()
        self.workers = [mp.Process(target=save_worker, args=(self.save_queue,), daemon=True) for _ in range(num_workers)]
        for worker in self.workers:
            worker.start()
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Saver, cls).__new__(cls)
        return cls.instance
    
    def __call__(self, state_dict_able, path):
        self.save_queue.put((state_dict_able, path))

    def close(self):
        for _ in self.workers:
            self.save_queue.put(None)
        self.save_queue.join()

    def __del__(self):
        self.close()