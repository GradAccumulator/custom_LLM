import threading
threads:list[threading.Thread] = []

def start(target, *args, **kwargs):
    global threads
    t = threading.Thread(target=target, args=args, kwargs=kwargs)
    threads.append(t)
    return t.start()

def join_all():
    for t in threads:
        t.join()