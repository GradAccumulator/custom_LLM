import os

def print_tree(path, prefix=""):
    items = sorted(os.listdir(path))
    
    # 폴더와 파일 분리
    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
    files = [item for item in items if not os.path.isdir(os.path.join(path, item))]
    
    # 폴더 먼저 출력
    entries = dirs + files
    
    pointers = ['├── '] * (len(entries) - 1) + ['└── ']
    
    for pointer, item in zip(pointers, entries):
        full_path = os.path.join(path, item)
        print(prefix + pointer + item)
        if os.path.isdir(full_path):
            extension = '│   ' if pointer == '├── ' else '    '
            print_tree(full_path, prefix + extension)

root_path = os.path.dirname(os.path.abspath(__file__))
print("custom_LLM")
print_tree(root_path)
#python draw_tree.py > folder_tree.txt