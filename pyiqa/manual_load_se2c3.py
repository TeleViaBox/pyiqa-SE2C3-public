import sys
from os import path as osp

# 确保当前目录和父目录在 sys.path 中
current_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

print("Current directory:", current_dir)
print("Parent directory:", parent_dir)
print("System path:", sys.path)

# 手动读取和执行 se2c3_model.py 文件
se2c3_model_path = osp.join(current_dir, 'models', 'se2c3_model.py')
try:
    with open(se2c3_model_path, 'r') as file:
        exec(file.read())
    print(f'Successfully loaded {se2c3_model_path}')
except FileNotFoundError as e:
    print(f'File not found: {se2c3_model_path}')
except Exception as e:
    print(f'An error occurred while loading {se2c3_model_path}: {e}')
    import traceback
    traceback.print_exc()
