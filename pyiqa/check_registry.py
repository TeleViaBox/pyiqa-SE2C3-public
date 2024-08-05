import importlib
import sys
import traceback
from os import path as osp
from pyiqa.utils.registry import MODEL_REGISTRY

# 确保当前目录和父目录在 sys.path 中
current_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))

print("Current directory:", current_dir)
print("System path:", sys.path)

# 测试单个模块导入
module_name = 'pyiqa.models.se2c3_model'
try:
    print(f"Attempting to import {module_name}")
    module = importlib.import_module(module_name)
    print(f"Successfully imported {module_name}")
    print(f"Module content: {dir(module)}")
except ModuleNotFoundError as e:
    print(f'Failed to import {module_name}: {e}')
except Exception as e:
    print(f'An error occurred: {e}')
    traceback.print_exc()

# 打印所有已注册的模型
print("Registered models in MODEL_REGISTRY:", list(MODEL_REGISTRY.keys()))
