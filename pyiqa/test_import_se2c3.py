# import importlib
# import sys
# import traceback
# from os import path as osp

# # 确保当前目录和父目录在 sys.path 中
# current_dir = osp.dirname(osp.abspath(__file__))
# parent_dir = osp.dirname(current_dir)
# sys.path.append(current_dir)
# sys.path.append(parent_dir)

# print("Current directory:", current_dir)
# print("Parent directory:", parent_dir)
# print("System path:", sys.path)

# # 尝试直接导入 se2c3_model 文件
# try:
#     module_name = 'pyiqa.models.se2c3_model'
#     print(f"Attempting to import {module_name}")
#     module = importlib.import_module(module_name)
#     print(f'Successfully imported {module_name}')
#     print(f"Module content: {dir(module)}")
# except ModuleNotFoundError as e:
#     print(f'Failed to import {module_name}: {e}')
# except Exception as e:
#     print(f'An error occurred: {e}')
#     traceback.print_exc()



"""
(base) televiabox@LAPTOP-A4BFENS0:~/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa$ python test_import_se2c3.py
SE2C3Model 註冊失敗
Traceback (most recent call last):
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/test_import_se2c3.py", line 53, in <module>
    model = build_model(opt)
            ^^^^^^^^^^^^^^^^
  File "/home/televiabox/miniconda3/lib/python3.12/site-packages/pyiqa/models/__init__.py", line 27, in build_model
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/televiabox/miniconda3/lib/python3.12/site-packages/pyiqa/utils/registry.py", line 65, in get
    raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
KeyError: "No object named 'SE2C3Model' found in 'model' registry!"
(base) televiabox@LAPTOP-A4BFENS0:~/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa$ 
"""
# 導入 models 模塊，這會觸發 models/__init__.py 文件中的代碼執行
import pyiqa.models
# 檢查 SE2C3Model 是否註冊成功
from pyiqa.utils.registry import MODEL_REGISTRY
if 'SE2C3Model' in MODEL_REGISTRY:
    print('SE2C3Model 註冊成功')
else:
    print('SE2C3Model 註冊失敗')
# 如果註冊成功，嘗試構建模型
from pyiqa.models import build_model
opt = {
    'model_type': 'SE2C3Model',
    # 其他必要的配置選項
}
model = build_model(opt)
print(model)
