```
(base) televiabox@LAPTOP-A4BFENS0:~/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3$ python pyiqa/train.py -opt options/train/SE2C3/train_SE2C3_e.yml 
Reading model file: dbcnn_model
Reading model file: wadiqam_model
Reading model file: se2c3_model
Reading model file: base_model
Reading model file: inference_model
Reading model file: general_iqa_model
Reading model file: distiqa_model
Reading model file: bapps_model
Reading model file: pieapp_model
Disable distributed.
Path already exists. Rename it to /home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/experiments/001_SE2C3_LIVEC_archived_20240611_163608
Path already exists. Rename it to /home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/tb_logger_archived/001_SE2C3_LIVEC_archived_20240611_163608
2024-06-11 16:36:09,016 INFO: Dataset [LIVEChallengeDataset] - livechallenge is built.
2024-06-11 16:36:09,017 INFO: Training statistics:
        Number of train images: 930
        Dataset enlarge ratio: 1
        Batch size per gpu: 8
        World size (gpu number): 1
        Require iter number per epoch: 117
        Total epochs: 1710; iters: 200000.
2024-06-11 16:36:09,018 INFO: Dataset [LIVEChallengeDataset] - livechallenge is built.
2024-06-11 16:36:09,018 INFO: Number of val images/folders in livechallenge: 232
Building model with options: OrderedDict({'name': '001_SE2C3_LIVEC', 'model_type': 'SE2C3Model', 'num_gpu': 1, 'manual_seed': 123, 'split_num': 10, 'save_final_results_path': './experiments/SE2C3_LIVEChallenge_10splits_results.txt', 'datasets': OrderedDict({'train': OrderedDict({'name': 'livechallenge', 'type': 'LIVEChallengeDataset', 'dataroot_target': './datasets/LIVEC/', 'meta_info_file': './datasets/meta_info/meta_info_LIVEChallengeDataset.csv', 'split_file': './datasets/meta_info/livechallenge_seed123.pkl', 'split_index': 2, 'augment': OrderedDict({'hflip': True, 'random_crop': 448}), 'img_range': 1, 'use_shuffle': True, 'num_worker_per_gpu': 4, 'batch_size_per_gpu': 8, 'dataset_enlarge_ratio': 1, 'prefetch_mode': None, 'phase': 'train'}), 'val': OrderedDict({'name': 'livechallenge', 'type': 'LIVEChallengeDataset', 'dataroot_target': './datasets/LIVEC', 'meta_info_file': './datasets/meta_info/meta_info_LIVEChallengeDataset.csv', 'split_file': './datasets/meta_info/livechallenge_seed123.pkl', 'split_index': 2, 'phase': 'val'})}), 'network': OrderedDict({'type': 'SE2C3Net', 'in_channels': 3, 'num_classes': 1}), 'path': OrderedDict({'pretrain_network_g': None, 'strict_load_g': True, 'resume_state': None, 'auto_resume': True, 'experiments_root': '/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/experiments/001_SE2C3_LIVEC', 'models': '/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/experiments/001_SE2C3_LIVEC/models', 'training_states': '/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/experiments/001_SE2C3_LIVEC/training_states', 'log': '/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/experiments/001_SE2C3_LIVEC', 'visualization': '/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/experiments/001_SE2C3_LIVEC/visualization'}), 'train': OrderedDict({'optim': OrderedDict({'type': 'Adam', 'lr': 0.001, 'weight_decay': '1e-4'}), 'scheduler': OrderedDict({'type': 'MultiStepLR', 'milestones': [50000, 100000, 150000], 'gamma': 0.5}), 'cri_mos_opt': OrderedDict({'type': 'MSELoss'}), 'cri_metric_opt': OrderedDict({'type': 'PLCCLoss'}), 'total_iter': 200000, 'val_freq': 1000, 'save_checkpoint_freq': 5000, 'mos_loss_opt': OrderedDict({'type': 'MSELoss', 'loss_weight': 1.0, 'reduction': 'mean'})}), 'val': OrderedDict({'val_freq': 100.0, 'save_img': False, 'pbar': True, 'key_metric': 'srcc', 'metrics': OrderedDict({'srcc': OrderedDict({'type': 'calculate_srcc'}), 'plcc': OrderedDict({'type': 'calculate_plcc'}), 'krcc': OrderedDict({'type': 'calculate_krcc'})})}), 'logger': OrderedDict({'print_freq': 10, 'save_checkpoint_freq': 5000000000.0, 'save_latest_freq': 500.0, 'use_tb_logger': True, 'wandb': OrderedDict({'project': None, 'resume_id': None})}), 'dist_params': OrderedDict({'backend': 'nccl', 'port': 29500}), 'dist': False, 'rank': 0, 'world_size': 1, 'auto_resume': False, 'is_train': True, 'root_path': '/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3'})
Registered models in MODEL_REGISTRY: ['GeneralIQAModel', 'DBCNNModel', 'WaDIQaMModel', 'SE2C3Model', 'DistIQAModel', 'BAPPSModel', 'PieAPPModel']
2024-06-11 16:36:09,022 INFO: Network [SE2C3Net] is created.
2024-06-11 16:36:10,143 INFO: Network: SE2C3Net, with parameters: 796,929
2024-06-11 16:36:10,143 INFO: SE2C3Net(
  (layer1): SE2C3Conv(
    (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (layer2): SE2C3Conv(
    (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (layer3): SE2C3Conv(
    (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (layer4): SE2C3Conv(
    (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (fc): Linear(in_features=131072, out_features=1, bias=True)
)
2024-06-11 16:36:10,146 INFO: Network [SE2C3Net] is created.
2024-06-11 16:36:10,148 INFO: Loss [MSELoss] is created.
Traceback (most recent call last):
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/train.py", line 258, in <module>
    train_pipeline(root_path)
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/train.py", line 146, in train_pipeline
    model = build_model(opt)
            ^^^^^^^^^^^^^^^^
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/__init__.py", line 33, in build_model
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/se2c3_model.py", line 19, in __init__
    super(SE2C3Model, self).__init__(opt)
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/general_iqa_model.py", line 33, in __init__
    self.init_training_settings()
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/se2c3_model.py", line 22, in init_training_settings
    super(SE2C3Model, self).init_training_settings()
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/general_iqa_model.py", line 54, in init_training_settings
    self.setup_optimizers()
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/general_iqa_model.py", line 101, in setup_optimizers
    self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/televiabox/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3/pyiqa/models/base_model.py", line 122, in get_optimizer
    optimizer = optim_class(params, lr, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/televiabox/miniconda3/lib/python3.12/site-packages/torch/optim/adam.py", line 38, in __init__
    if not 0.0 <= weight_decay:
           ^^^^^^^^^^^^^^^^^^^
TypeError: '<=' not supported between instances of 'float' and 'str'
(base) televiabox@LAPTOP-A4BFENS0:~/publication_iqa/fromgithub/fromzip/pyiqa-SE2C3$ python pyiqa/train.py -opt options/train/SE2C3/train_SE2C3_e.yml 
```