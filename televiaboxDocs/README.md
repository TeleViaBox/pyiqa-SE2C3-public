

commands:

- train
```
(base) televiabox@LAPTOP-A4BFENS0:~/publication_iqa$ python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN.yml
```
```
python pyiqa/train.py -opt options/train/SE2C3/train_SE2C3_e.yml
```


- validate
```
(base) televiabox@LAPTOP-A4BFENS0:~/publication_iqa$ pyiqa nima --target ./ResultsCalibra/dist_dir/I06.bmp --ref ./ResultsCalibra/ref_dir/
```

# describe my method: SE2C3: SE(2) Group with charge-conserving ConvNet
