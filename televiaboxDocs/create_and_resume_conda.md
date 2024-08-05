# in the original computer

```
conda create --name iqa_se2c3

conda activate env_name

conda env export > environment.yaml

```

# to the new computer

```
conda env create -f environment.yaml

conda activate myenv

```