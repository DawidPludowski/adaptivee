# Adaptivee: Adaptive Ensemble for Tabular Data

This repository is the official implementation of Adaptivee: Adaptive Ensemble for Tabular Data.

## Experiments reproduction

All epxeriments were made on Linux (Ubuntu) environment. There is no guarantee that the code is executable using different OS.

```{bash}
#!/bin/bash

# liltab dependency is not published on PyPi 
cd ..
git clone https://github.com/DawidPludowski/liltab
cd liltab
pip install .
cd ./../adaptivee

conda create --name=apativee python=3.10.14
conda activate adaptivee
pip install -r requirements-dev.txt

export PYTHONPATH=`pwd`

./scripts/download_data.sh # multitab not included
./scripts/split_data.sh
./prepare_pretraining_weights.sh
./create_baselines.sh
./pretrain_encoder.sh
./scripts/run_experiments.sh
```

The results are saved in `report` directory. all visualization was made using notebooks from `explore` directory.

## Example of usage

For example of usage please check `usage_example.ipynb` notebook.
