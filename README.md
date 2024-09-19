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

python bin/download_openml_data.py
python bin/prepare_liltab_data.py
python bin/pretrain_encoder.py
python bin/run_analysis.py # path to the liltab model should be updated
```

The results are saved in `report` directory. all visualization was made using notebooks from `explore` directory.

## Example of usage

For example of usage please check `usage_example.ipynb` notebook.
