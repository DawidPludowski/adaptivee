# Adaptivee

Adaptivee is an adaptive ensembler framework that assign models weights for each observation separately.

## Experiments reproduction

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

## Example of usage

For example of usage please check `usage_example.ipynb` notebook.
