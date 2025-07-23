source .venv/bin/activate

export PYTHONPATH=`pwd`

# CC18
python bin/data/download_openml_data.py

# Tabzilla
cd resources/tabzilla
python tabzilla_data_preprocessing.py --process_all
python restructure_data.py
cd ../..

# Multitab
## downloaded as a repo from https://huggingface.co/datasets/LGAI-DILab/Multitab