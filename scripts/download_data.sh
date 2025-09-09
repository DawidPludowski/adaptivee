#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=ada-d
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gamil.com
#SBATCH --output=/mnt/evafs/faculty/home/dpludowski/code/adaptivee/logs/download-data.log

source .venv/bin/activate

export PYTHONPATH=`pwd`

# CC18
# python bin/data/download_openml_data.py

# Tabzilla
cd resources/tabzilla
# python tabzilla_data_preprocessing.py --process_all
python restructure_datasets.py
cd ../..

# Multitab
## downloaded as a repo from https://huggingface.co/datasets/LGAI-DILab/Multitab

export PYTHONPATH=`pwd`

i_frac=0.50
i_train_frac=0.33
e_train_frac=0.8
seed=123


python bin/data/split_data.py \
    --inference-frac $i_frac \
    --inference-train-frac $i_train_frac \
    --encoder-train-frac $e_train_frac \
    --seed $seed \
    --input-path resources/liltab/raw \
    --output-path resources/liltab/split \
    --encoder-outer-split

python bin/data/split_data.py \
    --inference-frac $i_frac \
    --inference-train-frac $i_train_frac \
    --encoder-train-frac $e_train_frac \
    --seed $seed \
    --input-path resources/tabzilla/datasets_restructured \
    --output-path resources/tabzilla/datasets_split \
    --encoder-outer-split

python bin/data/split_data.py \
    --inference-frac $i_frac \
    --inference-train-frac $i_train_frac \
    --encoder-train-frac $e_train_frac \
    --seed $seed \
    --input-path resources/Multitab/data \
    --output-path resources/Multitab/data-split \
    --encoder-outer-split