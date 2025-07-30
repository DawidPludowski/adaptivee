#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=ada-exp
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/adaptivee/logs/run-experiments-%A_%a.log

source .venv/bin/activate
export PYTHONPATH=`pwd`

# liltab small
python bin/analysis/run_experiments.py \
    --data-dir resources/liltab/split/inference \
    --encoder-path results/07-30-2025-17:13:53/model_checkpoints/model-epoch=9999-val_loss=-0.31.ckpt \
    --out-path results-analysis/initial \
    --model-list-id "SIMPLE-1" \
    --alpha "inf" \
    --reweighter simple \
    --n-iter 10

# liltab big
python bin/analysis/run_experiments.py \
    --data-dir resources/liltab/split/inference \
    --encoder-path results/07-30-2025-21:49:05/model_checkpoints/last.ckpt \
    --out-path results-analysis/initial \
    --model-list-id "SIMPLE-1" \
    --alpha "inf" \
    --reweighter simple \
    --n-iter 10

# tabzilla big
python bin/analysis/run_experiments.py \
    --data-dir resources/tabzilla/datasets_split/inference \
    # --encoder-path results/07-30-2025-21:49:05/model_checkpoints/last.ckpt \
    --out-path results-analysis/initial \
    --model-list-id "SIMPLE-1" \
    --alpha "inf" \
    --reweighter simple \
    --n-iter 10
