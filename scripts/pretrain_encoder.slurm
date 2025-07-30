#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=ada-pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/adaptivee/logs/pretrain-encoder-%A_%a.log

source .venv/bin/activate
export PYTHONPATH=`pwd`

# python bin/models/pretrain_encoder.py \
#     --train-path resources/liltab/split/encoder/train \
#     --val-path resources/liltab/split/encoder/test \
#     --model-list-id "SIMPLE-1" \
#     --alpha-name "alpha=1.0" \
#     --n-hidden-layers 8 \
#     --hidden-size 32 \
#     --n-epochs 10000

python bin/models/pretrain_encoder.py \
    --train-path resources/tabzilla/datasets_split/encoder/train \
    --val-path resources/tabzilla/datasets_split/encoder/test \
    --model-list-id "SIMPLE-1" \
    --alpha-name "alpha=1.0" \
    --n-hidden-layers 16 \
    --hidden-size 64 \
    --n-epochs 10000 \
    --out-path "results/tabzilla-alpha=1.0"

# python bin/models/pretrain_encoder.py \
#     --train-path resources/liltab/split/encoder/train \
#     --val-path resources/liltab/split/encoder/test \
#     --model-list-id "SIMPLE-1" \
#     --alpha-name "onehot" \
#     --n-hidden-layers 16 \
#     --hidden-size 64 \
#     --n-epochs 10000

# python bin/models/pretrain_encoder.py \
#     --train-path resources/liltab/split/encoder/train \
#     --val-path resources/liltab/split/encoder/test \
#     --model-list-id "SIMPLE-1" \
#     --alpha-name "alpha=1.0" \
#     --n-hidden-layers 16 \
#     --hidden-size 64 \
#     --n-epochs 10000
