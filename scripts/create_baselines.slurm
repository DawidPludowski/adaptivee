#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=ada-baseline
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/faculty/home/dpludowski/code/adaptivee/logs/des-baselines-%A_%a.log

source .venv/bin/activate

echo "Start $INPUTDIR"

for filename in "$INPUTDIR"/*; do
    echo "start $filename"
    train_path=$filename
    test_path="${train_path/train/test}"
    python bin/models/des_baselines.py \
        --train-path $train_path \
        --test-path $test_path \
        --model-list-id "SIMPLE-1" \
        --save-out
done
