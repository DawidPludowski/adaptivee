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

python bin/analysis/run_experiments.py \
    --data-dir $data_dir \
    --encoder-path $encoder_path \
    --out-path $out_path \
    --model-list-id $model_list_id \
    --alpha $alpha \
    --reweighter $reweighter \
    --n-iter $n_iter