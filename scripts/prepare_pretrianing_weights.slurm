#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=ada-w
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/faculty/home/dpludowski/code/adaptivee/logs/prepare_pretraining_weights-%A_%a.log

source .venv/bin/activate

IFS=' ' read -r -a alphas <<< "$ALPHAS"
IFS=' ' read -r -a model_ids <<< "$MODELS"
IFS=' ' read -r -a input_paths <<< "$INPUTS"

cnt=0
for path in "${input_paths[@]}"; do
    for id in "${model_ids[@]}"; do
        for alpha in "${alphas[@]}"; do
            if (( ${SLURM_ARRAY_TASK_ID} == cnt )); then
                if [[ "$alpha" == "inf" ]]; then
                    echo "path=$path, id=$id, onehot"
                    python bin/data/prepare_pretraining_weights.py \
                        --model-list-id $id \
                        --input-path $path \
                        --use-onehot
                else
                    echo "path=$path, id=$id, alpha=$alpha"
                    python bin/data/prepare_pretraining_weights.py \
                        --model-list-id $id \
                        --input-path $path \
                        --alpha $alpha
                fi
            fi
            ((cnt+=1))
        done
    done
done