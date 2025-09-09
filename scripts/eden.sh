#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=lasts
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/faculty/home/dpludowski/code/adaptivee/eden-1.log

. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh
conda activate adaptivee

export PYTHONPATH=`pwd`

python bin/run_analysis.py