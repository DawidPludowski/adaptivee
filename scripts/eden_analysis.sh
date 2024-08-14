#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=lasts
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski.stud@pw.edu.pl
#SBATCH --output=/mnt/evafs/faculty/home/dpludowski/code/adaptivee/eden.log

. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh
conda activate adaptivee

export PYTHONPATH=`pwd`

python bin/run_analysis.py