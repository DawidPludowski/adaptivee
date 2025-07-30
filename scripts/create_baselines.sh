export INPUTDIR="resources/liltab/split/inference/train"
sbatch --export=ALL scripts/create_baselines.slurm

export INPUTDIR="resources/Multitab/data-split/inference/train"
sbatch --export=ALL scripts/create_baselines.slurm

export INPUTDIR="resources/tabzilla/datasets_split/inference/train"
sbatch --export=ALL scripts/create_baselines.slurm