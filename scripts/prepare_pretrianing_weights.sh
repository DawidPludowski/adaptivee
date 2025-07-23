source .venv/bin/activate

export PYTHONPATH=`pwd`

alphas=(
    1
    0.1
    10
    inf
)
model_ids=("SIMPLE-1")

input_paths=(
    "resources/liltab/split/encoder/test"
    "resources/liltab/split/encoder/train"
    "resources/liltab/split/inference/test"
    "resources/liltab/split/inference/train"
    "resources/Multitab/data-split/encoder/test"
    "resources/Multitab/data-split/encoder/train"
    "resources/Multitab/data-split/inference/test"
    "resources/Multitab/data-split/inference/train"
    "resources/tabzilla/datasets_split/encoder/test"
    "resources/tabzilla/datasets_split/encoder/train"
    "resources/tabzilla/datasets_split/inference/test"
    "resources/tabzilla/datasets_split/inference/train"
)

export ALPHAS="${alphas[*]}"
export MODELS="${model_ids[*]}"
export INPUTS="${input_paths[*]}"

n_tasks=(( ${#alphas[@]} * ${#model_ids[@]} * ${#input_paths[@]} ))

export ALPHAS MODELS INPUTS

sbatch --export=ALL --array=0-$((n_tasks - 1)) bin/data/prepare_pretraining_weights.slurm