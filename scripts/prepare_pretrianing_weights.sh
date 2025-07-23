source .venv/bin/activate

export PYTHONPATH=`pwd`

alphas=(
    1
    # 0.1
    # 10
)
model_ids=("SIMPLE-1")

input_paths=(
    "resources/liltab/split/encoder/test"
    # "resources/liltab/split/encoder/train"
    # "resources/liltab/split/inference/test"
    # "resources/liltab/split/inference/train"
    # "resources/Multitab/data-split/encoder/test"
    # "resources/Multitab/data-split/encoder/train"
    # "resources/Multitab/data-split/inference/test"
    # "resources/Multitab/data-split/inference/train"
    # "resources/tabzilla/datasets_split/encoder/test"
    # "resources/tabzilla/datasets_split/encoder/train"
    # "resources/tabzilla/datasets_split/inference/test"
    # "resources/tabzilla/datasets_split/inference/train"
)

for path in "${input_paths[@]}"; do
    for id in "${model_ids[@]}"; do
        for alpha in "${alphas[@]}"; do
            echo "path=$path, id=$id, alpha=$alpha"
            python bin/data/prepare_pretraining_weights.py \
                --model-list-id $id \
                --input-path $path \
                --alpha $alpha
        done
        echo "path=$path, id=$id, onehot"
        python bin/data/prepare_pretraining_weights.py \
            --model-list-id $id \
            --input-path $path \
            --use-onehot
    done
done