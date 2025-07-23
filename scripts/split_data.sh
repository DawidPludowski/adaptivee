source .venv/bin/activate

export PYTHONPATH=`pwd`

i_frac=0.50
i_train_frac=0.33
e_train_frac=0.8
seed=123


python bin/data/split_data.py \
    --inference-frac $i_frac \
    --inference-train-frac $i_train_frac \
    --encoder-train-frac $e_train_frac \
    --seed $seed \
    --input-path resources/liltab/raw \
    --output-path resources/liltab/split \
    --encoder-outer-split

python bin/data/split_data.py \
    --inference-frac $i_frac \
    --inference-train-frac $i_train_frac \
    --encoder-train-frac $e_train_frac \
    --seed $seed \
    --input-path resources/tabzilla/datasets_restructured \
    --output-path resources/tabzilla/datasets_split \
    --encoder-outer-split

python bin/data/split_data.py \
    --inference-frac $i_frac \
    --inference-train-frac $i_train_frac \
    --encoder-train-frac $e_train_frac \
    --seed $seed \
    --input-path resources/Multitab/data \
    --output-path resources/Multitab/data-split \
    --encoder-outer-split