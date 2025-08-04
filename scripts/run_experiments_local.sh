#!/bin/bash

# source .venv/bin/activate
# export PYTHONPATH=`pwd`

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-17:13:53/model_checkpoints/model-epoch=9999-val_loss=-0.31.ckpt"
# export out_path=results-analysis/final/small
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter=simple
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-17:13:53/model_checkpoints/model-epoch=9999-val_loss=-0.31.ckpt"
export out_path=results-analysis/final/small
export model_list_id="SIMPLE-1"
export alpha="1.0"
export reweighter=simple
export n_iter=10

python bin/analysis/run_experiments.py \
    --data-dir $data_dir \
    --encoder-path $encoder_path \
    --out-path $out_path \
    --model-list-id $model_list_id \
    --alpha $alpha \
    --reweighter $reweighter \
    --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter=simple
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter=simple
# export n_iter=100

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter="direction-0.1"
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter="direction-0.5"
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="1.0"
# export reweighter=simple
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="1.0"
# export reweighter="direction-0.1"
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="1.0"
# export reweighter="direction-0.5"
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="resources/models/model_2025-08-01_13-47-03.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter=simple
# export n_iter=10

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter $n_iter

# export data_dir=resources/liltab/split/inference
# export encoder_path="results/07-30-2025-17:13:53/model_checkpoints/model-epoch=9999-val_loss=-0.31.ckpt"
# export out_path=results-analysis/final/big
# export model_list_id="SIMPLE-1"
# export alpha="inf"
# export reweighter=simple
# export n_iter=1

# python bin/analysis/run_experiments.py \
#     --data-dir $data_dir \
#     --encoder-path $encoder_path \
#     --out-path $out_path \
#     --model-list-id $model_list_id \
#     --alpha $alpha \
#     --reweighter $reweighter \
#     --n-iter 100