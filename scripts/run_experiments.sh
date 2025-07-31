#!/bin/bash

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-17:13:53/model_checkpoints/model-epoch=9999-val_loss=-0.31.ckpt"
export out_path=results-analysis/final/small
export model_list_id="SIMPLE-1"
export alpha="inf"
export reweighter=simple
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-17:13:53/model_checkpoints/model-epoch=9999-val_loss=-0.31.ckpt"
export out_path=results-analysis/final/small
export model_list_id="SIMPLE-1"
export alpha="1.0"
export reweighter=simple
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="inf"
export reweighter=simple
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="inf"
export reweighter=simple
export n_iter=100

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="inf"
export reweighter="direction-0.1"
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="inf"
export reweighter="direction-0.5"
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="1.0"
export reweighter=simple
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="1.0"
export reweighter="direction-0.1"
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm

export data_dir=resources/liltab/split/inference
export encoder_path="results/07-30-2025-21:49:05/model_checkpoints/last.ckpt"
export out_path=results-analysis/final/big
export model_list_id="SIMPLE-1"
export alpha="1.0"
export reweighter="direction-0.5"
export n_iter=10

sbatch --export=ALL ./scripts/run_experiments.slurm