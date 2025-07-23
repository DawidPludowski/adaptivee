---
configs:
- config_name: default
  data_files:
  - split: train
    path: multitab_stats_logs.csv
---

# Multitab: A Comprehensive Benchmark Suite for Multi-Dimensional Evaluation in Tabular Domains

This repository hosts the core benchmark **data** from the MULTITAB benchmark suite, a large-scale, structured evaluation of tabular learning algorithms. 
The full benchmark includes:

- 📂 Preprocessed `.npz` datasets (optional, for fast loading)
- 📊 A single **summary CSV file** (`multitab_stats_logs.csv`) with:
  - Model performance across datasets
  - Dataset-level statistical properties
- 🗂 Additional Log Archives
  - 📦 `optimization_logs.zip`: Contains raw optimization outputs for all model–dataset combinations.
    - Each file includes:
      - Validation performance per trial
      - Best hyperparameter configurations
      - Metadata such as time, seed, and trial count
  - 📦 `reproduction_logs.zip`: Logs from reproduction runs under fixed hyperparameter settings.
    - Useful for verifying benchmark consistency and computing final ranks.
    - Includes:
      - Full prediction outputs
      - Final evaluation metrics per split

> To use these files, download and extract locally.  
> Refer to the GitHub repository for code that parses and processes the logs.


**⚠️ This dataset page only contains the data artifacts.  
For full implementation code, training pipelines, and model optimization scripts, please refer to our GitHub repository:  
👉 [https://github.com/kyungeun-lee/multitab](https://github.com/kyungeun-lee/multitab)**

---

## Overview

MULTITAB is designed to facilitate data-aware benchmarking by evaluating 13 diverse tabular models across 196 datasets from OpenML. 
Instead of relying on aggregate scores, this benchmark focuses on how model performance varies with dataset characteristics such as:

- Task types
- Sample size
- Feature heterogeneity
- Feature-to-sample ratio
- Label imbalance
- Function irregularity
- Feature interaction

Each model is optimized with a consistent hyperparameter tuning budget and evaluated via stratified cross-validation. 
Results are normalized per dataset to enable fair comparisons.

---

## Data Files

### 🔹 `multitab_stats_logs.csv`

This file is the **main table** for comparative and statistical analysis. Each row corresponds to one dataset and includes:

- **Model performance columns**:
  - `{MODEL}_{METRIC}_{RAW METRIC FOR CLASSIFICATION}_{RAW METRIC FOR REGERSSION}`: e.g., `FTT_error_logloss_rmse`, `XGBoost_rank_acc_rmse`
  - METRIC includes average **normalized predictive error** (as described in the main text), and average **ranks**.
  - RAW METRIC FOR CLASSIFICATION includes **log loss** (as the main metric in the suite), and **accuracy**.
  - RAW METRIC FOR REGRESSION includes **RMSE** only.
- **Dataset statistical properties**:
  - `stats_task_type`, `stats_sample_size`, `stats_num_features`, `stats_imbalance_factor`, etc.

## Column Description

| Column Category       | Description |
|------------------------|-------------|
| `data_id`, `data_name` | OpenML identifier and dataset name |
| `Model_*` columns      | Performance metrics for each model. Includes RMSE/log loss and rank-based metrics (e.g., `MLP_rank_logloss_rmse`) |
| `stats_*` columns      | Dataset-level statistical properties (e.g., `stats_num_features`, `stats_entropy_ratio`, `stats_skewness`, etc.) |
| `subcategory_*` columns | Data regime classification based on specific criteria (see Table 1 in the paper). |

A complete list of all 80+ columns is available in the paper and can also be printed via:

```python
import pandas as pd
df = pd.read_csv("multitab_stats_logs.csv")
print(df.columns.tolist())
```

### 📁 `.npz` files (optional)

Each dataset is also available in compressed numpy format for quick loading in research workflows.

Contents of each `.npz` file:
- `X`: Feature matrix
- `y`: Target
- `X_cat`, `X_cat_cardinality`, `X_num`: Column indices and metadata

---

