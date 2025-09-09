import optuna, argparse, os, torch, json, joblib, datetime
from libs.data import TabularDataset
from libs.data import load_data
import warnings
import json
from tqdm import tqdm
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

openml_id = 43611

try:
    X, y, X_cat, X_cat_cardinality, X_num = load_data(openml_id)

    np.savez(
        f"data/{openml_id}",
        X=X,
        y=y,
        X_cat=np.array(X_cat, dtype=object),
        X_cat_cardinality=np.array(X_cat_cardinality, dtype=object),
        X_num=np.array(X_num, dtype=object),
    )
except Exception as e:
    print(f"Error processing dataset {openml_id}: {e}")
    