from pathlib import Path
import pandas as pd

def get_data(path: str) -> pd.DataFrame:
    path = Path(path)
    
    train_path = path / 'train'
    test_path = path / 'test'
    
    for data_path in train_path.glob('*.csv'):
        dataname = data_path.stem
        
        train_df = pd.read_csv(train_path / f'{dataname}.csv')
        test_df = pd.read_csv(test_path / f'{dataname}.csv')
        
        yield train_df, test_df, dataname
    