from pathlib import Path
import pandas as pd
import os


def get_project_root() -> Path:
    # return Path(__file__).parent.parent
    return Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


def get_train_df() -> Path:
    return pd.read_csv(os.path.join(get_project_root(), "data", "raw", "train.csv"))


def get_test_df() -> Path:
    return pd.read_csv(os.path.join(get_project_root(), "data", "raw", "test.csv"))
