import logging
import os
import re

import audeer
import joblib
import numpy as np
import pandas as pd
from pandas._libs.lib import u8max
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from models.train_model_sklearn_classify import clean
from utils.utils import get_project_root
from utils.utils import get_test_df
from utils.utils import get_train_df

PROJECT_ROOT = get_project_root()
MODEL_NAME = "sklearn_classify"

logger = logging.getLogger(__name__)
logger.info("making final data set from raw data")
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def main():
    model_dump_fn = os.path.join(get_project_root(), "models", MODEL_NAME + ".pkl")
    print(f"Loading Model to {model_dump_fn}.")
    grid_search = joblib.load(model_dump_fn)
    print(f"done.")
    logger.info("info")
    logger.debug("debug")

    df_test = get_test_df()

    df_test["text"] = df_test["text"].apply(clean)
    predictions = grid_search.predict(df_test["text"])
    submission_df = {"id": df_test["id"], "target": predictions}
    submission = pd.DataFrame(submission_df)

    audeer.mkdir(os.path.join(get_project_root(), "submissions"))
    submission_fname = os.path.join(
        get_project_root(), "submissions", "submission_" + MODEL_NAME + ".csv"
    )

    logger.info(f"writing submissions to {submission_fname}.")
    submission.to_csv(submission_fname, index=False)
    logger.info("done.")


if __name__ == "__main__":
    main()
