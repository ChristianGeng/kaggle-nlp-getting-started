# from . import get_project_root

import os

import audeer

import re

import joblib
import numpy as np
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils.utils import get_project_root
from utils.utils import get_test_df
from utils.utils import get_train_df

PROJECT_ROOT = get_project_root()


MODEL_NAME = "sklearn_classify"

logger = logging.getLogger(__name__)
logger.info("making final data set from raw data")
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def clean(text):
    res = re.sub(
        r"http(s)?:\/\/([\w\.\/])*", " ", text
    )  # clean url:  http://x.x.x.x/xxx
    res = re.sub("[0-9]+", "", res)  # clean numbers
    res = re.sub(
        r'[!"#$%&()*+,-./:;=?@\\^_`"~\t\n\<\>\[\]\{\}]', " ", res
    )  # clean special chars
    res = re.sub(r"  +", " ", res)  #  multiple blank chars to a single blank char 。
    return res.strip()


def main():

    # sklearn_classify
    df = get_train_df()
    dft = get_test_df()

    print("cleaning data")
    df["text"] = df["text"].apply(clean)

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier(max_iter=2000, tol=5e-4)),
        ]
    )

    param_grid = {"clf__max_iter": [2000, 3000, 4000], "clf__tol": [1e-2, 1e-3, 1e-4]}

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    grid_search.fit(df["text"], df["target"])
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values("mean_test_score", ascending=False)
    cv_results[
        ["mean_test_score", "std_test_score", "param_clf__max_iter", "param_clf__tol"]
    ].head(5)

    print("Best params:")
    print(grid_search.best_params_)
    print(f"Internal CV score: {grid_search.best_score_:.3f}")

    audeer.mkdir(os.path.join(get_project_root(), "models"))
    model_dump_fn = os.path.join(get_project_root(), "models", MODEL_NAME + ".pkl")
    print(f"Dumping Model to {model_dump_fn}.")
    joblib.dump(grid_search, model_dump_fn)
    print(f"done.")

    # # predictions
    # dft['text'] = dft['text'].apply(clean)
    # predictions = grid_search.predict(dft['text'])
    # submission_df = {"id":dft['id'], "target":predictions}
    # submission = pd.DataFrame(submission_df)
    # submission.to_csv('submission.csv',index=False)


if __name__ == "__main__":
    main()
