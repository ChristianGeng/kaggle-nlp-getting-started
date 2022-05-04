from pathlib import Path
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logger.info("making final data set from raw data")
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def get_project_root() -> Path:
    # return Path(__file__).parent.parent
    return Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


def get_train_df() -> Path:
    return pd.read_csv(os.path.join(get_project_root(), "data", "raw", "train.csv"))


def get_test_df() -> Path:
    return pd.read_csv(os.path.join(get_project_root(), "data", "raw", "test.csv"))


def get_samplesubmission_df() -> Path:
    return pd.read_csv(
        os.path.join(get_project_root(), "data", "raw", "sample_submission.csv")
    )


def write_submission_csv(submission, model_name, write_index=False):
    """Write a submission to the submission directory"""

    submission_fname = os.path.join(
        get_project_root(), "submissions", "submission_" + model_name + ".csv"
    )

    logger.info(f"writing submissions to {submission_fname}.")
    submission.to_csv(submission_fname, index=write_index)
    logger.info("done.")
