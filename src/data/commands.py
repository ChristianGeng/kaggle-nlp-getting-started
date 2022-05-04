import logging
import os
import subprocess

import click
import dotenv
import nltk

from utils.utils import get_project_root

dotenv_path = os.path.join(get_project_root(), ".env")
dotenv.load_dotenv(dotenv_path)

competition = os.getenv("COMPETITION_NAME")


logger = logging.getLogger(__name__)
logger.info("making final data set from raw data")
# log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_fmt = "%(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("description")
def submit(input_filepath, description):
    logger.info(f"submitting {input_filepath}")
    command = [
        "kaggle",
        "competitions",
        "submit",
        competition,
        "-f",
        input_filepath,
        "-m",
        description,
    ]

    res = subprocess.call(command)
    logger.info(f"submitting finished with result {res}")


@click.command()
def nltk_stopwords():
    """download the nltk stopwords"""

    nltk.download("stopwords")


@click.command()
def nltk_twitter_samples():
    """download the nltk twitter samples"""

    nltk.download("twitter_samples")
