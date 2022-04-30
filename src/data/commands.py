import click
import nltk


@click.command()
def nltk_stopwords():
    """download the nltk stopwords"""

    nltk.download("stopwords")
