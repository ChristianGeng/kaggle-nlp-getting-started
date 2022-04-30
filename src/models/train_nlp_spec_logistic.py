import re
import string

import numpy as np

import pandas as pd
import logging

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn import linear_model
from sklearn import pipeline

from utils.utils import get_project_root
from utils.utils import get_test_df
from utils.utils import get_train_df
from utils.utils import write_submission_csv

PROJECT_ROOT = get_project_root()


MODEL_NAME = "nlp_spec_logistic"

logger = logging.getLogger(__name__)
logger.info("making final data set from raw data")
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    # remove stock market tickers like $GE
    tweet = re.sub(r"\$\w*", "", tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # remove hyperlinks
    tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r"#", "", tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (
            word not in stopwords_english
            and word not in string.punctuation  # remove stopwords
        ):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(tweet, freqs):
    """
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    """
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # loop through each word in the list of words
    for word in word_l:

        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)

    ### END CODE HERE ###
    assert x.shape == (1, 3)
    return x


# sklearn_classify
df = get_train_df()
dft = get_test_df()

print("cleaning data")
# df["text"] = df["text"].apply(clean)
freqs = build_freqs(df["text"], df["target"])

# dft['text']
# this applies feature extraction tequnique to first tweet
extract_features(dft["text"][1], freqs)

# this is what we will use to make the submission
X_train = df["text"].apply(lambda x: extract_features(x, freqs)[0]).values
X_test = dft["text"].apply(lambda x: extract_features(x, freqs))

X_train = pd.DataFrame.from_records(
    df["text"].apply(lambda x: extract_features(x, freqs)[0])
)


X_test = pd.DataFrame.from_records(
    dft["text"].apply(lambda x: extract_features(x, freqs)[0])
)

dft["text"].apply(lambda x: extract_features(x, freqs))

reg = linear_model.LogisticRegression(fit_intercept=False).fit(X_train, df["target"])

# reg.coef_
# reg.intercept

# id,target
#0,1
# 2,1
# 3,1
# 9

y_test = reg.predict(X_test)

# submission = pd.DataFrame()
df_submission = pd.DataFrame(y_test, index=dft['id'], columns=['target'])
df_submission.index.name  = 'id'
#df_submission.columns = ['id', 'target']
# df_submission.to_csv("submission.csv", index=True, sep=',')
write_submission_csv(df_submission, MODEL_NAME, write_index=True)
# x = extract_features(tweet,freqs)
