pyenv activate kaggle

kaggle competitions download -c nlp-getting-started
kaggle competitions     submit nlp-getting-started   -f submission.csv  -m "sklearn_classify"
