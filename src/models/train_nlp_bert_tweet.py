# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from utils.utils import get_project_root, get_samplesubmission_df, get_test_df, get_train_df

# pip install datasets
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    default_data_collator,
    Trainer,
    set_seed,
)

# %env WANDB_DISABLED=True
os.environ["WANDB_DISABLED"] = "True"


import pandas as pd

# dataset = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
# dataset.head()
dataset = get_train_df()
df = dataset.drop(["keyword", "location"], axis=1)

df["text"] = df["text"].apply(lambda x: x.lower())
df["text"] = df["text"].apply(lambda x: x.replace("#", ""))
import string

print(string.punctuation)
str.maketrans(dict.fromkeys(string.punctuation))
df["text"] = df["text"].apply(
    lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation)))
)

df["text"] = df["text"].apply(
    lambda x: x.translate(str.maketrans("", "", string.digits))
)


for index, row in df.iterrows():
    text = []
    for word in row["text"].split():
        if "http" not in word:
            text.append(word)
    row["text"] = " ".join(text)
    df["text"][index] = row["text"]


set_seed(42)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForSequenceClassification.from_pretrained("bert-large-uncased")


maxx, minn = 0, 1000000
for index, row in df.iterrows():
    if len(row["text"].split()) > maxx:
        maxx = len(row["text"].split())
    if len(row["text"].split()) < minn:
        minn = len(row["text"].split())
maxx, minn

max_length = 64
df["input_ids"] = df["text"].apply(
    lambda x: tokenizer(x, max_length=max_length, padding="max_length")["input_ids"]
)
df.head(n=10)


df.rename(columns={"target": "labels"}, inplace=True)

df = df[["input_ids", "labels"]]

train_df = df[: -int(len(df) * 0.01)].reset_index(drop=True)
test_df = df[-int(len(df) * 0.01) :].reset_index(drop=True)

len(train_df), len(test_df)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)


# batch_size = 16 - from original
batch_size = 1

args = TrainingArguments(
    "nlp-getting-started",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

data_collator = default_data_collator
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# --------
# achtung
# --------

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.evaluate()


# df_submission = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_submission = get_test_df()

df_submission.head()
df_submission["text"] = df_submission["text"].apply(lambda x: x.lower())
df_submission["text"] = df_submission["text"].apply(lambda x: x.replace("#", ""))
df_submission["text"] = df_submission["text"].apply(
    lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation)))
)
df_submission["text"] = df_submission["text"].apply(
    lambda x: x.translate(str.maketrans("", "", string.digits))
)
for index, row in df_submission.iterrows():
    text = []
    for word in row["text"].split():
        if "http" not in word:
            text.append(word)
    row["text"] = " ".join(text)
    # print(row['text'])
    df_submission["text"][index] = row["text"]

df_submission["input_ids"] = df_submission["text"].apply(
    lambda x: tokenizer(x, max_length=max_length, padding="max_length")["input_ids"]
)
df_submission.head()

df_submission=df_submission[['input_ids']]

output_ds = Dataset.from_pandas(df_submission)
output_ds

outputs=trainer.predict(output_ds)
outputs.predictions.argmax(1)

from utils.utils import get_samplesubmission_df
#sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sub = get_samplesubmission_df()
#sub.head()

# sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sub.head()
sub['target'] = outputs.predictions.argmax(1)
sub.head()
import os
model_name = "nlp_bert_tweet"
submission_fname = os.path.join(get_project_roots(), "submissions", 'submission_' +
                                model_name + '.csv')

sub.to_csv(submission_fname, index=False)
