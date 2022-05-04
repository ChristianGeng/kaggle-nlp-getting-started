"""https://towardsdataschedience.com/text-classification-with-bert-in-pytorch-887965e5820f
"""

import pandas as pd
import logging
import os


import torch

from torch.optim import Adam
from tqdm import tqdm

import re

from torch import nn
from transformers import BertModel


import audeer
import joblib
import numpy as np
import pandas as pd

from utils.utils import get_project_root
from utils.utils import get_test_df
from utils.utils import get_train_df

from transformers import BertTokenizer

PROJECT_ROOT = get_project_root()
MODEL_NAME = "pytorch_transformers"

MODEL_DIR = os.path.join(get_project_root(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


logger = logging.getLogger(__name__)
logger.info("making final data set from raw data")
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)


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


# If you have datasets from different languages,
# you might want to use bert-base-multilingual-cased.
# Specifically, if your dataset is in German, Dutch, Chinese, Japanese, or Finnish,
# you might want to use a tokenizer pre-trained specifically in these languages.
# You can check the name of the corresponding pre-trained tokenizer at https://huggingface.co/models
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def explain_tokenizer():

    example_text = "I will watch Memento tonight"

    # padding : to pad each sequence to the maximum length that you specify.
    # max_length : the maximum length of each sequence.
    # In this example we use 10, but for our actual dataset we will use 512,
    # which is the maximum length of a sequence allowed for BERT
    # truncation : if True, the tokens in each sequence that exceed the maximum
    # length will be truncated.
    # return_tensors : the type of tensors that will be returned.
    # Since we’re using Pytorch, then we use pt. If you use Tensorflow, then you need to use tf .
    bert_input = tokenizer(
        example_text,
        padding="max_length",
        max_length=10,
        truncation=True,
        return_tensors="pt",
    )

    # tokenizer.decode(bert_input.input_ids[0])
    # [CLS] Our Deeds are the Reason of this [SEP]

    # bert_input.attention_mask
    # binary mask that identifies whether a token is a real word or just padding.
    # If the token contains [CLS], [SEP], or any real word, then the mask would be 1.
    # Meanwhile, if the token is just padding or [PAD], then the mask would be 0.

    # bert_input.token_type_ids
    # binary mask that identifies in which sequence a token belongs.
    # If we only have a single sequence, then all of the token type ids will be 0


class Dataset(torch.utils.data.Dataset):
    """Dataset class that will serve as a class to generate data."""

    def __init__(self, df):

        # self.labels = [labels[label] for label in df['category']]
        self.labels = df["target"].values
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):
    """Model using a pre-trained BERT base model which has 12 layers of Transformer encoder."""

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        # If your dataset is not in English, it would be best if you use bert-base-multilingual-cased model.
        # If your data is in German, Dutch, Chinese, Japanese, or Finnish,
        # you can use the model pre-trained specifically in these languages.
        # You can check the name of the corresponding pre-trained model
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        # At the end of the linear layer, we have a vector of size n categories,
        # each corresponds to a category of our labels
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        # _ contains the embedding vectors of all of the tokens in a sequence.
        # pooled_output, contains the embedding vector of [CLS] token.
        # For a text classification task, it is enough to use this embedding as an input for our classifier.
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

        checkpoint = dict()
        checkpoint["model"] = model.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["epoch"] = epoch_num
        checkpoint["train_loss"] = total_loss_train / len(train_data)
        checkpoint["train_acc"] = total_acc_train / len(train_data)
        checkpoint["val_loss"] = total_loss_val / len(val_data)
        checkpoint["val_acc"] = total_acc_val / len(val_data)
        os.path.join(PROJECT_ROOT)
        ckpt_fname = os.path.join(
            MODEL_DIR, MODEL_NAME + "_" + str(epoch_num).zfill(4) + ".pth"
        )
        logger.info(f"Writing checkpoiont to {ckpt_fname}")
        torch.save(checkpoint, ckpt_fname)


def main():

    df = get_train_df()
    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    print(len(df_train), len(df_val), len(df_test))
    EPOCHS = 5
    model = BertClassifier()

    LR = 1e-6

    train(model, df_train, df_val, LR, EPOCHS)


if __name__ == "__main__":
    main()
