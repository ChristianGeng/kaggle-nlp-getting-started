from transformers import BertTokenizer

MODEL_NAME = "pytorch_transformers"


from models.train_pytorch_transformers import BertClassifier
from models.train_pytorch_transformers import Dataset

from utils.utils import get_project_root
from utils.utils import get_test_df
from utils.utils import get_train_df

import audeer


import os
import torch

MODEL_NAME = "pytorch_transformers"

MODEL_DIR = os.path.join(get_project_root(), "models")
MODEL_DIR = audeer.safe_path(MODEL_DIR)

epoch_num = 0
ckpt_fname = os.path.join(MODEL_DIR, MODEL_NAME + '_' +str(epoch_num).zfill(4) +  ".pth")


if not os.path.exists(ckpt_fname):
    raise FileNotFoundError(ckpt_fname)

checkpoint = torch.load(ckpt_fname)
model = BertClassifier()
model.load_state_dict(checkpoint['model'])


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class DatasetTest(torch.utils.data.Dataset):

    def __init__(self, df):

        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        # batch_y = self.get_batch_labels(idx)

        return batch_texts


test_data = get_test_df()
test = DatasetTest(test_data)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    model = model.cuda()

total_acc_test = 0


test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)


results = []

idx = 0

with torch.no_grad():
    # returns no label
    for test_input in test_dataloader:

        # test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        result = output.argmax(dim=1)
        results.append(result)
        # print(f"processing {idx} out of {test_data.shape[0]}")
        idx += 1
        # int(output.argmax(dim=1))
        # acc = (output.argmax(dim=1) == test_label).sum().item()
        # total_acc_test += acc
breakpoint()


predictions = [int(x) for x in results]

import pandas as pd
submission_df = pd.DataFrame({"id": test_data["id"], "target": predictions})
# submission = pd.DataFrame(submission_df)

audeer.mkdir(os.path.join(get_project_root(), "submissions"))
submission_fname = os.path.join(get_project_root(), "submissions", "submission_" + MODEL_NAME + ".csv")

logger.info(f"writing submissions to {submission_fname}.")
submission_df.to_csv(submission_fname, index=False)
logger.info("done.")
