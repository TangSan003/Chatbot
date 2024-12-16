from nltk.sem.chat80 import items
from torch.utils.data import Dataset, DataLoader

import torch
from PreProcessing import load_json_file, create_df, extract_json_info, input_model
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class Bot_Dataset(Dataset):
    def __init__(self, datapath, is_train, transform=None):
        intents = load_json_file(datapath)
        df = create_df()
        df = extract_json_info(intents, df)
        num_labels, id2label, label2id = input_model(df)
        # {label: id for id, label in enumerate(labels)}


        df['labels'] = df['Tag'].map(lambda x: label2id[x.strip()])

        X =list(df['Pattern'])
        y=list(df['labels'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

        if is_train:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test


    def __len__(self):
        return (len(self.y))
    def __getitem__(self, index):
        model_name = "bert-base-uncased"
        max_len = 256

        tokenizer = BertTokenizer.from_pretrained(model_name, max_length=max_len)
        encoding = tokenizer(self.X,truncation=True, padding=True)

        items = {key: torch.tensor(val[index]) for key, val in encoding.items()}
        items['labels'] = torch.tensor(self.y[index])

        return items

if __name__ == '__main__':
    print("-----------------Testing the dataset.py file-----------------")
    path = "../../../KLTN/KLTNWEB/dataset/intents.json"
    dataset = Bot_Dataset(path, is_train=False)
    print(dataset.__len__())
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    # print(dataset.__getitem__(0))
    for i in dataloader:
        print(i)
# attention_mask