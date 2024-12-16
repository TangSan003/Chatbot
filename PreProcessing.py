import random
from idlelib.iomenu import encoding

import torch
import json
import pandas as pd
import missingno as msno

from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import seaborn as sns
from nltk import PorterStemmer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import TrainingArguments, Trainer, pipeline
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertTokenizerFast
from transformers import BertForSequenceClassification


filename = '/home/tangsan/KLTN/KLTNWEB/Chatbot/dataset/intents.json'
# Load the data¶
def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file


# Extract Info from the Json data file and Store it in dataframe¶
def create_df():
    df = pd.DataFrame({
        'Pattern': [],
        'Tag': []
    })
    return df


def extract_json_info(json_file, df):
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag
    return df


intents = load_json_file(filename)
df = create_df()
df = extract_json_info(intents, df)

df2 = df.copy()
df2.head()
# print(df2.head())

# Check the shape of the dataset
def print_shape_df(df, ds_name="df"):
    print(f"{ds_name} dataset has {df.shape[0]} rows and {df.shape[1]} columns")
# print_shape_df(df, "Chatbot")


# Display information about the dataset
def print_dfInfo(df, ds_name="df"):
    print(f"The info of {ds_name} dataset\n")
    print(df.info())
# print_dfInfo(df, "Chatbot")


# Display Number of Classes
def num_classes(df, target_col, ds_name="df"):
    print(f"The {ds_name} dataset has {len(df[target_col].unique())} classes")
# num_classes(df, 'Tag', "Chatbot")


# Check the null values in the dataset
def check_null(df, ds_name='df'):
    print(f"Null Values in each col in the {ds_name} dataset:\n")
    print(df.isnull().sum())
# check_null(df, "Chatbot")


# Visualize the Null Values
def plot_miss_vals(df, ds_name="df"):
    msno.matrix(df)
    plt.title(f'Distribution of Missing Values in {ds_name} dataset', fontsize=30, fontstyle='oblique')
    plt.show()
# plot_miss_vals(df, "Chatbot")


# Visualize the distrbution of our Classes
def count_plot(x, df, title, xlabel, ylabel, width, height, order = None, rotation=False, palette='winter', hue=None):
    ncount = len(df)
    plt.figure(figsize=(width,height))
    ax = sns.countplot(x = x, palette=palette, order = order, hue=hue)
    plt.title(title, fontsize=20)
    if rotation:
        plt.xticks(rotation = 'vertical')
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)

    ax.yaxis.set_label_position('left')
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), ha='center', va='bottom') # set the alignment of the text

    plt.show()

sns.set(font_scale = 0.8)
order = df['Tag'].value_counts().index


stemmer = PorterStemmer()
ignore_words=['?', '!', ',', '.']

def preprocess_pattern(pattern):
    words = word_tokenize(pattern.lower())
    stemmed_words = [stemmer.stem(word) for word in words if word not in ignore_words]
    return " ".join(stemmed_words)
# df['Pattern'] = df['Pattern'].apply(preprocess_pattern)
# print(df.head())
# print("---------------------------------")
# print(df['Pattern'].isnull().sum())


# Display Word Cloud
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=STOPWORDS).generate(' '.join(df['Pattern']))
# plt.imshow(wc, interpolation='bilinear')
# plt.xticks([])
# plt.yticks([])
# plt.show()

# Distrbution of Number of letters in each text
letters_len=df['Pattern'].str.len()
# plt.hist(letters_len,color='red')
# plt.title("Distrbution of Number of letters in each text")
# plt.xlabel("Num of letters")
# plt.ylabel("Frequency")
# plt.show()


# Visualize top words
def get_corpus(series):
    words = []
    for text in series:
        for word in text.split():
            words.append(word.strip())
    return words

corpus = get_corpus(df.Pattern)
# print(corpus[:15])
# print(f"dataset contains {len(corpus)} words")


counter = Counter(corpus)
most_common = counter.most_common(10)
most_common = dict(most_common)
# print(most_common)


def get_top_text_ngrams(corpus, n,g):
    vec = CountVectorizer(ngram_range=(1, 1)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
# plt.figure(figsize = (16,9))
# most_common_uni = get_top_text_ngrams(df.Pattern,10,1)
# most_common_uni = dict(most_common_uni)
# sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))
# plt.title("Top 10 Unigrams")
# plt.show()


# Data Preprocessing
def input_model(df):
    df['Tag'] = df['Tag'].apply(lambda x: x.strip())
    labels = df['Tag'].unique().tolist()
    labels = [s.strip() for s in labels]

    num_labels = len(labels)
    id2label = {id:label for id, label in enumerate(labels)}
    label2id = {label:id for id, label in enumerate(labels)}

    df['labels'] = df['Tag'].map(lambda x: label2id[x.strip()])
    return num_labels, id2label, label2id
num_labels, id2label, label2id = input_model(df)

df2['labels'] = df2['Tag'].map(lambda x: label2id[x.strip()])


# Split the data into train and test
X = list(df2['Pattern'])
y = list(df2['labels'])

# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 47)
#
#
# # Load BERT Pretrained model and Tokenizer
# model_name = "bert-base-uncased"
# max_len = 256
#
# tokenizer = BertTokenizer.from_pretrained(model_name,
#                                           max_length=max_len)
# model = BertForSequenceClassification.from_pretrained(model_name,
#                                                       num_labels=num_labels,
#                                                       id2label=id2label,
#                                                       label2id = label2id)
#
# # print(model)
# # Transform the data into numerical format
# train_encoding = tokenizer(X_train, truncation=True, padding=True)
# test_encoding = tokenizer(X_test, truncation=True, padding=True)
#
# full_data = tokenizer(X, truncation=True, padding=True)
#
#
# # Build Data Loader
# class DataLoader(Dataset):
#
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         # print("encodding_pre \n",self.encodings.items())
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#
#         return item
#
#     def __len__(self):
#         return len(self.labels)
# train_dataloader = DataLoader(train_encoding, y_train)
# test_dataloader = DataLoader(test_encoding, y_test)
#
#
# fullDataLoader = DataLoader(full_data, y_test)
#
#
#
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)
#
#     return {
#         'Accuracy': acc,
#         'F1': f1,
#         'Precision': precision,
#         'Recall': recall
#     }
# training_args = TrainingArguments(
#     output_dir='output',
#     num_train_epochs=100,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=16,
#     warmup_steps=100,
#     weight_decay=0.05,
#     logging_steps=50,
#     evaluation_strategy="steps",
#     eval_steps=50,
#     save_strategy="steps",
#     load_best_model_at_end=True
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataloader,
#     eval_dataset=test_dataloader,
#     compute_metrics= compute_metrics
# )
#
# trainer.train()
#
#
# q=[trainer.evaluate(eval_dataset=df2) for df2 in [train_dataloader, test_dataloader]]
#
# pd.DataFrame(q, index=["train","test"]).iloc[:,:5]
#
# model_path = "/home/tangsan/KLTN/KLTNWEB/Chatbot/chatbot"
# trainer.save_model(model_path)
# tokenizer.save_pretrained(model_path)
