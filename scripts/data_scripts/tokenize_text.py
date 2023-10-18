import sys
import os
import io
import re
import pandas as pd
from nltk.tokenize import word_tokenize


if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 tokenize_text.py data-file\n")
    sys.exit(1)


f_input = sys.argv[1]
f_output = os.path.join("data", "stage1", "train.pkl")
os.makedirs(os.path.join("data", "stage1"), exist_ok=True)


train_df = pd.read_csv(f_input, index_col=0)


def del_mail(text):
    mail = re.compile(r'^([a-z0-9_\.-]+)@([a-z0-9_\.-]+)\.([a-z\.]{2,6})&')
    return mail.sub(r' ', str(text))


def del_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def del_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def tokenize(text):
    text = del_html(text)
    text = del_URL(text)
    text = del_mail(text)
    text = text.lower().replace("ё", "е")
    text = re.sub("[^А-яа-яЁёЙй]", " ", text)
    text = re.sub("\s+", " ", text)
    tokens_ = word_tokenize(text)
    tokens = []
    for token in tokens_:
        tokens.append(token)

    return tokens


train_df['Key'] = train_df['Text'].map(tokenize)
train_df.to_pickle(f_output)