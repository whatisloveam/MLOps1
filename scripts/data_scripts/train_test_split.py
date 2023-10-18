import yaml
import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

params = yaml.safe_load(open("params.yaml"))["split"]


if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython3 train_test_split.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output_train = os.path.join("data", "stage3", "train.csv")
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)
f_output_test = os.path.join("data", "stage3", "test.csv")
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)
os.makedirs('tokenizer', exist_ok=True)

p_split_ratio = params["split_ratio"]

df = pd.read_pickle(f_input)

X = df['pre_text']
y = df['Score']

tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 3))
tfidf.fit(X)
with open('tokenizer\\tokenizer.pkl', "wb") as fd:
    pickle.dump(tfidf, fd)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=p_split_ratio, stratify=y)

pd.concat([X_train, y_train], axis=1).to_csv(
    f_output_train, index=None)
pd.concat([X_test, y_test], axis=1).to_csv(
    f_output_test, index=None)