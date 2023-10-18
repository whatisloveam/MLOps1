import sys
import os
import yaml
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py data-file model tokenizer \n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = sys.argv[2]
os.makedirs(os.path.join("models"), exist_ok=True)
params = yaml.safe_load(open("params.yaml"))["train"]
c = params["C"]
gamma = params["gamma"]
kernel = params["kernel"]

train_df = pd.read_csv(f_input)


with open(sys.argv[3], 'rb') as f:
    tfidf = pickle.load(f)
    X = tfidf.transform(train_df["pre_text"])


svc = SVC(C=c, gamma=gamma, kernel=kernel)
svc.fit(X, train_df['Score'])

with open(f_output, "wb") as fd:
    pickle.dump(svc, fd)