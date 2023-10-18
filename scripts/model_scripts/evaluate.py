import os
import sys
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model tokenizer\n")
    sys.exit(1)

test_df = pd.read_csv(sys.argv[1])


with open(sys.argv[3], 'rb') as f:
    tfidf = pickle.load(f)
    X = tfidf.transform(test_df["pre_text"])


with open(sys.argv[2], "rb") as fd:
    clf = pickle.load(fd)

score = clf.score(X, test_df["Score"])


prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)


with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)