from nltk.corpus import stopwords
import sys
import os
import io
import nltk
import pandas as pd
nltk.download("stopwords")

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 remove_stopwords.py data-file\n")
    sys.exit(1)


f_input = sys.argv[1]
f_output = os.path.join("data", "stage2", "train.pkl")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)


train_df = pd.read_pickle(f_input)

stopwords = stopwords.words("russian")


def find_del_stop_words(tokens):
    '''Поиск и удаление стоп-слов'''
    filtred_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(filtred_tokens)


train_df['pre_text'] = train_df['Key'].map(find_del_stop_words)
train_df.to_pickle(f_output)