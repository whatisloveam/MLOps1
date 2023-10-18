import sys
import os
import io
import gdown


f_output = os.path.join("data", "raw", "train_raw.csv")
os.makedirs(os.path.join("data", "raw"), exist_ok=True)


url = 'https://drive.google.com/uc?id=1-0uRwm89-ASZ7xjm6qmZYzp8E8rWkZT_'
gdown.download(url, f_output, quiet=False)