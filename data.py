import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler
import joblib
import shutil
import os
import zipfile
import sys

from Lib.config import lookback, per_news_length, news_length, price_length, news_embedding_dim, max_words, nb_labels

# DATA_PATH = sys.argv[1]
# shutil.unpack_archive(DATA_PATH, extract_dir=DATA_DIR)
data = pd.read_csv(os.path.join("./Data", "All_new.csv"))

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,2:7])) # 5 * 50

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(headlines) # [20015] 查找表 word dict

def news_to_vec(data, start_index, end_index):
    pads = []
    for i in range(start_index, end_index+1):
        sequences_data = tokenizer.texts_to_sequences(data.iloc[:, i])
        padded_data_seqs = sequence.pad_sequences(sequences_data, maxlen=per_news_length) # [1989, 50]
        pads.append(padded_data_seqs)
    return np.hstack(pads) # [1989, 5*50]

def lookback_price(data, lb): # [1989, 15*4]
    X = []
    for i in range(len(data) - lb - 1):
        X.append(data[i:(i+lb), :])
    return np.array(X) # [1981, 7, 60]


def lookback_news(data, lb): # [1989, 5*50], 7
    X = []
    for i in range(lb):
        X.append([])
    for i in range(len(data) - lb - 1):
        for j in range(0, lb):
            X[j].append(data[i + j, :])
    for i in range(lb):
        X[i] = np.array(X[i])
    return X

def lookback_label(data, lb):
    X = []
    for i in range(len(data) - lb - 1):
        X.append(data[i+lb, :])
    return np.array(X)

news_data = news_to_vec(data, 2, 6) # [1989, 5*50]
x_news = lookback_news(news_data, lookback) # 7 * [1981, 250] 七天所有新闻，共 1981 个样本

# ## Data Processing

price_scl = MinMaxScaler()
price_data = data.iloc[:, 7:67].values # [1989, 15*4]
price_data = price_scl.fit_transform(price_data)
x_price = lookback_price(price_data, lookback) # [1981, 7, 60]


label_scl = MinMaxScaler()
label_data = data.iloc[:, 67:82].values # [1989, 15]
label_data = label_scl.fit_transform(label_data)
y = lookback_label(label_data, 7) # [1981, 15]

n_samples = y.shape[0] # 1981
p = int(n_samples * 0.8) # 1584
q = p + int(n_samples * 0.1) # 1782

x_news_train = [] # 7 * [1584, 250]
x_news_valid = [] # 7 * [198, 250]
x_news_test = [] # 7 * [199, 250]
x_news_3_valid=[] # 7 * [3, 250]
for i in range(lookback):
    x_news_train.append(x_news[i][:p])
    x_news_valid.append(x_news[i][p:q])
    x_news_3_valid.append(x_news[i][p:p+3])
    x_news_test.append(x_news[i][q:])

x_price_train = x_price[:p] # [1584, 7, 60]
x_price_valid = x_price[p:q] # [198, 7, 60]
x_price_3_valid=x_price[p:p+3] # [3, 7, 60]
x_price_test = x_price[q:] # [199, 7, 60]

y_train = y[:p] # [1584, 15]
y_valid = y[p:q] # [198, 15]
y_3_valid= y[p:p+3] # [3, 15]
y_test = y[q:] # [199, 15]

TEMP_DATA_DIR= os.path.join("./Tmp")
if not os.path.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)

#ToDo: Unzip code  shutil.unpack_archive("Data/Train/Under_10_min_training/data.zip",extract_dir="temp")
DATA_DIR = os.path.join("./Data", "Train", "Under_10_min_training")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(os.path.join(DATA_DIR, "x_news_train.npy"), x_news_train)
np.save(os.path.join(DATA_DIR, "x_price_train.npy"), x_price_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)

#ToDo: Unzip code  shutil.unpack_archive("Data/Train/Under_90_min_tuning/data.zip",extract_dir="temp")
DATA_DIR = os.path.join("./Data", "Train", "Under_90_min_tuning")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(os.path.join(DATA_DIR, "x_news_train.npy"), x_news_train)
np.save(os.path.join(DATA_DIR, "x_price_train.npy"), x_price_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)

#ToDo: Unzip code  shutil.unpack_archive("Data/Train/Best_hyperparameter_80_percent/data.zip",extract_dir="temp")
DATA_DIR = os.path.join("./Data", "Train", "Best_hyperparameter_80_percent")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(os.path.join(DATA_DIR, "x_news_train.npy"), x_news_train)
np.save(os.path.join(DATA_DIR, "x_price_train.npy"), x_price_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)


#ToDo:Unzip code  shutil.unpack_archive("Data/Validation/3_samples/data.zip",extract_dir="temp")
#TODO:Needs check
DATA_DIR = os.path.join("./Data", "Validation", "3_samples")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(os.path.join(DATA_DIR, "x_news_3_valid.npy"), x_news_3_valid)
np.save(os.path.join(DATA_DIR, "x_price_3_valid.npy"), x_price_3_valid)
np.save(os.path.join(DATA_DIR, "y_3_valid.npy"), y_3_valid)

#**********************************************************************************

#ToDO: Unzip code  shutil.unpack_archive("Data/Validation/Validation_10_percent/data.zip",extract_dir="temp")
DATA_DIR = os.path.join("./Data", "Validation", "Validation_10_percent")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(os.path.join(DATA_DIR, "x_news_valid.npy"), x_news_valid)
np.save(os.path.join(DATA_DIR, "x_price_valid.npy"), x_price_valid)
np.save(os.path.join(DATA_DIR, "y_valid.npy"), y_valid)

#ToDO:Unzip code  shutil.unpack_archive("Data/Test/Test_10_percent/data.zip",extract_dir="temp")
DATA_DIR = os.path.join("./Data", "Test", "Test_10_percent")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(os.path.join(DATA_DIR, "x_news_test.npy"), x_news_test)
np.save(os.path.join(DATA_DIR, "x_price_test.npy"), x_price_test)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

# -----------------------
if not os.path.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)
joblib.dump(price_scl, os.path.join(TEMP_DATA_DIR, 'price_scl.scl'))
joblib.dump(label_scl, os.path.join(TEMP_DATA_DIR, 'label_scl.scl'))

# os.remove('Data/All_new.csv')

