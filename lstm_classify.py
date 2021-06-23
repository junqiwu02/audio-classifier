# %%
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import glob
import natsort
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import pandas as pd

# %%
# Read files and extract mfcc
filelist = glob.glob('dataset/dev_splits_complete/wav/*.wav')
x_list = []
for f in tqdm(natsort.os_sorted(filelist)[:100]):
    rate, sig = wav.read(f)
    mfcc_feat = mfcc(sig,rate, nfft=1200)
    x_list.append(mfcc_feat)
# %%
# Pad shorter sequences with 0s
length = max(map(len, x_list))
print(length)
x_list_padded = []
for xi in x_list:
    res = np.zeros((length, 13))
    res[:xi.shape[0], :xi.shape[1]] = xi
    x_list_padded.append(res)
x = np.array(x_list_padded)
x.shape

# %%
# Clear mem
x_list = None
x_list_padded = None

# %%
df = pd.read_csv('dataset/dev_sent_emo.csv')
df
# %%
one_hot = pd.get_dummies(df['Emotion'])
one_hot
# %%
y = one_hot.values[:100]
y.shape

# %%
model = keras.Sequential()
model.add(keras.Input(shape=(None, 13)))
model.add(layers.LSTM(128, return_sequences=True, activation="relu"))
model.add(layers.LSTM(128))
model.add(layers.Dense(7))
model.summary()
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape

# %%
# Clear mem
x = None

# %%
print(model.summary())
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
# %%
model.fit(x_train, y_train, batch_size=32, epochs=5)

# %%
