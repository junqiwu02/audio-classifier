# %%
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import glob
import natsort
from tqdm.notebook import tqdm

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import pandas as pd

# %%
NUM_INPUTS = 1000
# %%
# Read files and extract mfcc
filelist = glob.glob('dataset/dev_splits_complete/wav/*.wav')
mfccs = []
for f in tqdm(natsort.os_sorted(filelist)[:NUM_INPUTS]):
    rate, sig = wav.read(f)
    mfcc_feat = mfcc(sig,rate, nfft=1200)
    mfccs.append(mfcc_feat)
# %%
# Pad shorter sequences with 0s and cut longer sequences
tensor_len = sum(map(len, mfccs)) // len(mfccs)

resized = []
for x in mfccs:
    res = None
    if len(x) < tensor_len:
        res = np.zeros((tensor_len, 13))
        res[:x.shape[0], :x.shape[1]] = x
    else:
        res = np.array(x[:tensor_len])
    resized.append(res)
X = np.array(resized)
X.shape
# %%
# Normalize
norm = np.linalg.norm(X)
X = X/norm

# %%
df = pd.read_csv('dataset/dev_sent_emo.csv')
df
# %%
one_hot = pd.get_dummies(df['Emotion'])
one_hot
# %%
y = one_hot.values[:NUM_INPUTS]
y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape


# %%
model = keras.Sequential()
model.add(keras.Input(shape=(None, 13)))
model.add(layers.LSTM(128, return_sequences=True, activation="relu"))
model.add(layers.LSTM(128))
model.add(layers.Dense(7))
model.summary()

# %%
print(model.summary())
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
# %%
model.fit(X_train, y_train, batch_size=32, epochs=5)

# %%
model.evaluate(X_test, y_test)
# 38.5% Testing accuracy in 1000 samples
# %%
# %%
