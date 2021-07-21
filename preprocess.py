import numpy as np
import pandas as pd

import os.path
from tqdm.notebook import tqdm

import vggish_keras as vgk

from imblearn.over_sampling import RandomOverSampler

RAND_STATE = 42
LABELS = {
    'neutral': 0,
    'surprise': 1,
    'fear': 2,
    'sadness': 3,
    'joy': 4,
    'disgust': 5,
    'anger': 6
}
FEATURE_DIM = 512
VGGISH_DUR = 0.05
VGGISH_HOP = 0.05

class Preprocess:

    def __init__(self):
        self.vggish = vgk.get_embedding_function(duration=VGGISH_DUR, hop_duration=VGGISH_HOP)

    def get_data(self, csv_path, wav_path, max_inputs=None):
        df = pd.read_csv(csv_path)
        num_inputs = df.shape[0] if not max_inputs else min(max_inputs, df.shape[0])

        X = []
        y = []
        for _, emo, dia, utt in tqdm(zip(range(num_inputs), df['Emotion'], df['Dialogue_ID'], df['Utterance_ID']), total=num_inputs):
            file = f'{wav_path}/dia{dia}_utt{utt}.wav'

            try:
                _, feat = self.vggish(file)
                X.append(feat)
                y.append(LABELS[emo])
            except FileNotFoundError:
                print(f'{file} not found! Skipping...')

        # resize all sequences to the avg length
        tensor_len = sum(map(len, X)) // len(X)

        resized = []
        for x in X:
            res = None
            if len(x) < tensor_len:
                res = np.zeros((tensor_len, FEATURE_DIM))
                res[:x.shape[0], :x.shape[1]] = x
            else:
                res = np.array(x[:tensor_len])
            resized.append(res)
        X = np.array(resized)

        # normalize
        X = X / np.linalg.norm(X)

        return X, y

    def resample(self, X, y):
        # flatten
        X = X.reshape(len(X), -1)

        X, y = RandomOverSampler(random_state=RAND_STATE).fit_resample(X, y)
        X = X.reshape(len(X), -1, FEATURE_DIM)

        return X, y

    def one_hot(self, y):
        Y = np.zeros((len(y), len(LABELS)))
        Y[np.arange(len(y)), y] = 1

        return Y

