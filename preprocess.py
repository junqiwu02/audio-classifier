import numpy as np
import pandas as pd

import glob
import natsort
from tqdm.notebook import tqdm

import scipy.io.wavfile as wav
from python_speech_features import mfcc, logfbank

from sklearn.model_selection import train_test_split

class Preprocess:

    def __init__(self):
        # list of wav files
        self.wav_files = natsort.os_sorted(glob.glob('./data/dev_splits_complete/wav/*.wav'))
        # emotion one-hot labels
        df = pd.read_csv('./data/dev_sent_emo.csv')
        self.one_hot = pd.get_dummies(df['Emotion']).values

    def get_train_test(self, feature_type, num_inputs):
        feature_dim = 0
        if feature_type == 'mfcc':
            feature_dim = 13
        elif feature_type == 'logfbank':
            feature_dim = 26
        else:
            print('Invalid feature type')
            return

        seqs = []
        for f in tqdm(self.wav_files[:num_inputs]):
            rate, sig = wav.read(f)

            feat = None
            if feature_type == 'mfcc':
                feat = mfcc(sig, rate, nfft=1200)
            elif feature_type == 'logfbank':
                feat = logfbank(sig, rate, nfft=1200)

            seqs.append(feat)

        # resize all sequences to the avg length
        tensor_len = sum(map(len, seqs)) // len(seqs)

        resized = []
        for x in seqs:
            res = None
            if len(x) < tensor_len:
                res = np.zeros((tensor_len, feature_dim))
                res[:x.shape[0], :x.shape[1]] = x
            else:
                res = np.array(x[:tensor_len])
            resized.append(res)
        X = np.array(resized)

        # normalize
        X = X / np.linalg.norm(X)

        Y = self.one_hot[:num_inputs]

        return train_test_split(X, Y, test_size=0.2, random_state=42)
