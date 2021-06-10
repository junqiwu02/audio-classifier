# %%
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import librosa.display
# %%
(rate,sig) = wav.read("dataset/male.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])
# %%
x, sr = librosa.load("dataset/male.wav")
librosa.display.waveplot(x, sr=sr)
# %%
#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
# %%
