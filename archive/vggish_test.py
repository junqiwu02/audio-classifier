# %%
import librosa
import numpy as np
import vggish_keras as vgk

# %%
# loads the model once and provides a simple function that takes in `filename` or `y, sr`
compute = vgk.get_embedding_function(duration=0.25, hop_duration=0.25)

# %%
# compute from filename
ts, Z = compute('./data/dev_splits_complete/wav/dia0_utt1.wav')
# %%
for t in ts:
    print(t)
# %%
Z.shape
# %%
