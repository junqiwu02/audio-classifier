# %%
from preprocess import *
import analysis

from tensorflow import keras
from tensorflow.keras.layers import InputLayer, LSTM, Dense

# %%
# data = Preprocess()
# X_train, y_train = data.get_data('./data/train_sent_emo.csv', './data/train_splits/wav')

# %%
X_train = pickle_load('./data/x_train_vggish.pkl')

# %%
y_train = pickle_load('./data/y_train_vggish.pkl')

# %%
# sample_weight = get_sample_weight(y_train)
X_train, y_train = resample(X_train, y_train)

# %%
y_train = one_hot(y_train)

# %%
model = keras.Sequential()
model.add(InputLayer(input_shape=(None, 512)))
model.add(LSTM(128, return_sequences=True, activation="tanh"))
model.add(LSTM(128, activation="tanh"))
model.add(Dense(7))
print(model.summary())

# %%
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# %%
# model.fit(X_train, y_train, sample_weight=sample_weight, batch_size=50, epochs=100)
model.fit(X_train, y_train, batch_size=64, epochs=100)

# %%
# X_test, y_test = data.get_data('./data/test_sent_emo.csv', './data/output_repeated_splits_test/wav')

# %%
X_test = pickle_load('./data/x_test_vggish.pkl')
y_test = pickle_load('./data/y_test_vggish.pkl')

# %%
y_test = one_hot(y_test)

# %%
analysis.show_stats(y_test, model.predict(X_test))

# %%
