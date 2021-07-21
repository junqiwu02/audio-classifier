# %%
from preprocess import Preprocess
import analysis

from tensorflow import keras
from tensorflow.keras import layers
import pickle

# %%
# data = Preprocess()
# X_train, y_train = data.get_data('./data/train_sent_emo.csv', './data/train_splits/wav')

# %%
# with open('./data/x_train_vggish.pkl', 'wb') as X_train_pickle:
#     pickle.dump(X_train, X_train_pickle)
# with open('./data/y_train_vggish.pkl', 'wb') as y_train_pickle:
#     pickle.dump(y_train, y_train_pickle)

# %%
X_train, y_train = data.resample(X_train, y_train)
y_train = data.one_hot(y_train)

# %%
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(None, 512)))
model.add(layers.LSTM(128, return_sequences=True, activation="tanh"))
model.add(layers.LSTM(128))
model.add(layers.Dense(7))
print(model.summary())

# %%
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# %%
model.fit(X_train, y_train, batch_size=64, epochs=100)

# %%
# X_test, y_test = data.get_data('./data/test_sent_emo.csv', './data/output_repeated_splits_test/wav')

# %%
# with open('./data/x_test_vggish.pkl', 'wb') as X_test_pickle:
#     pickle.dump(X_test, X_test_pickle)
# with open('./data/y_test_vggish.pkl', 'wb') as y_test_pickle:
#     pickle.dump(y_test, y_test_pickle)

# %%
y_test = data.one_hot(y_test)

# %%
analysis.show_stats(y_test, model.predict(X_test))

# %%
