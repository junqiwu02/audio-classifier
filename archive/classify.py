# %%
import preprocess
import analysis

from tensorflow import keras
from tensorflow.keras import layers

# %%
data = preprocess.Preprocess()
# X_train, X_test, Y_train, Y_test = data.get_train_test('vggish', 1000)
X_train, y_train = data.get_data('./data/train_sent_emo.csv', './data/train_splits/wav', 100)

# %%
# X_train, Y_train =  data.resample(X_train, Y_train, 512)
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
model.fit(X_train, y_train, batch_size=32, epochs=100)

# %%
# Y_pred = model.predict(X_test)
# analysis.show_stats(Y_test, Y_pred)
# %%
X_test, y_test = data.get_data('./data/test_sent_emo.csv', './data/output_repeated_splits_test/wav', 100)

# %%
analysis.show_stats(y_test, model.predict(X_test))