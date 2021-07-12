# %%
from preprocess import Preprocess
from analysis import *

from tensorflow import keras
from tensorflow.keras import layers

# %%
data = Preprocess()
X_train, X_test, Y_train, Y_test = data.get_train_test('mfcc', 1000)

# %%
model = keras.Sequential()
model.add(keras.Input(shape=(None, 13)))
model.add(layers.LSTM(128, return_sequences=True, activation="relu"))
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
model.fit(X_train, Y_train, batch_size=64, epochs=5)

# %%
Y_pred = model.predict(X_test)
show_stats(Y_test, Y_pred)
# %%
